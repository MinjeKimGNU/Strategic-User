import argparse
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from recbole.quick_start import load_data_and_model


CLUSTERING_CHOICES = ["kmeans", "len_based", "pop_based"]
INJECTION_CHOICES = ["no_injection", "target", "random", "pop"]
NUM_CLUSTERS = 4
TOPK_SIZE = 20
NUM_INJECT_VALUES = range(1, 6)
SAMPLE_SIZE = 100
SAMPLE_SEED = 42
RBO_P = 0.9
TOP_POP_RATIO = 0.2


def calc_batch_rbo(l1, l2, p=RBO_P):
    m, k = l1.shape
    device = l1.device
    rbo_scores = torch.zeros(m, device=device)
    for d in range(1, k + 1):
        l1_d = l1[:, :d].unsqueeze(2)
        l2_d = l2[:, :d].unsqueeze(1)
        matches = (l1_d == l2_d).float()
        intersection_size = matches.sum(dim=(1, 2))
        a_d = intersection_size / d
        weight = (1 - p) * (p ** (d - 1))
        rbo_scores += weight * a_d
    return rbo_scores


def get_item_embeddings(model, device):
    if hasattr(model, "item_embedding"):
        return model.item_embedding.weight.detach().to(device)
    return model.item_emb.weight.detach().to(device)


def extract_eval_tensors(model, test_data, device, item_embeddings):
    all_outputs = []
    all_seqs = []
    all_lengths = []
    all_top_k = []

    with torch.no_grad():
        for batched_data in test_data:
            interaction = batched_data[0] if isinstance(batched_data, (list, tuple)) else batched_data
            item_seq = interaction["item_id_list"]
            item_seq_len = interaction["item_length"].to(device)
            output = model.forward(item_seq.to(device), item_seq_len)
            scores = torch.matmul(output, item_embeddings.transpose(0, 1))
            _, topk_indices = torch.topk(scores, k=TOPK_SIZE, dim=-1)
            all_outputs.append(output.cpu())
            all_seqs.append(item_seq.cpu())
            all_lengths.append(item_seq_len.cpu())
            all_top_k.append(topk_indices.cpu())

    return (
        torch.cat(all_outputs, dim=0),
        torch.cat(all_seqs, dim=0),
        torch.cat(all_lengths, dim=0),
        torch.cat(all_top_k, dim=0),
    )


def build_cluster_labels(clustering_method, representations, sequences, seq_lengths, dataset, n_items):
    repr_np = representations.numpy()
    if clustering_method == "kmeans":
        num_clusters = NUM_CLUSTERS
        labels = KMeans(n_clusters=num_clusters, random_state=SAMPLE_SEED).fit_predict(repr_np)
        return labels, num_clusters

    if clustering_method == "len_based":
        lengths = seq_lengths.numpy()
        labels = np.array(pd.qcut(pd.Series(lengths).rank(method="first"), q=NUM_CLUSTERS, labels=False))
        return labels, int(labels.max() + 1)

    item_ids = dataset.inter_feat["item_id"].numpy()
    item_counts = np.bincount(item_ids, minlength=n_items)
    user_pop_avged = np.zeros(len(sequences))
    seqs_np = sequences.numpy()
    lens_np = seq_lengths.numpy()
    for i in range(len(sequences)):
        valid_items = seqs_np[i, : lens_np[i]]
        valid_items = valid_items[valid_items > 0]
        if len(valid_items) > 0:
            user_pop_avged[i] = np.mean(item_counts[valid_items])
    labels = np.array(
        pd.qcut(pd.Series(user_pop_avged).rank(method="first"), q=NUM_CLUSTERS, labels=False)
    )
    return labels, int(labels.max() + 1)


def build_cluster_data(cluster_labels, num_clusters, sequences, seq_lengths, representations, top_k_lists):
    cluster_data = {}
    for k in range(num_clusters):
        idx_k = np.where(cluster_labels == k)[0]
        cluster_data[k] = {
            "N_k": len(idx_k),
            "seqs": sequences[idx_k],
            "lengths": seq_lengths[idx_k],
            "reprs": representations[idx_k],
            "topk": top_k_lists[idx_k],
        }
    return cluster_data


def build_popularity_pool(dataset, n_items):
    item_ids = dataset.inter_feat["item_id"].numpy()
    item_counts = np.bincount(item_ids, minlength=n_items)
    valid_item_ids = np.arange(1, n_items)
    valid_counts = item_counts[1:]
    order = np.argsort(-valid_counts, kind="stable")
    sorted_items = valid_item_ids[order]
    top_n = max(1, int(np.ceil(len(sorted_items) * TOP_POP_RATIO)))
    return sorted_items[:top_n]


def sample_pop_items(pop_pool, num_inject, batch_size, seen_items_list, rng):
    out = np.zeros((batch_size, num_inject), dtype=np.int64)
    for i in range(batch_size):
        used = set()
        seen = seen_items_list[i]
        for j in range(num_inject):
            attempts = 0
            item_val = int(pop_pool[rng.integers(0, len(pop_pool))])
            while (item_val == 0 or item_val in seen or item_val in used) and attempts < 50:
                item_val = int(pop_pool[rng.integers(0, len(pop_pool))])
                attempts += 1
            if item_val == 0 or item_val in seen or item_val in used:
                fallback = [int(x) for x in pop_pool if int(x) not in seen and int(x) not in used]
                item_val = fallback[0] if fallback else int(pop_pool[0])
            out[i, j] = item_val
            used.add(item_val)
    return torch.tensor(out, dtype=torch.long)


def append_items(orig_seq, orig_len, items_to_add):
    batch_size, num_inject = items_to_add.shape
    max_len = orig_seq.shape[0]
    expanded = orig_seq.unsqueeze(0).expand(batch_size, -1).clone()
    new_lens = torch.full((batch_size,), orig_len, dtype=torch.long)
    shift_size = (orig_len + num_inject) - max_len

    if shift_size > 0:
        expanded[:, :-shift_size] = expanded[:, shift_size:].clone()
        expanded[:, -num_inject:] = items_to_add
        new_lens[:] = max_len
    else:
        for i in range(num_inject):
            expanded[:, orig_len + i] = items_to_add[:, i]
        new_lens += num_inject
    return expanded, new_lens


def sanitize_random_items(items_to_add, seen_items, n_items, rng):
    batch_size, num_inject = items_to_add.shape
    for i in range(batch_size):
        used = set()
        for j in range(num_inject):
            item_val = int(items_to_add[i, j].item())
            attempts = 0
            while (item_val == 0 or item_val >= n_items or item_val in seen_items or item_val in used) and attempts < 20:
                item_val = int(rng.integers(1, n_items))
                attempts += 1
            items_to_add[i, j] = item_val
            used.add(item_val)
    return items_to_add


def masked_topk(scores, seen_items, device):
    if seen_items:
        seen_tensor = torch.tensor(sorted(seen_items), dtype=torch.long, device=device)
        scores[:, seen_tensor] = -1e9
    _, topk = torch.topk(scores, k=TOPK_SIZE, dim=-1)
    return topk


def compute_output_dir(args):
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "SU_exp",
        args.dataset_name,
        args.model_name,
        args.clustering_method,
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def resolve_runtime_metadata(args, config):
    checkpoint_stem = Path(args.model_file).stem
    model_name = args.model_name or config["model"]
    dataset_name = args.dataset_name or config["dataset"]
    run_id = args.run_id if args.run_id and args.run_id != "auto" else checkpoint_stem
    return model_name, dataset_name, run_id


def evaluate_no_injection(cluster_data, cluster_pairs, item_embeddings, device, rng):
    cos = torch.nn.CosineSimilarity(dim=-1)
    records = []

    with torch.no_grad():
        for c1, c2 in cluster_pairs:
            pool_c1 = cluster_data[c1]["N_k"]
            pool_c2 = cluster_data[c2]["N_k"]
            if pool_c1 == 0 or pool_c2 == 0:
                continue

            idx_c1 = rng.choice(pool_c1, size=min(pool_c1, SAMPLE_SIZE), replace=False)
            idx_c2 = rng.choice(pool_c2, size=min(pool_c2, SAMPLE_SIZE), replace=False)
            topk_tgt_all = cluster_data[c2]["topk"][idx_c2].to(device)
            seq1 = cluster_data[c1]["seqs"][idx_c1]
            len1 = cluster_data[c1]["lengths"][idx_c1]
            repr1 = cluster_data[c1]["reprs"][idx_c1]
            topk_src_all = cluster_data[c1]["topk"][idx_c1]

            for n in range(len(idx_c1)):
                orig_len = len1[n].item()
                orig_seq = seq1[n]
                seen_items = set(orig_seq[:orig_len].tolist())
                seen_items.discard(0)

                orig_latent = repr1[n].unsqueeze(0).to(device)
                scores = torch.matmul(orig_latent, item_embeddings.transpose(0, 1))
                strategic_topk = masked_topk(scores, seen_items, device)
                topk_n = topk_src_all[n].to(device)

                topk_self = topk_n.unsqueeze(0).expand(len(idx_c2), -1)
                topk_base = strategic_topk.expand(len(idx_c2), -1)
                rbo_n = calc_batch_rbo(topk_base, topk_self)
                rbo_m = calc_batch_rbo(topk_base, topk_tgt_all)
                diff_rbo = rbo_m - rbo_n

                v_l_strat = item_embeddings[strategic_topk.squeeze(0)].mean(dim=0).unsqueeze(0).expand(len(idx_c2), -1)
                v_l_n = item_embeddings[topk_n].mean(dim=0).unsqueeze(0).expand(len(idx_c2), -1)
                v_l_m = item_embeddings[topk_tgt_all].mean(dim=1)
                sim_emb_n = cos(v_l_strat, v_l_n)
                sim_emb_m = cos(v_l_strat, v_l_m)
                diff_emb = sim_emb_m - sim_emb_n
                ratio_emb = diff_emb / sim_emb_n.abs().clamp(min=1e-8)
                diff_latent = torch.zeros(len(idx_c2), device=device)

                for m in range(len(idx_c2)):
                    records.append(
                        {
                            "cluster_src": c1,
                            "cluster_tgt": c2,
                            "user_idx": int(idx_c1[n]),
                            "num_inject": 0,
                            "RBO_Diff": diff_rbo[m].item(),
                            "Emb_Sim_Diff": diff_emb[m].item(),
                            "Emb_Sim_Ratio": ratio_emb[m].item(),
                            "Latent_Shift": diff_latent[m].item(),
                        }
                    )
    return records


def evaluate_injection_mode(cluster_data, cluster_pairs, model, item_embeddings, device, injection_mode, n_items, rng, pop_pool):
    cos = torch.nn.CosineSimilarity(dim=-1)
    records = []

    with torch.no_grad():
        for c1, c2 in cluster_pairs:
            pool_c1 = cluster_data[c1]["N_k"]
            pool_c2 = cluster_data[c2]["N_k"]
            if pool_c1 == 0 or pool_c2 == 0:
                continue

            idx_c1 = rng.choice(pool_c1, size=min(pool_c1, SAMPLE_SIZE), replace=False)
            idx_c2 = rng.choice(pool_c2, size=min(pool_c2, SAMPLE_SIZE), replace=False)
            m_c2 = len(idx_c2)

            seq1 = cluster_data[c1]["seqs"][idx_c1]
            len1 = cluster_data[c1]["lengths"][idx_c1]
            repr1 = cluster_data[c1]["reprs"][idx_c1]
            seq2 = cluster_data[c2]["seqs"][idx_c2]
            len2 = cluster_data[c2]["lengths"][idx_c2]
            topk_tgt_all = cluster_data[c2]["topk"][idx_c2].to(device)

            for n in range(len(idx_c1)):
                orig_len = len1[n].item()
                orig_seq = seq1[n]
                seen_items = set(orig_seq[:orig_len].tolist())
                seen_items.discard(0)
                source_repr = repr1[n].unsqueeze(0).to(device)
                topk_n = cluster_data[c1]["topk"][idx_c1[n]].to(device)

                for num_inject in NUM_INJECT_VALUES:
                    if injection_mode == "target":
                        idx_matrix = (len2 - num_inject).unsqueeze(1) + torch.arange(num_inject).unsqueeze(0)
                        idx_matrix = torch.clamp(idx_matrix, min=0)
                        items_to_add = seq2.gather(dim=1, index=idx_matrix).clone()
                        items_to_add = sanitize_random_items(items_to_add, seen_items, n_items, rng)
                    elif injection_mode == "random":
                        items_to_add = torch.randint(1, n_items, (m_c2, num_inject))
                        items_to_add = sanitize_random_items(items_to_add, seen_items, n_items, rng)
                    else:
                        seen_list = [seen_items for _ in range(m_c2)]
                        items_to_add = sample_pop_items(pop_pool, num_inject, m_c2, seen_list, rng)

                    n_seq_expanded, new_lens = append_items(orig_seq, orig_len, items_to_add)
                    n_seq_expanded = n_seq_expanded.to(device)
                    new_lens = new_lens.to(device)

                    strategic_outputs = model.forward(n_seq_expanded, new_lens)
                    new_scores = torch.matmul(strategic_outputs, item_embeddings.transpose(0, 1))
                    strategic_topk = masked_topk(new_scores, seen_items, device)

                    topk_self = topk_n.unsqueeze(0).expand(m_c2, -1)
                    rbo_n = calc_batch_rbo(strategic_topk, topk_self)
                    rbo_m = calc_batch_rbo(strategic_topk, topk_tgt_all)
                    diff_rbo = rbo_m - rbo_n

                    v_l_strat = item_embeddings[strategic_topk].mean(dim=1)
                    v_l_n = item_embeddings[topk_n].mean(dim=0).unsqueeze(0).expand(m_c2, -1)
                    v_l_m = item_embeddings[topk_tgt_all].mean(dim=1)
                    sim_emb_n = cos(v_l_strat, v_l_n)
                    sim_emb_m = cos(v_l_strat, v_l_m)
                    diff_emb = sim_emb_m - sim_emb_n
                    ratio_emb = diff_emb / sim_emb_n.abs().clamp(min=1e-8)
                    diff_latent = torch.norm(strategic_outputs - source_repr.expand(m_c2, -1), p=2, dim=1)

                    for m in range(m_c2):
                        records.append(
                            {
                                "cluster_src": c1,
                                "cluster_tgt": c2,
                                "user_idx": int(idx_c1[n]),
                                "num_inject": num_inject,
                                "RBO_Diff": diff_rbo[m].item(),
                                "Emb_Sim_Diff": diff_emb[m].item(),
                                "Emb_Sim_Ratio": ratio_emb[m].item(),
                                "Latent_Shift": diff_latent[m].item(),
                            }
                        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustering_method", type=str, default="kmeans", choices=CLUSTERING_CHOICES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--injection_mode", type=str, default="target", choices=INJECTION_CHOICES)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default="auto")
    args = parser.parse_args()

    start_time = time.time()
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=args.model_file)
    args.model_name, args.dataset_name, args.run_id = resolve_runtime_metadata(args, config)

    device = torch.device(args.device)
    config["device"] = device
    model.to(device)
    model.eval()

    item_embeddings = get_item_embeddings(model, device)
    n_items = item_embeddings.shape[0]
    representations, sequences, seq_lengths, top_k_lists = extract_eval_tensors(model, test_data, device, item_embeddings)
    cluster_labels, num_clusters = build_cluster_labels(
        args.clustering_method, representations, sequences, seq_lengths, dataset, n_items
    )
    cluster_data = build_cluster_data(cluster_labels, num_clusters, sequences, seq_lengths, representations, top_k_lists)
    cluster_pairs = list(itertools.permutations(range(num_clusters), 2))
    rng = np.random.default_rng(SAMPLE_SEED)
    pop_pool = build_popularity_pool(dataset, n_items) if args.injection_mode == "pop" else None

    if args.injection_mode == "no_injection":
        agg_records = evaluate_no_injection(cluster_data, cluster_pairs, item_embeddings, device, rng)
    else:
        agg_records = evaluate_injection_mode(
            cluster_data,
            cluster_pairs,
            model,
            item_embeddings,
            device,
            args.injection_mode,
            n_items,
            rng,
            pop_pool,
        )

    output_dir = compute_output_dir(args)
    df = pd.DataFrame(agg_records)
    df["model_name"] = args.model_name
    df["dataset_name"] = args.dataset_name
    df["run_id"] = args.run_id
    df["injection_mode"] = args.injection_mode

    csv_filename = (
        f"raw_{args.dataset_name}_{args.model_name}_{args.clustering_method}_{args.injection_mode}_{args.run_id}.csv"
    )
    raw_data_path = os.path.join(output_dir, csv_filename)
    df.to_csv(raw_data_path, index=False)
    print(f"Saved to {raw_data_path}")
    print(f"Elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
