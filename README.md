# Strategic User Workflow

Run the pipeline in this order:

```bash
bash recbole/run_seed_sasrec.sh
bash recbole/run_all.sh
```

## Step 1: Train SASRec

`run_seed_sasrec.sh` trains SASRec on:

- `Amazon_Beauty`
- `Amazon_Sports_and_Outdoors`
- `Amazon_Toys_and_Games`
- `yelp`

The checkpoints are saved in:

```bash
recbole/saved/
```

## Step 2: Export SU results

`run_all.sh` reads the saved SASRec checkpoints and runs `su_extractor.py`.

For each checkpoint, it evaluates:

- clustering: `kmeans`, `len_based`, `pop_based`
- injection: `no_injection`, `target`, `random`, `pop`

The outputs are saved in:

```bash
SU_exp/<dataset>/<model>/<clustering_method>/
```

Example file name:

```bash
raw_yelp_SASRec_kmeans_pop_<run_id>.csv
```
