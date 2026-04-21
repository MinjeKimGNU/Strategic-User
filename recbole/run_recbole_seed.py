# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse

from recbole.quick_start import run

import torch
import gc
if __name__ == "__main__":                            
    parser = argparse.ArgumentParser()
    # 1783808(ours) < 1784320
    # parser.add_argument("--model", "-m", type=str, default="PSR", help="name of models")
    # parser.add_argument("--model", "-m", type=str, default="PatchSR2", help="name of models")
    # parser.add_argument("--model", "-m", type=str, default="SASRec", help="name of models")
    parser.add_argument("--model", "-m", type=str, default="GRU4Rec", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Amazon_Beauty", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default='configs/GRU4Rec/Amazon_Beauty.yaml', help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    
    parser.add_argument(
        "--n_layers", type=int, default=4, help="n_layers"
    )
    
    parser.add_argument(
        "--n_heads", type=int, default=8, help="n_heads"
    )
    
    parser.add_argument(
        "--att_hidden_size", type=int, default=256, help="att_hidden_size"
    )
    
    parser.add_argument(
        "--inner_size", type=int, default=256, help="inner_size"
    )
    parser.add_argument(
        "--seed", type=int, default=2120, help="total number of session"
    )
    
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    args = parser.parse_args()
    parameter_dict={
                'seed': args.seed,
                #'n_layers': args.n_layers,
                #'n_heads': args.n_heads,
                #'attribute_hidden_size': [args.att_hidden_size],
                'gpu_id': 2,
                # 'MAX_ITEM_LIST_LENGTH': max_seq
                'train_batch_size': 1024
    }
    parser.add_argument(
            "--config_dict",
            type=dict,
            default=parameter_dict,
        )
    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run(    
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
        config_dict=args.config_dict,
    )
