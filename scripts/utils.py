import os
import dgl
import torch
import shutil
import random
import numpy as np
import argparse
import nni
import json

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_param():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="test")
    parser.add_argument("--split_mode", type=str, default="random")
    parser.add_argument("--input_dim", type=int, default=7)
    parser.add_argument("--output_dim", type=int, default=7)
    parser.add_argument("--ppi_hidden_dim", type=int, default=512)
    parser.add_argument("--prot_hidden_dim", type=int, default=128)
    parser.add_argument("--ppi_num_layers", type=int, default=2)
    parser.add_argument("--prot_num_layers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--dropout_ratio", type=float, default=0.0)

    parser.add_argument("--pre_epoch", type=int, default=50)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--sce_scale", type=float, default=1.5)
    parser.add_argument("--mask_loss", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_num", type=int, default=1)
    parser.add_argument("--data_mode", type=int, default=0)
    parser.add_argument("--data_split_mode", type=int, default=0)
    parser.add_argument("--pre_train", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)

    args = parser.parse_known_args()[0]
    param = args.__dict__
    param.update(nni.get_next_parameter())
    if os.path.exists("../configs/param_configs.json"):
        param = json.loads(open("../configs/param_configs.json", 'r').read())[param['dataset']][param['split_mode']]
    set_seed(args.seed)

    return param,args