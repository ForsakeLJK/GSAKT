import argparse

import numpy as np
from model import Model
from utils import evaluate
from torch.utils.data.dataloader import DataLoader
from custom_dataset import CustomDataset
import torch
from tqdm import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dataset", type=str, required=True)
arg_parser.add_argument("--model_dir", type=str, required=True)
arg_parser.add_argument("--batch_size", type=int, default=32)
arg_parser.add_argument("--seq_len", type=int, default=21)
arg_parser.add_argument("--node_feature_size", type=int, default=100)
arg_parser.add_argument("--hidden_dim", type=int, default=100)
arg_parser.add_argument("--head_num", type=int, default=5)
arg_parser.add_argument("--test_cnt", type=int, default=5)

args = arg_parser.parse_args()

test_cnt = args.test_cnt

test_dir = "data/" + args.dataset + "/" + args.dataset + "_test.csv"

model_dir = args.model_dir

batch_size = args.batch_size

seq_len = args.seq_len

node_feature_size = args.node_feature_size
hidden_dim = args.hidden_dim
head_num = args.head_num

qs_graph_dir = "data/" + args.dataset + "/" + args.dataset + "_qs_graph.json"

if args.dataset == "assist09":
    single_skill_cnt = 123
    skill_cnt = 167
    max_idx = 17905
elif args.dataset == "assist12":
    single_skill_cnt = 265
    skill_cnt = 265
    max_idx = 53331
elif args.dataset == "ednet":
    single_skill_cnt = 189
    skill_cnt = 1886
    max_idx = 14037
else:
    raise ValueError("metadata not defined")

test_set = CustomDataset(test_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
test_loader = DataLoader(test_set, batch_size=batch_size)

print("cuda availability: {}".format(torch.cuda.is_available()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model(node_feature_size, hidden_dim, node_feature_size, seq_len, head_num, qs_graph_dir, device)
model.load_state_dict(torch.load(model_dir, map_location=device))
model.to(device)

model.eval()

print(evaluate(model, test_loader, device))