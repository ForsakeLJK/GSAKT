import argparse
import json
import torch
import wandb
from utils import get_edge_index, get_node_labels
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import shortuuid
from tqdm import tqdm


def main():
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--dataset", type=str, default="assist09")
    arg_parser.add_argument("--embedding_dim", type=int, default=100)
    arg_parser.add_argument("--epoch_num", type=int, default=30)
    arg_parser.add_argument("--device", type=str, default='cpu')
    
    args = arg_parser.parse_args()
    
    embedding_dim = args.embedding_dim
    qs_graph_dir = "data/" + args.dataset + "/" + args.dataset + "_qs_graph.json"
    epoch_num = args.epoch_num
    
    with open(qs_graph_dir, "r") as src:
        qs_graph = json.load(src)
        
    print("cuda availability: {}".format(torch.cuda.is_available()))
    print("device: {}".format(args.device))
    
    device = torch.device("cuda:0" if args.device == 'gpu' and torch.cuda.is_available() else "cpu")
    
    qs_graph_torch = Data(x= None, 
            edge_index=get_edge_index(qs_graph), 
            y=get_node_labels(qs_graph)).to(device)
    
    model_uuid = shortuuid.uuid()
    save_dir = "pretrained/" + model_uuid + ".pt"
    
    model = pyg_nn.Node2Vec(edge_index=qs_graph_torch.edge_index, embedding_dim=embedding_dim, 
                            walk_length=20, context_size=10, walks_per_node=10, 
                            num_negative_samples=1, p=1, q=1, sparse=True)
    model.to(device)
    loader = model.loader(batch_size=128, shuffle=True)
    
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    wandb.init(entity="fmlab-its", project="KT")
    config = wandb.config
    config.save_num = model_uuid
    config.epoch_num = epoch_num
    config.embedding_dim = embedding_dim
    
    wandb.watch(model)
    
    print("start...")
    
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        
        for pos_rw, neg_rw in tqdm(enumerate(loader)):
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        print("loss: {}".format(avg_loss))
    
    
    torch.save(model.state_dict(), save_dir)
    
    print("done.")
    
    return 

if __name__ == '__main__':
    main()
    