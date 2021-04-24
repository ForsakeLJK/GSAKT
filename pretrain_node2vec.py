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
    arg_parser.add_argument("--p", type=float, default=1)
    arg_parser.add_argument("--q", type=float, default=1)
    arg_parser.add_argument("--walk_length", type=int, default=20)
    arg_parser.add_argument("--context_size", type=int, default=10)
    arg_parser.add_argument("--walks_per_node", type=int, default=10)
    
    args = arg_parser.parse_args()
    
    p = args.p
    q = args.q
    walk_length = args.walk_length
    context_size = args.context_size
    walks_per_node = args.walks_per_node
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
    save_dir = "pretrained/" + args.dataset + "/" + model_uuid + ".pt"
    
    model = pyg_nn.Node2Vec(edge_index=qs_graph_torch.edge_index, embedding_dim=embedding_dim, 
                            walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node, 
                            num_negative_samples=1, p=p, q=q, sparse=True)
    model.to(device)
    loader = model.loader(batch_size=128, shuffle=True)
    
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    wandb.init(entity="fmlab-its", project="KT")
    config = wandb.config
    config.save_num = model_uuid
    config.epoch_num = epoch_num
    config.embedding_dim = embedding_dim
    config.dataset = args.dataset
    config.p = args.p
    config.q = args.q
    config.walk_length = walk_length
    config.context_size = context_size
    
    wandb.watch(model)
    
    print("start...")
    
    for epoch in tqdm(range(epoch_num)):
        model.train()
        total_loss = 0.0
        
        for _, (pos_rw, neg_rw) in tqdm(enumerate(loader)):
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
    
