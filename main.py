import json
from utils import evaluate, get_edge_index, get_node_labels
from torch.utils.data.dataloader import DataLoader
from custom_dataset import CustomDataset
import torch
from model import Model
from tqdm import tqdm
import numpy as np
import wandb
import argparse
import shortuuid
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import warnings

def train(model : Model, optimizer : torch.optim.Adam, epoch_num, train_loader, test_loader, save_dir_best, save_dir_final, device : torch.device):
    
    train_losses = []
    
    best_test_auc = 0.0
    
    for epoch in tqdm(range(epoch_num)):
        
        model.train()
        
        for _, (hist_seq, hist_answers, new_seq, target_answers, _) in tqdm(enumerate(train_loader)):
            
            hist_seq, hist_answers, new_seq, target_answers = \
                hist_seq.to(device), hist_answers.to(device), new_seq.to(device), target_answers.to(device)
            
            # * foward pass
            # (batch_size, seq_len - 1, 1)
            pred = model(hist_seq, hist_answers, new_seq)

            # * compute loss
            loss = model.loss(pred, target_answers.float())
            
            train_losses.append(loss.item())

            # * backward pass & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_loss = np.sum(train_losses) / len(train_losses)
        
        model.eval()
        
        test_auc = evaluate(model, test_loader, device)
        
        print("epoch {}: train_loss: {}, test_auc: {}".format(epoch+1, epoch_loss, test_auc))
        
        wandb.log({"train_loss": epoch_loss, "test_auc": test_auc})
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            torch.save(model.state_dict(), save_dir_best)
            print("best_auc: {} at epoch {}".format(best_test_auc, epoch + 1))
        
    wandb.log({"best_auc": best_test_auc})
    print("best_auc: {}".format(best_test_auc))
    torch.save(model.state_dict(), save_dir_final)
    
    print("done.")

def init_proj(config):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="assist09")
    arg_parser.add_argument("--head_num", type=int, default=5)
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--seq_len", type=int, default=41)
    arg_parser.add_argument("--epoch_num", type=int, default=40)
    arg_parser.add_argument("--lr", type=float, default=0.001)
    arg_parser.add_argument("--node_feature_size", type=int, default=100)
    arg_parser.add_argument("--node_embedding_size", type=int, default=100)
    arg_parser.add_argument("--hidden_dim", type=int, default=200)
    arg_parser.add_argument("--dropout", type=float, nargs="?", default=[0.3, 0.2, 0.2])
    arg_parser.add_argument("--gcn_layer_num", type=int, default=1)
    arg_parser.add_argument("--n_hop", type=int, default=2)
    arg_parser.add_argument("--gcn_on", type=int, default=1)
    arg_parser.add_argument("--gcn_type", type=str, default='sgconv')
    arg_parser.add_argument("--pretrain_uuid", type=str, default=None)
    
    args = arg_parser.parse_args()

    if args.dataset not in ["assist09", "assist12", "ednet"]:
        raise ValueError("unknown dataset <{}>".format(args.dataset))
    
    train_dir = "data/" + args.dataset + "/" + args.dataset + "_train.csv"
    test_dir = "data/" + args.dataset + "/" + args.dataset + "_test.csv"
    skill_matrix_dir = "data/" + args.dataset + "/" + args.dataset + "_skill_matrix.csv"
    qs_graph_dir = "data/" + args.dataset + "/" + args.dataset + "_qs_graph.json"
    if args.pretrain_uuid is not None:
        pretrain_dir = "pretrained/" + args.dataset + "/" + args.pretrain_uuid + ".pt"
    else:
        pretrain_dir = None
    # get skill cnt
    # skill idx -> 0 ~ skill_cnt - 1 
    # question idx -> skill_cnt ~ max_idx - 2
    # correctness idx -> max_idx - 1, max_idx 
    # please refer to utils.get_metadata to see how to get them
    node_feature_size = args.node_feature_size
    hidden_dim = args.hidden_dim
    node_embedding_size = args.node_embedding_size
    if args.gcn_on in [0, 1]:
        gcn_on = True if args.gcn_on == 1 else False
    else:
        raise ValueError("gcn_on must be 0 or 1")
    
    if args.gcn_type in ['gconv', 'sgconv']:
        gcn_type = args.gcn_type
    else:
        raise ValueError('unknown gcn_type {}'.format(args.gcn_type))
    
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
    
    # ! (seq_len - 1) must be divisible by head_num
    head_num = args.head_num
    batch_size = args.batch_size
    seq_len = args.seq_len
    
    if not (seq_len - 1) % head_num == 0:
        raise ValueError("seq_len - 1 <{}> is not divisible by head_num <{}>".format(seq_len - 1, head_num))
    
    epoch_num = args.epoch_num
    lr = args.lr
    
    model_uuid = shortuuid.uuid()
    
    save_dir_best = "saved/" + model_uuid + "_best.pt"
    save_dir_final =  "saved/" + model_uuid + "_final.pt"
    dropout = args.dropout
    gcn_layer_num = args.gcn_layer_num
    n_hop = args.n_hop
    
    config.dataset = args.dataset
    config.node_feature_size = node_feature_size
    config.hidden_dim = hidden_dim
    config.head_num = head_num
    config.batch_size = batch_size
    config.seq_len = seq_len
    config.lr = lr
    config.dropout = dropout
    config.gcn_layer_num = gcn_layer_num
    config.n_hop = n_hop
    config.gcn_on = gcn_on
    config.gcn_type = gcn_type
    config.save_num = model_uuid
    
    config.node_embedding_size = node_embedding_size
    
    print("cuda availability: {}".format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))
    
    print("gcn_on: {}".format(gcn_on))
    if gcn_on:
        print("gcn_type: {}".format(gcn_type))
        print("n_hop: {}".format(n_hop))
    
    
        
    return  lr, node_feature_size, hidden_dim, node_embedding_size, seq_len, head_num, gcn_on, dropout, gcn_layer_num, n_hop, gcn_type, batch_size, epoch_num,\
            single_skill_cnt, skill_cnt, max_idx, device,\
            train_dir, test_dir, qs_graph_dir, save_dir_best, save_dir_final, pretrain_dir

def main():
    
    wandb.init(entity="fmlab-its", project="KT")
    
    lr, node_feature_size, hidden_dim, node_embedding_size, seq_len, head_num, gcn_on, dropout, gcn_layer_num, n_hop, gcn_type, batch_size, epoch_num,\
        single_skill_cnt, skill_cnt, max_idx, device,\
        train_dir, test_dir, qs_graph_dir, save_dir_best, save_dir_final, pretrain_dir = init_proj(wandb.config)
    
    if pretrain_dir is None:
        model = Model(node_feature_size, hidden_dim, node_embedding_size, seq_len, head_num, qs_graph_dir, device, dropout, n_hop, gcn_type, gcn_layer_num, gcn_on)
    else:
        with open(qs_graph_dir, "r") as src:
            qs_graph = json.load(src)
        qs_graph_torch = Data(x= None,
            edge_index=get_edge_index(qs_graph), 
            y=get_node_labels(qs_graph)).to(device)
        pretrained_model = pyg_nn.Node2Vec(edge_index=qs_graph_torch.edge_index, embedding_dim=node_feature_size, 
                                walk_length=20, context_size=10, walks_per_node=10, 
                                num_negative_samples=1, p=1, q=1, sparse=True)
        pretrained_model.load_state_dict(torch.load(pretrain_dir, map_location=device))
        pretrained_model.to(device)
        pretrained_model.eval()
        pretrained_embedding = pretrained_model()
        
        print("pretrained model loaded.")
        
        model = Model(node_feature_size, hidden_dim, node_embedding_size, seq_len, head_num, qs_graph_dir, device, dropout, n_hop, gcn_type, gcn_layer_num, gcn_on, pretrained_embedding)
    
    model.to(device)
    
    wandb.watch(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    train_set = CustomDataset(train_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = CustomDataset(test_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    train(model, optimizer, epoch_num, train_loader, test_loader, save_dir_best, save_dir_final, device)
    
    return 

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
    