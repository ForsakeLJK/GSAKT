from utils import evaluate
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from custom_dataset import CustomDataset
import torch
from torch.optim import lr_scheduler
from model import Model
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import wandb
import argparse

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="assist09")
    arg_parser.add_argument("--head_num", type=int, default=5)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--seq_len", type=int, default=21)
    arg_parser.add_argument("--epoch_num", type=int, default=100)
    arg_parser.add_argument("--lr", type=float, default=0.01)
    arg_parser.add_argument("--node_feature_size", type=int, default=100)
    arg_parser.add_argument("--hidden_dim", type=int, default=100)
    arg_parser.add_argument("--dropout", type=float, nargs="?", default=[0.3, 0.2, 0.2])
    arg_parser.add_argument("--gcn_layer_num", type=int, default=3)
    arg_parser.add_argument("--save_num", type=str, required=True)
    arg_parser.add_argument("--lr_decay", type=float, default=None)
    
    args = arg_parser.parse_args()
    
    
    #* #### parameters ####
    if args.dataset not in ["assist09", "ednet"]:
        raise ValueError("dataset <{}> not supported".format(args.dataset))
    
    train_dir = "data/" + args.dataset + "/" + args.dataset + "_train.csv"
    test_dir = "data/" + args.dataset + "/" + args.dataset + "_test.csv"
    skill_matrix_dir = "data/" + args.dataset + "/" + args.dataset + "_skill_matrix.csv"
    qs_graph_dir = "data/" + args.dataset + "/" + args.dataset + "_qs_graph.json"
    # get skill cnt
    # skill idx -> 0 ~ skill_cnt - 1 
    # question idx -> skill_cnt ~ max_idx - 2
    # correctness idx -> max_idx - 1, max_idx 
    # please refer to utils.get_metadata to see how to get them
    node_feature_size = args.node_feature_size
    hidden_dim = args.hidden_dim
    
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
    
    save_dir_best = "saved/" + args.save_num + "_best.pt"
    save_dir_final =  "saved/" + args.save_num + "_final.pt"
    dropout = args.dropout
    gcn_layer_num = args.gcn_layer_num
    
    lr_decay = args.lr_decay
    
    if lr_decay is not None:
        print("lr_decay: True, {}".format(lr_decay))
    else:
        print("lr_decay: False")
    
    #* ####    end     ####
    
    wandb.init(entity="fmlab-its", project="KT")
    
    config = wandb.config
    config.dataset = args.dataset
    config.node_feature_size = node_feature_size
    config.hidden_dim = hidden_dim
    config.head_num = head_num
    config.batch_size = batch_size
    config.seq_len = seq_len
    config.lr = lr
    config.dropout = dropout
    config.gcn_layer_num = gcn_layer_num
    
    config.save_num = args.save_num
    
    print("cuda availability: {}".format(torch.cuda.is_available()))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Model(node_feature_size, hidden_dim, node_feature_size, seq_len, head_num, qs_graph_dir, device)
    model.to(device)
    
    wandb.watch(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    if lr_decay is not None:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=lr_decay)
    
    train_set = CustomDataset(train_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
    # train_set, val_set = random_split(train_set, [0.7 * len(train_set), 0.3 * len(train_set)])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    test_set = CustomDataset(test_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    # print("train samples: {}, val samples: {}, test samples: {}"\
    #     .format(len(train_set), len(test_set)))
    
    print("start...")
    
    train_losses = []
    
    best_test_auc = 0.0
    # best_test_auc = 0.0
    
    for epoch in tqdm(range(epoch_num)):
        
        print("epoch {} start".format(epoch + 1))
        
        model.train()
        
        
        # train_preds = []
        # train_targets = []
        
        for _, (hist_seq, hist_answers, new_seq, target_answers, _) in tqdm(enumerate(train_loader)):
            
            hist_seq, hist_answers, new_seq, target_answers = \
                hist_seq.to(device), hist_answers.to(device), new_seq.to(device), target_answers.to(device)
            
            # * foward pass
            # pred = model(x)
            # (batch_size, seq_len - 1, 1)
            pred = model(hist_seq, hist_answers, new_seq)
            
            # train_preds.append(pred)
            # train_targets.append(target_answers)
            
            # (batch_size, seq_len - 1, 1) -> (batch_size, seq_len)
            # pred = pred.squeeze()
    
            # * compute loss
            loss = model.loss(pred, target_answers.float())
            
            train_losses.append(loss.item())
    
            # * backward pass & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_loss = np.sum(train_losses) / len(train_losses)
        
        # train_targets = torch.cat(train_targets)
        # train_preds = torch.cat(train_preds).sigmoid()
        
        # train_auc = metrics.roc_auc_score(train_targets, train_preds)
        
        model.eval()
        
        test_auc = evaluate(model, test_loader, device)
        
        # print("epoch {}: train_loss: {}, valid_auc: {}, test_auc: {}".format(epoch+1, epoch_loss, valid_auc, test_auc))
        print("epoch {}: train_loss: {}, test_auc: {}".format(epoch+1, epoch_loss, test_auc))
        
        wandb.log({"train_loss": epoch_loss, "test_auc": test_auc})
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            torch.save(model.state_dict(), save_dir_best)
            print("best_auc: {} at epoch {}".format(best_test_auc, epoch + 1))
        
        if lr_decay is not None:
            scheduler.step()
        
    wandb.log({"best_auc": best_test_auc})
    print("best_auc: {}".format(best_test_auc))
    torch.save(model.state_dict(), save_dir_final)
    
    print("done.")
    
    return 

if __name__ == '__main__':
    
    main()
    