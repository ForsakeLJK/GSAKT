from torch.utils.data.dataloader import DataLoader
from custom_dataset import CustomDataset
import numpy as np
import pandas as pd
from utils import get_dataframe
import json
from torch_geometric.data import Data
import torch
from model import Model

def main():
    #### parameters ####
    train_dir = "assist09_train.csv"
    test_dir = "assist09_test.csv"
    skill_matrix_dir = "assist09_skill_matrix.txt"
    qs_graph_dir = "assist09_qs_graph.json"
    # get skill cnt
    # skill idx -> 0 ~ skill_cnt - 1 
    # question idx -> skill_cnt ~ max_idx - 2
    # correctness idx -> max_idx - 1, max_idx 
    # please refer to utils.get_metadata to see how to get them
    node_feature_size = 100
    hidden_dim = 100
    single_skill_cnt = 123
    skill_cnt = 167
    max_idx = 17905
    # ! (seq_len - 1) must be divisible by head_num
    head_num = 5
    batch_size = 32
    seq_len = 21
    epoch_num = 100
    lr = 0.1
    ####    end     ####
    
    """
    输入 (* batch_size):
        题目序列与回答序列, 会被分割为:
            历史题目序列 (数据集中的序列去掉最后一道题)
            历史回答序列 (数据集中的序列去掉最后一道题)
            新题序列 (数据集中的序列去掉第一道题)
            历史题目序列+历史回答序列组成新的输入
    输出 (* batch_size):
        对新题的正确率预测序列
    
    """

    
    # TODO: init before training
    model = Model(node_feature_size, hidden_dim, node_feature_size, seq_len, head_num, qs_graph_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    train_set = CustomDataset(train_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epoch_num):
        for batch_idx, (hist_seq, hist_answers, new_seq, target_answers) in enumerate(train_loader):
            # TODO: foward pass
            # pred = model(x)
            # (batch_size, seq_len - 1, 1)
            pred = model(hist_seq, hist_answers, new_seq)
            
            # (batch_size, seq_len - 1, 1) -> (batch_size, seq_len)
            pred = pred.squeeze()
    
            # TODO: compute loss
            loss = model.loss(pred, target_answers)
    
            # TODO: backward pass & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    
    return 

if __name__ == '__main__':
    main()
    