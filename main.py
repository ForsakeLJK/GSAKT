import numpy as np
import pandas as pd
from utils import get_dataframe
import json
from torch_geometric.data import Data
import torch

def main():
    #### parameters ####
    train_dir = "assist09_train.csv"
    test_dir = "assist09_test.csv"
    skill_matrix_dir = "assist09_skill_matrix.txt"
    qs_graph_dir = "assist09_qs_graph.json"
    node_feature_size = 100
    ####    end     ####
    
    df_train = get_dataframe(train_dir)
    df_test = get_dataframe(test_dir)
    # use this dataframe to extract the whole q-s graph
    df_total = pd.concat([df_train, df_test], ignore_index=True)

    # get skill cnt
    # skill idx -> 0 ~ skill_cnt - 1 
    # question idx -> skill_cnt ~ max_idx - 2
    # correctness idx -> max_idx - 1, max_idx 
    skill_matrix = np.loadtxt(skill_matrix_dir)
    skill_cnt = skill_matrix.shape[0]
    single_skill_cnt = 0
    for i in range(skill_cnt):
        if skill_matrix[i, i] == 1:
            single_skill_cnt+=1
        else:
            break
    # i.e., correctness
    # ! In this sense, both 'correct' and 'incorrect' must occur in the dataset
    max_idx =  np.max([np.max(i) for _, i in enumerate(df_total["correctness"])]) 
    print("single skill: 0 ~ {}, multi-skill: {} ~ {}, question: {} ~ {}, correctness: {} and {}"\
        .format(single_skill_cnt - 1, single_skill_cnt, skill_cnt - 1, skill_cnt, max_idx - 2, max_idx - 1, max_idx))
    f = open(qs_graph_dir, "r")
    qs_graph = json.load(f)
    f.close()
    
    """
    输入 (* batch_size):
        历史题目序列 (数据集中的序列去掉最后一道题)
        历史回答序列 (数据集中的序列去掉最后一道题)
        新题序列 (数据集中的序列去掉第一道题)
    输出 (* batch_size):
        对新题的正确率预测序列
    """
    node_num = len(qs_graph)
    # node feature size = 100
    node_embeddings = torch.empty(node_num, node_feature_size)
    # initialize node feature 
    node_embeddings = torch.nn.init.xavier_uniform_(node_embeddings)
    
    edge_num = 0
    # get label
    # label -> (node_num)
    node_labels = torch.empty(node_num, dtype=torch.short)
    for idx, node in enumerate(qs_graph):
        if node["type"] == "skill":
            node_labels[idx] = 0
        elif node["type"] == "question":
            node_labels[idx] = 1
        else:
            assert "Wrong Type {} in Node {}".format(node["type"], idx)

        edge_num += len(node["neighbor"])
        
    # get edge_index
    # edge_index -> (2, edge_num),  source nodes & target nodes
    # note: this node_num is implicitly doubled in the previous loop for fitting COO format
    edge_index = torch.empty((2, edge_num), dtype=torch.long)
    loc = 0
    for idx, node in enumerate(qs_graph):
        for n in node["neighbor"]:
            edge_index[0][loc] = idx
            edge_index[1][loc] = n
            loc += 1
    
    assert loc == edge_num, "edge_num ({}) is not equal to loc ({})".format(edge_num, loc)
    
    qs_graph_torch = Data(x=node_embeddings, edge_index=edge_index, y=node_labels)
    
    assert qs_graph_torch.num_edges == edge_num
    assert qs_graph_torch.num_nodes == node_num
    assert qs_graph_torch.num_node_features == node_feature_size
    
    print("edge_num : {}, node_num : {}".format(qs_graph_torch.num_edges, qs_graph_torch.num_nodes))
    
    return 

if __name__ == '__main__':
    main()
    