import json
from os import getegid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T
import numpy as np
  
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, head_num, qs_graph_dir, device, dropout=[0.3, 0.2, 0.2], gcn_layer_num=3):
        """[summary]

        Args:
            input_dim ([type]): [description]
            hidden_dim ([type]): [description]
            output_dim ([type]): output dim of GCN
            seq_len ([type]): [description]
            head_num ([type]): [description]
            qs_graph_dir ([type]): [description]
        """
        super(Model, self).__init__()
        
        self.device = device
        
        self.gcn_layer_num = gcn_layer_num
        self.dropout = dropout

        with open(qs_graph_dir, "r") as src:
            self.qs_graph = json.load(src)
        
        # ! input shape should be (seq_len - 1)        
        self.seq_len = seq_len
        
        # ! these embedding will all be trained in an end-to-end manner
        self.correctness_embedding = nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.empty(2, input_dim, dtype=torch.float32)))
        
        # ! BUG: x cannot be set as parameter
        self.qs_graph_torch = Data(x=nn.parameter.Parameter(nn.init.xavier_uniform_(torch.empty(len(self.qs_graph), input_dim, dtype=torch.float32))), 
                                  edge_index=get_edge_index(self.qs_graph), 
                                  y=get_node_labels(self.qs_graph)).to(device)
        
        # self.qs_graph_torch = data(x=torch.nn.init.xavier_uniform_(torch.empty(len(self.qs_graph), input_dim, dtype=torch.float32)), 
        #                     edge_index=get_edge_index(self.qs_graph), 
        #                     y=get_node_labels(self.qs_graph))
        
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(output_dim*2, output_dim, bias=True))
        
        for _ in range(3):
            self.linears.append(nn.Linear(output_dim, output_dim, bias=False))
        
        self.linears.append(nn.Linear(output_dim, 1))
        
        self.MHA = nn.MultiheadAttention(embed_dim=output_dim, num_heads=head_num, dropout=self.dropout[1])
        
        self.FFN = nn.ModuleList()
        
        for _ in range(2):
            self.FFN.append(nn.Linear(output_dim, output_dim))
        
        self.convs = nn.ModuleList()
        # self.convs.append(pyg_nn.SGConv(input_dim, hidden_dim, K=3))
        self.convs.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        
        self.lns.append(nn.LayerNorm(output_dim))
        self.lns.append(nn.LayerNorm(output_dim))
        
        for i in range(self.gcn_layer_num - 1):
            if i == self.gcn_layer_num - 2:
                self.convs.append(pyg_nn.GCNConv(hidden_dim, output_dim))
            else:
                # self.convs.append(pyg_nn.SGConv(hidden_dim, hidden_dim, K=3))
                self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
        
        self.pos_embedding = nn.Embedding(seq_len - 1, output_dim)
        
        self.dropout_layers = nn.ModuleList()

        self.dropout_layers.append(nn.Dropout(p=self.dropout[0]))
        self.dropout_layers.append(nn.Dropout(p=self.dropout[2]))
        
    
    def forward(self, hist_seq, hist_answers, new_seq):
        """Forward

        Args:
            graph ([type]): qs_graph_torch
            hist_seq ([type]): (batch_size, seq_len - 1)
            hist_answers ([type]): (batch_size, seq_len - 1)
            new_seq ([type]): (batch_size, seq_len - 1)
        """
        x, edge_index = self.qs_graph_torch.x, self.qs_graph_torch.edge_index
        correctness_embedding = self.correctness_embedding
        
        for i in range(self.gcn_layer_num):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = F.relu(x)
            x = self.dropout_layers[0](x)
            if not i == self.gcn_layer_num - 1:
                x = self.lns[i](x)
        
        # (idx_size, hidden_dim)
        # print(x.shape)
        
        hist_seq_embed = get_embedding(hist_seq, x)
        new_seq_embed = get_embedding(new_seq, x)
        hist_answers_embed = get_embedding(hist_answers, correctness_embedding)
        
        # (batch_size, seq_len - 1, hidden_dim * 2) -> (batch_size, seq_len - 1, hidden_dim)
        interaction_embed = self.linears[0](torch.cat([hist_seq_embed, hist_answers_embed], dim=2))

        # (seq_len - 1, hidden_dim)
        pos_embed = self.pos_embedding(torch.arange(self.seq_len - 1).unsqueeze(0).to(self.device))
        
        interaction_embed = pos_embed + interaction_embed
        
        value = self.linears[1](interaction_embed).permute(1,0,2)
        key = self.linears[2](interaction_embed).permute(1,0,2)
        query = self.linears[3](new_seq_embed).permute(1,0,2)
        
        atn, _ = self.MHA(query, key, value, attn_mask=torch.from_numpy(np.triu(np.ones((self.seq_len-1, self.seq_len-1)), k=1).astype('bool')).to(self.device))
        
        atn = self.lns[2](atn + query).permute(1,0,2)
        
        ffn = self.dropout_layers[1](self.FFN[1](F.relu(self.FFN[0](atn))))
        ffn = self.lns[3](ffn + atn)
        
        pred = self.linears[4](ffn)
        
        return pred.squeeze()
    
    def loss(self, pred, label):
        
        return F.binary_cross_entropy_with_logits(pred, label)

# (batch_size, seq_len - 1), (num, emb_dim)
def get_embedding(seq, embeddings):
    seq = seq.unsqueeze(-1)
    seq = seq.repeat(1, 1, embeddings.shape[1]).float()
    
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            idx = int(seq[i, j, 0].item())
            seq[i, j] = embeddings[idx]
    
    return seq

def get_node_labels(qs_graph):
    node_num = len(qs_graph)
    node_labels = torch.empty(node_num, dtype=torch.short)
    
    for idx, node in enumerate(qs_graph):
        if node["type"] == "skill":
            node_labels[idx] = 0
        elif node["type"] == "question":
            node_labels[idx] = 1
        else:
            assert "Wrong Type {} in Node {}".format(node["type"], idx)
            
            
    return node_labels

def get_edge_index(qs_graph):
    edge_num = 0
    
    for idx, node in enumerate(qs_graph):
        edge_num += len(node["neighbor"])
    
    edge_index = torch.empty((2, edge_num), dtype=torch.long)
    loc = 0
    for idx, node in enumerate(qs_graph):
        for n in node["neighbor"]:
            edge_index[0][loc] = idx
            edge_index[1][loc] = n
            loc += 1
    
    assert loc == edge_num, "edge_num ({}) is not equal to loc ({})".format(edge_num, loc)
    
    return edge_index
