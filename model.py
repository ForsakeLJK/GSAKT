import json
from utils import get_edge_index, get_embedding, get_node_labels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import numpy as np
  
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, head_num, qs_graph_dir, device, dropout, n_hop, gcn_type, gcn_layer_num, gcn_on, pretrained_embedding=None, freeze=True):
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
        
        self.gcn_on = gcn_on
        
        if pretrained_embedding is not None:
            self.pretrained_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze)
        else:
            self.pretrained_embedding = None
        
        if gcn_type == 'sgconv' and gcn_on:
            self.gcn_layer_num = gcn_layer_num
            self.convs = nn.ModuleList()
            if gcn_layer_num == 1:
                self.convs.append(pyg_nn.SGConv(input_dim, output_dim, K=n_hop))
            elif gcn_layer_num > 1:
                self.convs.append(pyg_nn.SGConv(input_dim, hidden_dim, K=n_hop))
            else:
                raise ValueError("Unsupported gcn_layer_num {}")
            
            for i in range(self.gcn_layer_num - 1):
                if i == self.gcn_layer_num - 2:
                    self.convs.append(pyg_nn.SGConv(hidden_dim, output_dim, K=3))
                else:
                    self.convs.append(pyg_nn.SGConv(hidden_dim, hidden_dim, K=3))
        elif gcn_type == 'gconv' and gcn_on:
            self.gcn_layer_num = gcn_layer_num
            self.convs = nn.ModuleList()
            if gcn_layer_num == 1:
                self.convs.append(pyg_nn.GCNConv(input_dim, output_dim))
            elif gcn_layer_num > 1:
                self.convs.append(pyg_nn.GCNConv(input_dim, hidden_dim, K=n_hop))
            else:
                raise ValueError("Unsupported gcn_layer_num {}")
            
            for i in range(self.gcn_layer_num - 1):
                if i == self.gcn_layer_num - 2:
                    self.convs.append(pyg_nn.GCNConv(hidden_dim, output_dim, K=3))
                else:
                    self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim, K=3))        
        
        self.dropout = dropout

        with open(qs_graph_dir, "r") as src:
            self.qs_graph = json.load(src)
        
        # ! input shape should be (seq_len - 1)        
        self.seq_len = seq_len
        
        self.correctness_embedding_layer = nn.Embedding(2, input_dim)
        
        if self.pretrained_embedding is None:
            self.node_embedding_layer = nn.Embedding(len(self.qs_graph), input_dim)
        
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(output_dim*2, output_dim, bias=True))
        
        for _ in range(3):
            self.linears.append(nn.Linear(output_dim, output_dim, bias=False))
        
        self.linears.append(nn.Linear(output_dim, 1))
        
        self.MHA = nn.MultiheadAttention(embed_dim=output_dim, num_heads=head_num, dropout=self.dropout[1])
        
        self.FFN = nn.ModuleList()
        
        for _ in range(2):
            self.FFN.append(nn.Linear(output_dim, output_dim))
        
        
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        
        self.lns.append(nn.LayerNorm(output_dim))
        self.lns.append(nn.LayerNorm(output_dim))
        
        
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
        if self.gcn_on:
            if self.pretrained_embedding is None:
                x = self.node_embedding_layer(torch.arange(len(self.qs_graph)).to(self.device))
            else:
                x = self.pretrained_embedding(torch.arange(len(self.qs_graph)).to(self.device))
            
            qs_graph_torch = Data(x= x, 
                        edge_index=get_edge_index(self.qs_graph), 
                        y=get_node_labels(self.qs_graph)).to(self.device)
            
            x, edge_index = qs_graph_torch.x, qs_graph_torch.edge_index
            
            
            
            for i in range(self.gcn_layer_num):
                x = self.convs[i](x, edge_index)
                x = F.relu(x)
                x = self.dropout_layers[0](x)
                if not i == self.gcn_layer_num - 1:
                    x = self.lns[i](x)
                    
            lookup_table = nn.Embedding.from_pretrained(x, freeze=True)
            hist_seq_embed = lookup_table(hist_seq.type(torch.LongTensor).to(self.device))
            new_seq_embed = lookup_table(new_seq.type(torch.LongTensor).to(self.device))
            
        elif self.pretrained_embedding is not None:
            hist_seq_embed = self.pretrained_embedding(hist_seq.type(torch.LongTensor).to(self.device))
            new_seq_embed = self.pretrained_embedding(new_seq.type(torch.LongTensor).to(self.device))
        else:
            hist_seq_embed = self.node_embedding_layer(hist_seq.type(torch.LongTensor).to(self.device))
            new_seq_embed = self.node_embedding_layer(new_seq.type(torch.LongTensor).to(self.device))
        
        correctness_embedding = self.correctness_embedding_layer(torch.arange(2).to(self.device))
        hist_answers_embed = get_embedding(hist_answers, correctness_embedding)
        
        # (batch_size, seq_len - 1, hidden_dim * 2) -> (batch_size, seq_len - 1, hidden_dim)
        interaction_embed = self.linears[0](torch.cat([hist_seq_embed, hist_answers_embed], dim=2))

        # (1, seq_len - 1, hidden_dim)
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
