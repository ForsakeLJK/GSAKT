from torch.utils.data import Dataset, DataLoader
from utils import get_dataframe
import numpy as np
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, csv_dir, metadata, seq_len):
        super(CustomDataset, self).__init__()
        self.df = slice_discard_seq(get_dataframe(csv_dir), seq_len)
        self.seq_len = seq_len
        # single skill cnt, skill cnt, max_idx
        self.metadata = metadata
        
    def __len__(self):
        # seq_num, not seq_len
        return len(self.df)
    
    def __getitem__(self, index):
        # get sequence
        # ! shift question id to access embeddings
        # skill_cnt - single_skill_cnt
        shift_num = self.metadata[1] - self.metadata[0]
        seq = np.subtract(self.df["question"][index], shift_num) 
        # get 0 and 1
        answers = np.subtract(self.df["correctness"][index], self.metadata[2] - 1) 
        target_answer_len = self.df["seq_len"][index] - 1
        # truncate or pad sequence
        # ? pad 0 or 1?
        out_seq = np.zeros(self.seq_len)
        out_answers = np.zeros(self.seq_len)
        if seq.shape[0] > self.seq_len:
            # truncate
            out_seq = seq[:self.seq_len]
            out_answers = answers[:self.seq_len]
        else:
            # pad
            out_seq[:seq.shape[0]] = seq
            out_answers[:seq.shape[0]] = answers

        hist_seq = out_seq[:-1]
        new_seq = out_seq[1:]
        target_answers = out_answers[1:]
        hist_answers = out_answers[:-1]
        
        return torch.from_numpy(hist_seq), torch.from_numpy(hist_answers), \
            torch.from_numpy(new_seq), torch.from_numpy(target_answers), \
            torch.tensor(target_answer_len)
    
def slice_discard_seq( df : pd.DataFrame, seq_len):
    # discard seq that smaller than 3
    df = df[df["seq_len"] >= 3]
    
    # slice seq that larger than seq_len & stack them on df
    sliced_df = pd.DataFrame(columns=["seq_len", "skill", "question", "correctness"])
    loc = 0
    
    long_df = df[df["seq_len"] > seq_len]
    
    for row in long_df.itertuples(index=False):
        slice_num = row[0] // seq_len
        for i in range(slice_num):
            # i * seq_len -> (i+1) * seq_len - 1
            sliced_df.loc[loc] = [seq_len, row[1][i * seq_len : ((i+1) * seq_len - 1)], \
                row[2][i * seq_len : ((i+1) * seq_len - 1)], row[3][i * seq_len : ((i+1) * seq_len - 1)]]
             
            loc += 1
        
        remain = row[0] % seq_len
        if remain != 0:
            sliced_df.loc[loc] = [remain, row[1][slice_num * seq_len : ], row[2][slice_num * seq_len : ], \
                row[3][slice_num * seq_len :]]
            loc += 1
            
    assert loc == len(sliced_df), "loc error in slice_df"
    
    # discard old uncliced long seq
    df = df[df["seq_len"] <= seq_len]
    df = df[df["seq_len"] >= 3] 
    # append new df
    df = df.append(sliced_df, ignore_index=True)
    
    return df

# dataset = CustomDataset("assist09_train.csv", [123, 167, 17905], 20)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch_idx, (hist_seq, hist_answers, new_seq, target_answers) in enumerate(data_loader):
#     if batch_idx == 1:
#         print("{}, {}, {}, {}".format(hist_seq.shape, hist_answers.shape, new_seq.shape, target_answers.shape))
        # print("{}, {}, {}".format(hist_seq, new_seq, target_answers))

# for batch_index, (hist_seq, new_seq, target_answers) in enumerate(train_loader):

# df = get_dataframe("data/assist09/assist09_train.csv")
# print("{} | {} | {} | ".format(len(df), len(df[df["seq_len"] == 20]), len(df[df["seq_len"] > 20]), len(df[df["seq_len"] < 3])))
# df = slice_discard_seq(df, 20)
# print("{} | {} | {} | ".format(len(df), len(df[df["seq_len"] == 20]), len(df[df["seq_len"] > 20]), len(df[df["seq_len"] < 3])))
# print(df[df["seq_len"] == 20].tail(5))
# print(type(df["question"][0])) # numpy.ndarray