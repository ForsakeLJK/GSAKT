from torch.utils.data import Dataset, DataLoader
from utils import get_dataframe, get_metadata
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_dir, metadata, seq_len):
        super(CustomDataset, self).__init__()
        self.df = get_dataframe(csv_dir)
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
        
        return hist_seq, hist_answers, new_seq, target_answers
    
# dataset = CustomDataset("assist09_train.csv", [123, 167, 17905], 20)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch_idx, (hist_seq, hist_answers, new_seq, target_answers) in enumerate(data_loader):
#     if batch_idx == 1:
#         print("{}, {}, {}, {}".format(hist_seq.shape, hist_answers.shape, new_seq.shape, target_answers.shape))
        # print("{}, {}, {}".format(hist_seq, new_seq, target_answers))

# for batch_index, (hist_seq, new_seq, target_answers) in enumerate(train_loader):