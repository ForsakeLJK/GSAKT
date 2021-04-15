from torch.utils.data.dataloader import DataLoader
from custom_dataset import CustomDataset
import torch
from model import Model
from tqdm import tqdm
from sklearn import metrics

def main():
    #### parameters ####
    train_dir = "data/assist09/assist09_train.csv"
    test_dir = "data/assist09/assist09_test.csv"
    skill_matrix_dir = "data/assist09/assist09_skill_matrix.txt"
    qs_graph_dir = "data/assist09/assist09_qs_graph.json"
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
    
    model = Model(node_feature_size, hidden_dim, node_feature_size, seq_len, head_num, qs_graph_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    train_set = CustomDataset(train_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = CustomDataset(test_dir, [single_skill_cnt, skill_cnt, max_idx], seq_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    print("training...")
    
    # train_losses = []
    
    for epoch in range(epoch_num):
        
        print("epoch {} start".format(epoch + 1))
        
        model.train()
        
        train_preds = []
        train_targets = []
        
        for _, (hist_seq, hist_answers, new_seq, target_answers) in tqdm(enumerate(train_loader)):
            # * foward pass
            # pred = model(x)
            # (batch_size, seq_len - 1, 1)
            pred = model(hist_seq, hist_answers, new_seq)
            
            train_preds.append(pred)
            train_targets.append(target_answers)
            
            # (batch_size, seq_len - 1, 1) -> (batch_size, seq_len)
            # pred = pred.squeeze()
    
            # * compute loss
            loss = model.loss(pred, target_answers)
            
            # train_losses.append(loss.item())
    
            # * backward pass & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_targets = torch.cat(train_targets)
        train_preds = torch.cat(train_preds).sigmoid()
        
        # train_auc = metrics.roc_auc_score(train_targets, train_preds)
        
        model.eval()
        
        # TODO: split validation set
        valid_auc = evaluate(model, train_loader)
        test_auc = evaluate(model, test_loader)
        
        print("epoch {}: train_auc: {}, valid_auc: {}, test_auc: {}".format(epoch+1, -1, valid_auc, test_auc))
                
    
    return 

def evaluate(model, dataloader):
    preds = []
    targets = []
    
    # with torch.no_grad():
    for _, (hist_seq, hist_answers, new_seq, target_answers) in enumerate(dataloader):
        with torch.no_grad():
            pred = model(hist_seq, hist_answers, new_seq)
        
        targets.append(target_answers)
        preds.append(pred)
    
    targets = torch.cat(targets)
    preds = torch.cat(preds).sigmoid()
    
    score = metrics.roc_auc_score(targets, preds)    
    
    
    return score

if __name__ == '__main__':
    main()
    