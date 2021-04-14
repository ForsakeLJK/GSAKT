import numpy as np
import pandas as pd
from utils import get_dataframe, NumpyEncoder, get_metadata
import json


train_dir = "assist09_train.csv"
test_dir = "assist09_test.csv"
skill_matrix_dir = "assist09_skill_matrix.txt"

df_train = get_dataframe(train_dir)
df_test = get_dataframe(test_dir)
# use this to extract the whole q-s graph
df_total = pd.concat([df_train, df_test], ignore_index=True)
skill_matrix = np.loadtxt(skill_matrix_dir)

single_skill_cnt, skill_cnt, max_idx = get_metadata(skill_matrix, df_total)

print("single skill: 0 ~ {}, multi-skill: {} ~ {}, question: {} ~ {}, correctness: {} and {}"\
      .format(single_skill_cnt - 1, single_skill_cnt, skill_cnt - 1, skill_cnt, max_idx - 2, max_idx - 1, max_idx))

# graph -> list of dict
# node: {"type": "skill" or "question", "neighbor": [indices]}  
qs_graph = []
# TODO: is it feasible to get rid of multi-skills?
# init graph 
node_cnt = single_skill_cnt + max_idx - 2 - skill_cnt + 1
for i in range(node_cnt):
    if i >= 0 and i < single_skill_cnt:
        qs_graph.append({"type":"skill", "neighbor":[]})
    else:
        qs_graph.append({"type":"question", "neighbor":[]})

# * 遍历 dataframe 的每一行, 再遍历每行的每一对 (question, skill)
# * 因为剔除了 multi-skill 所以检索节点和输入 neighbor 信息的时候要注意
for seq_idx, seq in df_total.iterrows():
    # * iterate (question, skill)
    seq_len = seq["seq_len"]
    for idx in range(seq_len):
        # ! Note: here q is modified since multi-skills are discarded in qs_graph
        q = seq["question"][idx] - (skill_cnt - single_skill_cnt)
        s = seq["skill"][idx]
        
        q_node = qs_graph[q]
        
        
        assert q_node["type"] == "question", "{} is not a question!".format(q)
        assert s < skill_cnt, "{} is not a skill!".format(s)
        
        
        
        # multi-skill
        if s >= single_skill_cnt and s < skill_cnt:
            # look up in skill matrix
            s_indices = np.argwhere(skill_matrix[s] == 1)
            s_indices = np.reshape(s_indices, s_indices.shape[0])
            
            assert s_indices.shape[0] != 0, "multi-skill {} has no constituent!".format(s)
            
            # add each s to q
            q_node["neighbor"].extend(list(set(s_indices)-set(q_node["neighbor"])))
            
            # add q to each s
            for i in s_indices:
                s_node = qs_graph[i]
                if q not in s_node["neighbor"]:
                    s_node["neighbor"].append(q)
        # single skill, just add neighbor
        else:
            s_node = qs_graph[s]
            # add s to q
            if s not in q_node["neighbor"]:
                q_node["neighbor"].append(s)
            # add q to s
            if q not in s_node["neighbor"]:
                s_node["neighbor"].append(q)

with open("qs_graph.json", "w") as out:
    json.dump(qs_graph, out, cls=NumpyEncoder)