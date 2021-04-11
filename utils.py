import pandas as pd
import numpy as np
import json

def get_dataframe(data_dir):
    """Convert Dataset into Dataframe

    Args:
        data_dir (str)

    Returns:
        pandas.DataFrame: dataframe from dataset 
    """
    return pd.read_csv(data_dir, converters={"skill": lambda x: np.fromstring(x,  dtype=int, sep=' '),
                             "question": lambda x: np.fromstring(x,  dtype=int, sep=' '),
                             "correctness": lambda x: np.fromstring(x,  dtype=int, sep=' ')})

def reform_data(old_dir, new_dir):
    """Reformat Old GIKT Dataset into Standard CSV Format

    Args:
        old_dir (str)
        new_dir (str)
    """
    with open(old_dir, 'r') as old, open(new_dir, 'w') as new:
        new.write("seq_len,skill,question,correctness\n")
        for line_id, line in enumerate(old):
            if line_id % 4 == 0: # seq_len
                new.write(line.rstrip() + ',')
            elif line_id % 4 == 3: # correctness
                new.write(line.replace(',', ' '))
            else: 
                new.write(line.replace(',', ' ').rstrip() + ',')

def get_metadata(skill_matrix, df_total):
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
    
    return single_skill_cnt, skill_cnt, max_idx
                
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(self, o)