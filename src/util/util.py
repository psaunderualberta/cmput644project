import pandas as pd
from tqdm import tqdm

def load_data(files):
    dfs = []
    for f in tqdm(files):
        dfs.append(pd.read_csv(f))
    
    return pd.concat(dfs, axis=0, ignore_index=True)