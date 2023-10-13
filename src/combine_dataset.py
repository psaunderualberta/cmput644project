import pandas as pd
from util.constants import (
    RAW_DATA_FILES,
    COMBINED_DATA_FILE,
    CLASSES_8_MAPPING,
    CLASSES_2_MAPPING,
    CLASSES_8_Y_COLUMN,
    CLASSES_2_Y_COLUMN,
    CLASSES_34_Y_COLUMN,
)
from tqdm import tqdm
import time

# Read in the data
dfs = []
for file in tqdm(RAW_DATA_FILES):
    dfs.append(pd.read_csv(file))

# Append the columns for the 8-class and 2-class task
t = time.time()
df = pd.concat(dfs)
df[CLASSES_8_Y_COLUMN] = df[CLASSES_34_Y_COLUMN].apply(lambda x: CLASSES_8_MAPPING[x])
df[CLASSES_2_Y_COLUMN] = df[CLASSES_34_Y_COLUMN].apply(lambda x: CLASSES_2_MAPPING[x])
print("Time to append columns: {:.2f}s".format(time.time() - t))

# Combine the data and output to csv
t = time.time()
df.to_csv(COMBINED_DATA_FILE, index=False)
print("Time to combine data: {:.2f}s".format(time.time() - t))
