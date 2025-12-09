import os
import numpy as np
import pandas as pd

data_dir = os.path.join("weights", "syn_data")
dataset = "FD004"
netname = "DiffUnet"
window = 96

for filename in os.listdir(data_dir):
    data_path = os.path.join(data_dir, filename)
    data = np.load(data_path)
    print(filename, data["data"].shape[0])