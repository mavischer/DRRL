import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def paths_from_savedir(dirpath):
    files = os.listdir(dirpath)
    net_path = [f for f in files if f.startswith("net")]
    stats_path = [f for f in files if f.startswith("stats")]
    if len(net_path)>1 or len(stats_path)>1:
        raise Exception(f"More than one savepoint found in {os.path.abspath(dirpath)}")
    net_path = None if len(net_path)==0 else os.path.join(dirpath, net_path[0])
    stats_path = None if len(stats_path)==0 else os.path.join(dirpath, stats_path[0])
    return net_path, stats_path

def append_cumsteps(df):
    """Append columns with episode steps (sum over workers) and global steps (cumulative sum over episode steps)."""
    df = df.copy()
    df["global steps"] = df["steps"].cumsum()
    df["global steps"] = df.groupby(["global ep"])["global steps"].transform(max)
    return df

def aggregate_df(df):
    n_workers = sum(df["global ep"] == 0) # how many workers worked in parallel?
    df = df.copy()
    df = df.groupby("global ep", as_index=False).aggregate([np.mean, np.var]).reset_index()
    df["cumulative steps"] = np.cumsum(df["steps", "mean"])*n_workers
    return df

data1 = pd.read_csv(paths_from_savedir("saves/1e4_newR")[1]).drop(columns=["Unnamed: 0"])
data2 = pd.read_csv(paths_from_savedir("saves/w19")[1]).drop(columns=["Unnamed: 0"])

data = append_cumsteps(data)
#
# agg_1e4 = aggregate_df(data_1e4)
# agg_1e5 = aggregate_df(data_1e5)

measures = ["cumulative reward", "loss", "steps"]
plt.figure()
for i, measure in enumerate(measures):
    plt.subplot(len(measures),1,i+1)
    sns.lineplot(x="global steps", y=measure, data=data)
