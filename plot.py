import os
from os.path import join
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import pdb

GAME = "Breakout"
PLOT_NAME = 'breakout_training_reward_nature.png'
OUTPUT_DIR = 'plots'
METRIC = "Total Unclipped Train Reward"
Y_LABEL = "Training Reward"

dfs = []
labels = []
## Breakout - DQN
#df = pd.read_csv("experiments/only_logs_csv/breakout/dqn/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("DQN")

#df = pd.read_csv("experiments/only_logs_csv/breakout/dan_unclipped/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("DQN")

## Breakout - Averaged-DQN
#df = pd.read_csv("experiments/only_logs_csv/breakout/avg_dqn/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=10)")

#df = pd.read_csv("experiments/only_logs_csv/breakout/avg_dqn_unclipped/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=10)")

# Asterix - DQN
#df = pd.read_csv("experiments/only_logs_csv/asterix/dqn/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("DQN")

#df = pd.read_csv("experiments/only_logs_csv/asterix/dqn_unclipped/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("DQN")

## Asterix - Averaged-DQN
#df = pd.read_csv("experiments/only_logs_csv/asterix/avg_dqn/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=10)")

#df = pd.read_csv("experiments/only_logs_csv/asterix/avg_dqn_unclipped/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=10)")

#df = pd.read_csv("experiments/only_logs_csv/asterix/avg_dqn_k_5/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=5)")

#df = pd.read_csv("experiments/only_logs_csv/asterix/avg_dqn_k_2/logs/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=2)")

## Seaquest - Averaged-DQN
#df = pd.read_csv("experiments/only_logs_csv/seaquest/avg_dqn_k_2/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=2)")

#df = pd.read_csv("experiments/only_logs_csv/seaquest/avg_dqn_k_5/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=5)")

#df = pd.read_csv("experiments/only_logs_csv/seaquest/avg_dqn_k_15/events.csv")
#dfs.append(df[df['metric'] == METRIC])
#labels.append("Averaged-DQN (K=15)")

##### Nature Hyper-parameters #####
df = pd.read_csv("experiments/only_logs_csv/hyper_params_nature/dqn/logs/events.csv")
dfs.append(df[df['metric'] == METRIC])
labels.append("DQN")

df = pd.read_csv("experiments/only_logs_csv/hyper_params_nature/avg_dqn_wrong/logs/events.csv")
dfs.append(df[df['metric'] == METRIC])
labels.append("Averaged-DQN (K=10)")



# Take minmum frames
min_frames = min([df.iloc[-1]['step'] for df in dfs])

for df, label in zip(dfs, labels):
    ax = sns.lineplot(data=df[df['step'] < min_frames], x="step", y="value", label=label)
    ax.set(xlabel='Frame', ylabel=Y_LABEL)
ax.set_title(GAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(join(OUTPUT_DIR, PLOT_NAME))
