# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:06:48 2016

@author: gAkos
"""
import pandas as pd
import numpy as np
import csv

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
trial_data_file = 'trials.csv'
trial_df = pd.read_csv(trial_data_file)

trial_df['color'] = np.where(trial_df['reached']==True, 'green', 'red')
trial_df['efficiency'] = trial_df['steps'] / trial_df['goal_distance']

trial_df.head()

# Define list of alphas (learning rate) to test 
alphas = [0.25, 0.5, 0.75, 1.0] 
# Define list of gammas (future discount factor) to test
gammas = [0.0, 0.25, 0.5, 0.75, 1.0]

# Split dataframe into new df per column
df = pd.DataFrame()
for a in alphas:
    df_a = trial_df.where(trial_df['alpha']==a).dropna()
    for g in gammas:
        series_g = df_a['efficiency'].where(df_a['gamma']==g).dropna().reset_index(drop=True)
        column_name = "{0},{1}".format(a,g)
        df[column_name] = series_g
#print df


# specify plot style and set color scheme
x = df.index
with sns.color_palette("RdBu_r", len(df.columns)):
    # plot speed rate of each trial
    plt.figure(figsize=(10,6))
    for col in df.columns:
        z = np.polyfit(df.index,df[col],3)
        p = np.poly1d(z)
        plt.plot(x, p(x),linewidth=0.5)
plt.legend(labels=df.columns.values, loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Iteration')
plt.ylabel('Efficiency (Steps / Distance to Goal)')
plt.show()

###
# Split dataframe into new df with means (alpha x: gamma y)

df_mean = pd.DataFrame()
for a in alphas:
    gamma_means = []
    df_a = trial_df.where(trial_df['alpha']==a).dropna()
    for g in gammas:
        gamma_means.append(np.mean(df_a['efficiency'].where(df_a['gamma']==g).dropna().reset_index(drop=True)))
    df_mean[a]=gamma_means

# visualize means with heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_mean, yticklabels=['Gamma '+str(y) for y in gammas], xticklabels=['Alpha '+ str(x) for x in alphas], annot=True, linewidth=.1,  fmt='.1f', cmap='RdBu_r')
plt.title('Mean Efficiencies')
plt.xticks(rotation=90, ha='center');
plt.show()
        