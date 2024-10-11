import scipy as sci
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_metrics(filename):
    data = pd.read_csv(filename)
    sns.set_theme()
    fig,ax1 = plt.subplots()
    data = data[:250]
    sns.lineplot(x='epoch', y='loss', data=data,  color='red', ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(x='epoch', y='f1', data=data, ax=ax2, color='blue')
    ax1.set_ylabel('Loss', color='red')
    ax2.set_ylabel('F1', color='blue')


draw_metrics('./val_results.csv')
plt.show()

