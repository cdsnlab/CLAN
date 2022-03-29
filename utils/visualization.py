from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.dataloader import count_label_labellist
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import cm
import torch

def early_stopping_visualization(train_loss, valid_loss):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # validation loss의 최저값 지점을 찾기
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # 일정한 scale
    plt.xlim(0, len(train_loss)+1) # 일정한 scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches = 'tight')

def fill_graph():
    fig = plt.figure(figsize=(10,8))
    x_values = range(10)
    y_values = [10, 12, 13, 13, 15, 19, 20, 22, 23, 29]
    y_lower = [8, 10, 11, 11, 13, 17, 18, 20, 21, 27]
    y_upper = [12, 14, 15, 15, 17, 21, 22, 24, 25, 31]
    
    plt.fill_between(x_values, y_lower, y_upper, alpha=0.2) #this is the shaded error
    plt.plot(x_values, y_values) #this is the line itself
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches = 'tight')