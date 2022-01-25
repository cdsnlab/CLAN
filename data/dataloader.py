from fileinput import filename
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import csv
from glob import glob

# Static variable
timespan = 1000 # for each timespan sec (1000==1 sec)
len_th = 10 # minimum sequence length

# for storing dataset element
class TSDataSet:
    def __init__(self,data, label, length):
        self.data = data
        self.label = label
        self.length= length
        
# Lapras data format : Sensor type, context name, start time, end time / file name = activity label
# Examples(csv) : Seat Occupy,1,1.490317862115E12,1.490319250294E12,23.136316666666666
def laprasLoader():
    
    print("Loading Lapras Dataset")

    return 'train', 'test'

# CASAS data format : timestamp(when activated), sensor type+context name, state, user #, activity label / file name = day 
# Examples(txt) : 2008-11-10 14:28:17.986759 M22 ON 2 2 
def casasLoader():
    print("Loading Casas Dataset")

    return 'train', 'test'

# ARAS data format : (for each second) sensor type+context name,..., activity label1, activity label2/ file name = day 
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
def arasLoader():
    print("Loading Aras Dataset")

    return 'train', 'test'

# Opportunity data format : sensor type+context name,..., activity label(ADL)/ file name = each user(4) * 5
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
def opportunityLoader():
    print("Loading Opportunity Dataset")

    return 'train', 'test'