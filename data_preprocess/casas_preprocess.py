from glob import glob
import numpy as np
import pandas as pd

class TSDataSet:
    def __init__(self,data, label, length):
        self.data = data
        self.label = int(label)
        self.length= int(length)

# CASAS data format : timestamp(when activated), sensor type+context name, state, user #, activity label : [1-15] 
# file name = day 
# Examples(txt) : 2008-11-10 14:28:17.986759 M22 ON 2 2 
# [1, 4, 6, 9, 10, 12, 14, 13, 2, 3, 7, 8, 11, 15, 5] : activity #, [50, 26, 27, 24, 26, 34, 26, 43, 26, 26, 26, 24, 27, 26, 17] : # of activities
def casasLoader(file_name,timespan, min_seq):

    print("Loading CASAS Dataset--------------------------------------")

    # for storing file names
    file_list = [] 
    
    