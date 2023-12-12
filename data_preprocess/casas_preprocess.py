from glob import glob
import numpy as np
import pandas as pd
import time, datetime

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
    
    # extract file names
    for x in glob(file_name):
        file_list.append(x)
    # sort list by file name
    file_list.sort() 
    
    # for finding sensor types
    sensor_list = []

    # for find sensor types
    item_list, state_list = [], []

    # for find sensor types
    # for file in file_list:
    #    temp_df = pd.read_csv(file, sep = ',', header = None).to_numpy() 
    # #if the file is not empty
    #    if(len(temp_df)>0):
    #        for i in range(0, len(temp_df)):
    #            temp_list = list(temp_df[i, 3].split(" "))    

    #            # time            
    #            timestamp = time.mktime(datetime.strptime((temp_df[i, 0]+' '+temp_df[i, 1]), '%Y-%m-%d %H:%M:%S.%f').timetuple())
            
    #            if temp_df[i, 2] not in item_list:
    #                item_list.append(temp_df[i, 2])   
    #            if temp_list[0] not in state_list:
    #                state_list.append(temp_list[0]) 
    
    current_label = 0 # current label

    # for constructing dataset's data structure (return variable : an object list)
    dataset_list = []
    # show how labels are displayed
    label_list = []

    sensor_list = ['M19', 'M23', 'M18', 'M01', 'M17', 'D07', 'M21', 'M22', 'M03', 'I04', 'D12', 'I06', 'M26', 'M04', 'M02', 'M07', 'M08', 'M09', 'M14', 'M15', 'M16', 'M06', 'M10', 'M11', 'M51', 'D11', 'M13', 'M12', 'D14', 'D13', 'D10', 'M05', 'D09', 'D15', 'M20', 'M25', 'M24']
    
    # construct dataset's data structure 
    # for each resident
    for rid in range(0,2):