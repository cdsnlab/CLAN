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
def casasLoader(file_name):
    print("Loading Casas Dataset")
    # variable initialization
    file_list = [] # store file names
    current_label = 0 # current label
    current_time = 0 # current time
    # return variable (an object list)
    dataset_list = []
    # show how labels are displayed
    label_list = []
    # sensor types
    item_list = []
    state_list = []

    # extract file names
    for x in glob(file_name):
        file_list.append(x)
    # sorting by file name
    file_list.sort()

    # for find sensor types
    #for file in file_list:
    #    temp_df = pd.read_csv(file, sep = '	', header = None)
    #    temp_df = temp_df.to_numpy()

    #    if(len(temp_df)>0):
    #        for i in range(0, len(temp_df)):
    #            temp_list = list(temp_df[i, 3].split(" "))    

    #            # time            
    #            timestamp = time.mktime(datetime.strptime((temp_df[i, 0]+' '+temp_df[i, 1]), '%Y-%m-%d %H:%M:%S.%f').timetuple())
            
    #            if temp_df[i, 2] not in item_list:
    #                item_list.append(temp_df[i, 2])   
    #            if temp_list[0] not in state_list:
    #                state_list.append(temp_list[0])
    

    item_list=['M19', 'M23', 'M18', 'M01', 'M17', 'D07', 'M21', 'M22', 'M03', 'I04', 'D12', 'I06', 'M26', 'M04', 'M02', 'M07', 'M08', 'M09', 'M14', 'M15', 'M16', 'M06', 'M10', 'M11', 'M51', 'D11', 'M13', 'M12', 'D14', 'D13', 'D10', 'M05', 'D09', 'D15', 'M20', 'M25', 'M24']

    # for each resident
    for rid in range(0,2):
        for file in file_list:
            temp_df = pd.read_csv(file, sep = '	', header = None)
            temp_df = temp_df.to_numpy()

            # at least one ADL exist in the file
            if(len(temp_df)>0):
                activity_list =  np.zeros(len(item_list)) # activity_list[0] for resident 1, activity_list[1] for resident 2 
                temp_dataset = np.array([activity_list]) 

                # for each row 
                for i in range(0, len(temp_df)):                    
                    temp_list = list(temp_df[i, 3].split(" ")) # 0 : State, 1:Resident# , 2: Resedient_A1, 3: Resident#, 4: Resident_A2 
                    
                    # if the row is related to the resident
                    if(len(temp_list)==3 and int(temp_list[1])-1 == rid) or (len(temp_list)>3 and (int(temp_list[1])-1 == rid or int(temp_list[3])-1 == rid)):
                        print(rid,":", file, i, temp_list)

                        if(current_label==0):                           
                            # for the first row
                            if(int(temp_list[1])-1 == rid):
                                current_label =  int(temp_list[2])  # 2 column is the label
                            elif(int(temp_list[3])-1 == rid):
                                current_label =  int(temp_list[4])    
                            
                            if(temp_list[0] in ['ON', 'OPEN', 'PRESENT']): # when ['ON', 'OPEN', 'PRESENT']
                                activity_list[item_list.index(temp_df[i, 2])] = 1    
                            else:   # when ['OFF', 'ABSENT', 'CLOSE']
                                activity_list[item_list.index(temp_df[i, 2])] = 0
                            
                            temp_dataset = np.array([activity_list]) # sensor sequence                   
                        
                        # if the same activity continue                
                        if((current_label == int(temp_list[2])) or (len(temp_list)>3 and current_label == int(temp_list[4]))):                            
                            if(temp_list[0] in ['ON', 'OPEN', 'PRESENT']): # when ['ON', 'OPEN', 'PRESENT']
                                activity_list[item_list.index(temp_df[i, 2])] = 1    
                            else:   # when ['OFF', 'ABSENT', 'CLOSE']
                                activity_list[item_list.index(temp_df[i, 2])] = 0                           
                            temp_dataset = np.concatenate((temp_dataset, [activity_list]), axis=0)
                        # if the activity is finished (new activity arrival)
                        else:                        
                            if(len(temp_dataset)>len_th):
                                # construct new object(for old activity)          
                                dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                                print(rid, current_label, len(temp_dataset))
                                # just for show 
                                label_list.append(current_label)          
                                            
                            # new activity append (likely the first row)
                            activity_list = [0 for k in range(len(item_list))]
                            if(temp_list[0] in ['ON', 'OPEN', 'PRESENT']): # when ['ON', 'OPEN', 'PRESENT']
                                activity_list[item_list.index(temp_df[i, 2])] = 1    
                            else:   # when ['OFF', 'ABSENT', 'CLOSE']
                                activity_list[item_list.index(temp_df[i, 2])] = 0
                            temp_dataset = np.array([activity_list])                                 
                            if(int(temp_list[1])-1 == rid):
                                current_label =  int(temp_list[2]) # 2 column is the label
                            elif(int(temp_list[3])-1 == rid):
                                current_label =  int(temp_list[4])   
                                 
                if(len(temp_dataset)>len_th):
                    # for the last activity
                    dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                    # just for show
                    label_list.append(current_label)
                    print(rid, current_label, len(temp_dataset))

    return dataset_list

# ARAS data format : (for each second) sensor type+context name,..., activity label1, activity label2/ file name = day 
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
def arasLoader(file_name):   
    # variable initialization
    file_list = [] # store file names
    current_label = [0,0] # current label
    current_time = 0 # current time
    # return variable (an object list)
    dataset_list = []
    # show how labels are displayed
    label_list = []

    # extract file names
    for x in glob(file_name):
        file_list.append(x)
    # sorting by file name
    file_list.sort()

    # for each file
    for file in file_list :
        temp_df = pd.read_csv(file, sep = ' ', header = None)
        temp_df = temp_df.to_numpy()
        # at least one ADL exist in the file
        if(len(temp_df)>0):
            # for the first row
            current_label[0] = temp_df[0, 20] # 20 column is the label of resident1
            current_label[1] = temp_df[0, 21] # 21 column is the label of resident2            
            
            temp_dataset = np.array([temp_df[0,0:19]]) # 0-19 column is the sensors       
            current_time = 0 
            # for each row 
            for i in range(1, len(temp_df)):
                # for each timespan sec
                if((i-current_time) >= (timespan/1000)): 
                    current_time = i
                    # if the same activity continue                
                    if((current_label[0] == temp_df[i, 20]) and (current_label[1] == temp_df[i, 21])):                           
                        temp_dataset = np.concatenate((temp_dataset, [np.array(temp_df[i,0:19])]), axis=0)    
                    # if the activity is finished (new activity arrival)                   
                    else:
                        if(len(temp_dataset)>len_th):
                            # construct new object(for old activity)          
                            dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                            # just for show 
                            label_list.append(current_label)      
                                        
                        # new activity append (likely the first row)
                        temp_dataset = np.array([temp_df[i,0:19]])                                   
                        current_label[0] = temp_df[i, 20] 
                        current_label[1] = temp_df[i, 21]

            if(len(temp_dataset)>len_th):
                # for the last activity
                dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                # just for show
                label_list.append(current_label)

    return dataset_list

# Opportunity data format : sensor type+context name,..., activity label(ADL)/ file name = each user(4) * 5
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
def opportunityLoader():
    print("Loading Opportunity Dataset")

    return 'train', 'test'