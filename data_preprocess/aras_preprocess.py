from glob import glob
import numpy as np
import pandas as pd

class TSDataSet:
    def __init__(self,data, label, length):
        self.data = data
        self.label = int(label)
        self.length= int(length)

# ARAS data format : (for each second) sensor type+context name,..., activity label1, activity label2 : [1-27]/ file name = day 
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
# types_label : [12, 17, 3, 4, 13, 27, 2, 22, 7, 15, 8, 26, 10, 11, 9, 18, 14, 25, 21, 5, 1, 6, 23, 24, 16, 20, 19]
# count_label : [149, 83, 31, 25, 41, 10, 23, 29, 26, 51, 20, 5, 38, 25, 40, 11, 28, 9, 8, 23, 25, 14, 7, 1, 9, 3, 1]
# types_label : [3, 4, 1, 13, 18, 5, 6, 9, 10, 12, 11, 25, 7, 17, 21, 27, 14, 2, 24, 8, 16]
# count_label : [11, 10, 11, 27, 7, 4, 4, 1, 3, 31, 7, 6, 8, 10, 1, 4, 1, 1, 1, 1, 1]
# After refining (min sample remove) ARAS A
# original label: [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 22] 
# changed label: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# After refining (min sample remove) ARAS B
# types_label : [2, 1]
# count_label : [31, 31]


def arasLoader(file_name, timespan, min_seq):

    print("Loading ARAS Dataset--------------------------------------")

    # variable initialization
    file_list = [] # store file names
    current_label = [0, 0] # current label
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
        temp_df = pd.read_csv(file, sep = ' ', header = None).to_numpy()

        # at least one ADL exist in the file
        if(len(temp_df)>0):
            # for the first row
            current_label[0] = temp_df[0, 20] # 20 column is the label of resident1            
            current_label[1] = temp_df[0, 21] # 21 column is the label of resident2 

            emp_dataset = np.array([temp_df[0,0:20]]) # 0-19 column is the sensors  
            current_datalist = temp_df[0,0:19]             
            current_time = 0 
            
            # for each row 
            for i in range(1, len(temp_df)):
                # for each timespan sec
                if((i-current_time) >= (timespan/1000)):
                    current_time = i
                    # if the same activity continue                
                    if((current_label[0] == temp_df[i, 20]) and (current_label[1] == temp_df[i, 21])):
                        if (current_datalist !=  temp_df[i,0:20]).any():                   
                            temp_dataset = np.concatenate((temp_dataset, [np.array(temp_df[i,0:20])]), axis=0)
                            current_datalist =  temp_df[i,0:20] 
                    # if the activity is finished (new activity arrival)                   
                    else:
                        if(len(temp_dataset)>min_seq):
                            # construct new object(for old activity)
                            if(current_label[0] != temp_df[i, 20]):  # first resident's activity is changed
                                dataset_list.append(TSDataSet(temp_dataset, (current_label[0]), len(temp_dataset)))
                            else: # second resident's activity is changed
                                dataset_list.append(TSDataSet(temp_dataset, (current_label[1]), len(temp_dataset)))
                            # just for show 
                            label_list.append(current_label)  
                        
                        # new activity append (likely the first row)
                        temp_dataset = np.array([temp_df[i,0:20]])                                   
                        current_label[0] = temp_df[i, 20] 
                        current_label[1] = temp_df[i, 21]

            if(len(temp_dataset)>min_seq):
                # for the last activity
                if(current_label[0] != temp_df[i, 20]):          
                    dataset_list.append(TSDataSet(temp_dataset, (current_label[0]), len(temp_dataset)))                
                else:
                    dataset_list.append(TSDataSet(temp_dataset, (current_label[1]), len(temp_dataset)))  

                label_list.append(current_label)            
                
    print("Loading ARAS Dataset Finished--------------------------------------")