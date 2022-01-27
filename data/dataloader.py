from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Static variable
#timespan = 1000 # for each timespan sec (1000==1 sec)
#min_seq = 10 # minimum sequence length
#min_samples = 10 # minimum # of samples

# for storing dataset element
class TSDataSet:
    def __init__(self,data, label, length):
        self.data = data
        self.label = int(label)
        self.length= int(length)

# use for lapras dataset
def label_num(filename):
    label_cadidate = ['Chatting', 'Discussion', 'GroupStudy', 'Presentation', 'NULL']
    label_num = 0
    for i in range(len(label_cadidate)):
        if filename.find(label_cadidate[i]) > 0:
            label_num = i+1    
    return label_num

# use for dataset normalization 
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return round(df_norm,3)


# Lapras data format : Sensor type, context name, start time, end time / file name = activity label : [1 : 'Chatting', 2: 'Discussion', 3: 'GroupStudy', 4: 'Presentation', 5: 'NULL']
# Examples(csv) : Seat Occupy,1,1.490317862115E12,1.490319250294E12,23.136316666666666
#[1, 2, 3, 5, 4] : activity #, [119, 52, 40, 116, 129], # of activities
def laprasLoader(file_name, timespan, min_seq):
    
    print("Loading Lapras Dataset--------------------------------------")
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
    time_list = []

    # extract file names
    for x in glob(file_name):
        file_list.append(x)
    # sorting by file name
    file_list.sort()

    start_time  = 0
    end_time = 0

    # for finding sensor types
    for file in file_list:
        temp_df = pd.read_csv(file, sep = ',', header = None)
        temp_df = temp_df.to_numpy() # 0: sensor type, 1: state, 2: start_time, 3: end_ time
        print(file)
        
        label_list.append(label_num(file))
        # if the file is not empty
        if(len(temp_df)>0):
            start_time = temp_df[0, 2]
            end_time = temp_df[len(temp_df)-1,3]
            # for each row
            for i in range(0, len(temp_df)):
                if(temp_df[i, 2] < start_time):
                    start_time = temp_df[i, 2] 
                if(temp_df[i, 3] > end_time):
                    end_time = temp_df[i, 3]                
                if temp_df[i, 0] not in item_list:
                    item_list.append(temp_df[i, 0])   
                if temp_df[i, 1] not in state_list:
                    state_list.append(temp_df[i, 1])

        time_list.append([start_time, end_time])
    # print(item_list)
    # print(state_list)
    # print(label_list)
    # print(time_list)
    item_list= ['Seat Occupy', 'Sound', 'Brightness', 'Light', 'Existence', 'Projector', 'Presentation']

    count_file = 0
    # for each file
    for file in file_list:
        temp_df = pd.read_csv(file, sep = ',', header = None)
        temp_df = temp_df.to_numpy()

        # at least one ADL exist in the file
        if(len(temp_df)>0):              
            #print(int((time_list[count_file-1][1]-time_list[count_file-1][0])/(timespan)),len(item_list))
            temp_dataset = np.zeros((int((time_list[count_file][1]-time_list[count_file][0])/(timespan)),len(item_list)))
            
            # for each sensor
            for i in range(0, len(temp_df)):
            #print("1", temp_df[i, 3], temp_df[i, 2], time_list[count_file][0] )
            #print(int((temp_df[i, 3]-time_list[count_file][0])/(timespan)), int((temp_df[i, 2]-time_list[count_file][0])/(timespan)))
                for j in range(int((temp_df[i, 2]-time_list[count_file][0])/(timespan)), int((temp_df[i, 3]-time_list[count_file][0])/(timespan))):
                    # count based event
                    if(temp_df[i, 0] == 'Seat Occupy' or temp_df[i, 0] == 'Existence'):                
                        temp_dataset[j][item_list.index(temp_df[i, 0])] += 1
                    # state based event
                    elif(temp_df[i,0] == 'Sound' or temp_df[i,0] == 'Brightness'):
                        temp_dataset[j][item_list.index(temp_df[i, 0])] = int(temp_df[i,1])%10
                    # actiation based event
                    elif(temp_df[i,0] == 'Light' or temp_df[i,0] == 'Projector' or temp_df[i,0] == 'Presentation'):
                        temp_dataset[j][item_list.index(temp_df[i, 0])] = 1

            if(len(temp_dataset)> min_seq):
                current_label = label_list[count_file] # label list  
                dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
        # for next file
        count_file+=1

    print("Loading Lapras Dataset Finished--------------------------------------")
    return dataset_list


# CASAS data format : timestamp(when activated), sensor type+context name, state, user #, activity label : [1-15] / file name = day 
# Examples(txt) : 2008-11-10 14:28:17.986759 M22 ON 2 2 
# [1, 4, 6, 9, 10, 12, 14, 13, 2, 3, 7, 8, 11, 15, 5] : activity #, [50, 26, 27, 24, 26, 34, 26, 43, 26, 26, 26, 24, 27, 26, 17] : # of activities
def casasLoader(file_name,timespan, min_seq):
    print("Loading CASAS Dataset--------------------------------------")
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
            print(file)
            # at least one ADL exist in the file
            if(len(temp_df)>0):
                activity_list =  np.zeros(len(item_list)) # activity_list[0] for resident 1, activity_list[1] for resident 2 
                temp_dataset = np.array([activity_list]) 
                
                # for each row 
                for i in range(0, len(temp_df)):                    
                    temp_list = list(temp_df[i, 3].split(" ")) # 0 : State, 1:Resident# , 2: Resedient_A1, 3: Resident#, 4: Resident_A2 
                    
                    # if the row is related to the resident
                    if(len(temp_list)==3 and int(temp_list[1])-1 == rid) or (len(temp_list)>3 and (int(temp_list[1])-1 == rid or int(temp_list[3])-1 == rid)):
                        #print(rid,":", file, i, temp_list)
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
                            if(len(temp_dataset)>min_seq):
                                # construct new object(for old activity)          
                                dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                                #print(rid, current_label, len(temp_dataset))
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
                if(len(temp_dataset)>min_seq):
                    # for the last activity
                    dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                    # just for show
                    label_list.append(current_label)
                    #print(rid, current_label, len(temp_dataset))
    print("Loading CASAS Dataset Finished--------------------------------------")
    return dataset_list


# ARAS data format : (for each second) sensor type+context name,..., activity label1, activity label2 : [1-27]/ file name = day 
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
# A: [1217, 2217, 1215, 2117, 1517, 1717, 1711, 1111, 1120, 1121, 1115, 1127, 1102, 102, 1202, 302, 402, 1502, 1702, 2102, 1302, 2202, 1902, 2702, 202, 214, 217, 212, 222, 210, 112, 712, 717, 715, 708, 808, 809, 817, 2221, 2212, 1212, 2112, 2622, 2722, 122, 1522, 1527, 2627, 2612, 1012, 2615, 2620, 2601, 2617, 2610, 1210, 1110, 1101, 902, 1002, 2402, 1402, 1422, 1427, 1418, 2525, 1825, 1818, 1812, 2717, 2727, 1722, 1222, 2215, 1214, 1221, 2511, 1511, 2111, 1104, 502, 227, 201, 2712, 1512, 1712, 1707, 707, 818, 1218, 1209, 922, 2222, 1710, 1718, 1211, 602, 1327, 1322, 2522, 1407, 702, 908, 909, 912, 215, 1510, 1509, 1521, 727, 722, 822, 812, 918, 1022, 1017, 1802, 2527, 1727, 1716, 116, 2216, 2214, 114, 1714, 109, 1721, 2121, 121, 115, 802, 1816, 1216, 2218, 2211, 1116, 117, 1715, 1701, 2302, 1516, 1817, 1814, 1810, 2116, 2110, 1112, 1815, 312, 412, 416, 916, 1412, 512, 612, 1827, 101, 1801, 1822, 1709, 1602, 110, 2201, 1109, 1227, 1027, 1015, 1201, 1207, 921, 2210, 1821, 127, 1501, 1010, 1312, 120, 2209, 1807, 107, 1001, 2101, 1614, 1610, 316, 317, 401, 415, 901, 1401, 2701, 1118, 1122, 301, 2118, 207, 1507, 2707, 2207, 1009, 1114, 1011, 1627, 1601, 2227, 1310, 1309, 1301, 1321, 1311, 911, 2311, 718, 716, 216, 915, 2721, 1411, 111, 927, 1315, 209, 221, 2602, 2517, 1103, 2715, 1317, 1325, 2515, 1314, 1125, 2503, 1503, 404, 421, 920, 1720, 1713, 1213, 113, 2213, 1723, 1316, 1708, 2308, 2309, 1518, 322, 422, 2127, 2719, 2312, 1719, 1813, 1220, 2307, 1108, 2512, 522, 517, 617, 917, 1318]
# A: [18, 16, 16, 4, 6, 19, 18, 35, 18, 13, 43, 22, 28, 61, 90, 34, 24, 71, 94, 32, 40, 69, 5, 37, 43, 2, 5, 15, 9, 6, 15, 7, 3, 2, 1, 12, 5, 1, 4, 34, 59, 5, 2, 3, 4, 2, 2, 1, 2, 13, 1, 1, 2, 2, 1, 11, 9, 18, 41, 25, 3, 18, 4, 3, 1, 16, 1, 3, 15, 1, 5, 12, 23, 6, 3, 7, 1, 6, 14, 3, 23, 7, 12, 8, 19, 49, 7, 10, 2, 4, 5, 3, 12, 10, 5, 4, 17, 4, 6, 2, 1, 9, 3, 7, 11, 4, 4, 3, 3, 2, 3, 1, 3, 2, 2, 5, 12, 3, 8, 6, 6, 5, 5, 2, 7, 2, 6, 2, 1, 5, 5, 4, 13, 3, 5, 11, 3, 10, 10, 6, 8, 3, 4, 3, 4, 1, 3, 2, 4, 3, 1, 1, 6, 3, 2, 3, 9, 6, 5, 5, 5, 3, 6, 4, 7, 3, 1, 6, 3, 2, 6, 1, 4, 4, 6, 8, 2, 2, 1, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1, 2, 2, 1, 7, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2, 1, 2, 1, 6, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# B: [1111, 1511, 111, 1115, 1118, 118, 1518, 318, 418, 404, 918, 115, 1318, 1315, 1313, 513, 606, 106, 101, 901, 913, 1316, 1516, 1310, 1018, 1218, 1215, 1212, 1512, 117, 1517, 317, 2525, 2727, 2127, 2121, 121, 202, 1527, 2715, 2701, 701, 801, 808, 1201, 1101, 1117, 127, 2118, 315, 1715, 1804, 1817, 1317, 1321, 1417, 2717, 2101, 1217, 2714, 114, 1814, 1827, 1801, 1501, 2501, 1112, 2711, 2702, 1102, 1127, 2111, 102, 1502, 1103, 1104, 1120, 302, 402, 902, 1802, 1302, 1002, 502, 602, 702, 802, 1202, 2102, 311, 401, 1525, 2515, 518, 515, 112, 1012, 1504, 1211, 1311, 1011, 1818, 1121, 1401, 1702, 1902, 1402, 1015, 1027, 1001, 2112, 411, 1811, 1322, 1304, 517, 915, 717, 815, 1427, 1727, 1701, 2712, 1712, 1227, 304, 1301, 1327, 2302, 917, 1717, 1815, 2415, 2418, 1918, 1418, 2718, 715, 912, 301, 1721, 727, 712, 108, 1708, 1017, 1116, 2716, 316, 104, 120, 920, 718, 1208, 1416, 116, 2211, 921, 2721, 1412, 1513, 413]
# B: [43, 21, 9, 26, 3, 13, 11, 2, 1, 17, 2, 20, 13, 8, 2, 1, 3, 1, 55, 6, 1, 2, 3, 1, 4, 3, 9, 39, 10, 8, 16, 8, 14, 15, 7, 1, 2, 29, 8, 8, 9, 1, 2, 4, 2, 11, 10, 23, 2, 6, 1, 4, 7, 14, 1, 2, 7, 7, 2, 4, 5, 1, 1, 3, 14, 1, 5, 2, 12, 9, 4, 2, 33, 31, 1, 1, 2, 4, 4, 2, 3, 24, 7, 3, 3, 6, 5, 25, 5, 7, 3, 1, 1, 1, 1, 8, 1, 2, 1, 7, 1, 3, 1, 2, 1, 2, 2, 1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 4, 2, 1, 3, 1, 4, 2, 3, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def arasLoader(file_name, timespan, min_seq):   
    print("Loading ARAS Dataset--------------------------------------")
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
        print(file)
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
                        if(len(temp_dataset)>min_seq):
                            # construct new object(for old activity)          
                            dataset_list.append(TSDataSet(temp_dataset, (current_label[0]*100+ current_label[1]), len(temp_dataset)))
                            # just for show 
                            label_list.append(current_label)      
                                        
                        # new activity append (likely the first row)
                        temp_dataset = np.array([temp_df[i,0:19]])                                   
                        current_label[0] = temp_df[i, 20] 
                        current_label[1] = temp_df[i, 21]

            if(len(temp_dataset)>min_seq):
                # for the last activity
                dataset_list.append(TSDataSet(temp_dataset, (current_label[0]*100+ current_label[1]), len(temp_dataset)))
                # just for show
                label_list.append(current_label)
    print("Loading ARAS Dataset Finished--------------------------------------")

    return dataset_list

# Opportunity data format : sensor type+context name,..., activity label(ADL)/ file name = each user(4) * 5
# Examples(txt) : 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
# the number of examples : 101(Relaxing) - 40, 102(Coffee time)-20, 103(Early morning)-20, 104(Cleanup)-20, 105(Sandwich time)-20
# [1, 3, 2, 5, 4] : activity #
# [40, 20, 20, 20, 20] : # of activity 
def opportunityLoader(file_name, timespan, min_seq):
    print("Loading Opportunity Dataset --------------------------------------")
    # variable initialization
    file_list = [] # store file names
    current_label = 0 # current label
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
        # extract data related to the target ADLs (column :244 => 101~105) and convert to numpy array
        temp_df = temp_df[temp_df[244]>100].to_numpy() 
        print(file)

        # at least one ADL exist in the file
        if(len(temp_df)>0):
            # for the first row
            current_label = temp_df[0, 244] # 244 column is the label
            current_time =  temp_df[0, 0] # 0 column is the timestamp
            temp_dataset = np.array([temp_df[0,1:242]]) # 1-242 column is the sensors       

            # for each row 
            for i in range(1, len(temp_df)):
                # for each timespan sec
                if((temp_df[i, 0]-current_time) >= timespan): 
                    current_time = temp_df[i, 0]
                    # if the same activity continue                
                    if(current_label == temp_df[i, 244]):                           
                        temp_dataset = np.concatenate((temp_dataset, [np.array(temp_df[i,1:242])]), axis=0)
                    # if the activity is finished (new activity arrival)
                    else:
                        # construct new object(for old activity)          
                        dataset_list.append(TSDataSet(temp_dataset,  (current_label-100), len(temp_dataset)))
                        # just for show 
                        label_list.append(current_label)          
                                        
                        # new activity append (likely the first row)
                        temp_dataset = np.array([temp_df[i,1:242]])                                   
                        current_label = temp_df[i, 244]

            # for the last activity
            dataset_list.append(TSDataSet(temp_dataset,  (current_label-100), len(temp_dataset)))
            # just for show
            label_list.append(current_label)
    print("Loading Opportunity Dataset Finished--------------------------------------")
    return dataset_list

class TimeseriesDataset(Dataset):   
    def __init__(self, data, window, target_cols):
        self.data = torch.Tensor(data)
        self.window = window
        self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__() 
    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.data[index+self.window,0:target_cols]
        return x, y 
    def __len__(self):
        return len(self.data) -  self.window     
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)    
    def __getsize__(self):
        return (self.__len__())

def visualization_data(dataset_list, file_name, activity_num):
    print("Visualizing Dataset --------------------------------------")
    label_count = [0 for x in range(activity_num)]
    # for visualization
    for k in range(len(dataset_list)):
        visual_df = pd.DataFrame(dataset_list[k].data)

        fig, ax = plt.subplots(figsize=(10, 6))
        axb = ax.twinx()

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)

        # Plotting on the first y-axis
        for i in range(len(dataset_list[0].data[0])):
            ax.plot(visual_df[i], label = str(i+1))

        ax.legend(loc='upper left')
        
        plt.savefig(file_name+'visualization/'+str(dataset_list[k].label)+'_'+str(label_count[dataset_list[k].label-1])+'.png')
        plt.close(fig)
        label_count[dataset_list[k].label-1]+=1

    print("Visualizing Dataset Finished--------------------------------------")

def count_label(dataset_list):
    # finding types and counts of label
    types_label_list =[]
    count_label_list = []
    for i in range(len(dataset_list)):
        if(dataset_list[i].label not in types_label_list):
            types_label_list.append(dataset_list[i].label)
            count_label_list.append(1)
        else:
            count_label_list[types_label_list.index(dataset_list[i].label)]+=1

    print('types_label :', types_label_list)
    print('count_label :', count_label_list)   
                
    return types_label_list, count_label_list

def count_label_labellist(datalist, labellist):
    # finding types and counts of label
    types_label_list =[]
    count_label_list = []
    for i in range(len(datalist)):
        if(labellist[i] not in types_label_list):
            types_label_list.append(labellist[i])
            count_label_list.append(1)
        else:
            count_label_list[types_label_list.index(labellist[i])]+=1

    print('types_label :', types_label_list)
    print('count_label :', count_label_list)   
                
    return types_label_list, count_label_list

def padding_by_max(lengthlist, normalized_df):
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    print("max", max(lengthlist))
    # reconstruction of normalized list
    # for each row
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        for j in range(count_lengthlist,(count_lengthlist+lengthlist[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += lengthlist[i]

        #padding to each data list
        if((max(lengthlist)-lengthlist[i])%2 == 0):
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i])/2), int((max(lengthlist)-lengthlist[i])/2))
        else :
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i]+1)/2)-1, int((max(lengthlist)-lengthlist[i]+1)/2))
        datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant",-1))
    
    return datalist

def padding_by_mean(lengthlist, normalized_df):
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    mean_length = int(sum(lengthlist)/len(lengthlist))
    print("mean", mean_length)
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        if(lengthlist[i]>=mean_length): # length is larger than mean
            for j in range(count_lengthlist, count_lengthlist+mean_length):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            datalist.append(torch.tensor(reconst_list))
        else: # length is smaller than mean
            for j in range(count_lengthlist, (count_lengthlist+lengthlist[i])):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            # padding to the end 
            p2d = (0, 0, 0, mean_length-lengthlist[i])
            datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant",-1))    
        count_lengthlist += lengthlist[i]    
    return datalist

def reconstrct_list(lengthlist, normalized_df):
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    # for each row
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        for j in range(count_lengthlist,(count_lengthlist+lengthlist[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += lengthlist[i]
        datalist.append(torch.tensor(reconst_list))
    return datalist

# split data into train/validate/test 
def splitting_data(dataset, test_ratio, valid_ratio, padding, seed, timespan, min_seq, min_samples): 

    print(timespan, min_seq, min_samples)
    if dataset == 'lapras':
        dataset_list = laprasLoader('data/Lapras/*.csv', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/Lapras/', 5)
    elif dataset == 'casas':
        dataset_list = casasLoader('data/CASAS/*.txt', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/CASAS/', 15)
    elif dataset == 'aras_a':
        dataset_list = arasLoader('data/Aras/HouseA/*.txt', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/Aras/HouseA/', 27*100 + 27)
    elif dataset == 'aras_b':
        dataset_list = arasLoader('data/Aras/HouseB/*.txt', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/Aras/HouseB/', 27*100 + 27)
    elif dataset == 'opportunity':
        dataset_list = opportunityLoader('data/Opportunity/*.dat', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/Opportunity/', 5)
    
    types_label_list, count_label_list = count_label(dataset_list)
    
    # Convert object-list to list-list
    labellist=[]
    # store each length of samples
    lengthlist=[]
    templist=[]
    for i in range(len(dataset_list)):
        # Select datalist by min_samples
        if(count_label_list[types_label_list.index(dataset_list[i].label)]>= min_samples):
            #datalist.append(dataset_list[i].data)        
            labellist.append(dataset_list[i].label)
            lengthlist.append(len(dataset_list[i].data))
            for j in range(len(dataset_list[i].data)):
                templist.append(dataset_list[i].data[j])     
               
    # normalization of dataframe
    normalized_df = min_max_scaling(pd.DataFrame(templist))
    normalized_df = normalized_df.fillna(0)

    # reconstruction of list (padding is option : max or mean)
    if padding == 'max':
        datalist = padding_by_max(lengthlist, normalized_df)
    elif padding =='mean':
        datalist = padding_by_mean(lengthlist, normalized_df)
    else:
        datalist = reconstrct_list(lengthlist, normalized_df)

    count_label_labellist(datalist, labellist)
    # Split train and valid dataset
    train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
    train_list, valid_list, train_label_list, valid_label_list= train_test_split(train_list, train_label_list, test_size=valid_ratio, stratify=train_label_list, random_state=seed)

    print(f"Train Data: {len(train_list)}") 
    count_label_labellist(train_list, train_label_list)
    print(f"Validation Data: {len(valid_list)}")
    count_label_labellist(valid_list, valid_label_list)
    print(f"Test Data: {len(test_list)}") 
    count_label_labellist(test_list, test_label_list)
    return train_list, valid_list, test_list, train_label_list, valid_label_list, test_label_list