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