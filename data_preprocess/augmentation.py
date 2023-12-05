from tsaug import *

def select_transformation(aug_method, seq_len):
    if(aug_method == 'AddNoise'):
        my_aug = (AddNoise(scale=0.01))
    elif(aug_method == 'Convolve'):
        my_aug = Convolve(window="flattop", size=11)