from tsaug import *

def select_transformation(aug_method, seq_len):
    if(aug_method == 'AddNoise'):
        my_aug = AddNoise(scale=0.01)
    elif(aug_method == 'Convolve'):
        my_aug = Convolve(window="flattop", size=11)
    elif(aug_method == 'Crop'):
        my_aug = PERMUTE(min_segments=1, max_segments=5, seg_mode="random")
   


class PERMUTE():   
    def __init__(self, min_segments=2, max_segments=15, seg_mode="random"):
        self.min = min_segments
        self.max = max_segments
        self.seg_mode = seg_mode