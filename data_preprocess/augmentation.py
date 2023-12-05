from tsaug import *
import numpy as np

class PERMUTE():   
    def __init__(self, min_segments=2, max_segments=15, seg_mode="random"):
        self.min = min_segments
        self.max = max_segments
        self.seg_mode = seg_mode
        
    def augment(self, x):
        # input : (N, T, C)
        # Note 'reshape' and 'swapaxes/transpose' are different 

        orig_steps = np.arange(x.shape[1])
        num_segs = np.random.randint(self.min, self.max , size=(x.shape[0]))
        ret = np.zeros_like(x)

        # for each sample
        for i, pat in enumerate(x):
            if num_segs[i] > 1:
                if self.seg_mode == "random":
                    split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                    split_points.sort()                
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs[i])
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[warp, : ]
            else:
                ret[i] = pat

        return ret 
    
class SCALE():
    def __init__(self, sigma=1.1, loc = 1.3):
        self.sigma = sigma
        self.loc = loc
    def augment(self, x):
        # https://arxiv.org/pdf/1706.00527.pdf
        # loc -> multification #
        factor = np.random.normal(loc=self.loc, scale=self.sigma, size=(x.shape[0], x.shape[2]))
        ai = []
        for i in range(x.shape[1]):
            xi = x[:, i, :]
            ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
        return np.concatenate((ai), axis=1)

def select_transformation(aug_method, seq_len):
    if(aug_method == 'AddNoise'):
        my_aug = AddNoise(scale=0.01)
    elif(aug_method == 'Convolve'):
        my_aug = Convolve(window="flattop", size=11)
    elif(aug_method == 'Crop'):
        my_aug = PERMUTE(min_segments=1, max_segments=5, seg_mode="random")
    elif(aug_method == 'Drift'):
        my_aug = Drift(max_drift=0.7, n_drift_points=5)
    elif(aug_method == 'Dropout'):
        my_aug = (Dropout(p=0.1,fill=0))       
    elif(aug_method == 'Pool'):
        my_aug = Pool(kind='max',size=4)
    elif(aug_method == 'Quantize'):
        my_aug = Quantize(n_levels=20)
    elif(aug_method == 'Resize'):
        my_aug = SCALE(sigma=1.1, loc = 2.)
    elif(aug_method == 'Reverse'):
        my_aug = Reverse()
    elif(aug_method == 'TimeWarp'):
        my_aug = TimeWarp(n_speed_change=5, max_speed_ratio=3)

