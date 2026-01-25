from random import sample
import numpy as np

from finaldata import FinalData
from torch.utils.data import Dataset
from utils import *

class attrDataset(Dataset):
    def __init__(self, finaldata:FinalData, negtsample_num=10) -> None:
        super(Dataset, self).__init__()
        self.len = len(finaldata.defiList_idx)
        self.indexed_data  = finaldata.defiList_idx
        self.attrDF     = finaldata.attrDF_idx.T
        self.negtDict   = finaldata.negtDict_idx
        self.upperdata  = finaldata.upperDict_idx
        self.negtsample_num = negtsample_num
        self.padAttr    = [0]*self.attrDF.shape[0]
        self.padPare    = [0]
        self.padNegt    = [1]    
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, items):
        #items += (ModelArg.model.num_nodefi)
        items += 1
        attrdata = self.attrDF.get(items, self.padAttr)
        upperdata = self.upperdata.get(items, 0)
        negdata = self.negtDict[items]
        negdata = sample(list(negdata), self.negtsample_num) 
        return [np.asarray(items), np.asarray(negdata), np.asarray(attrdata), np.asarray([upperdata])] # type: ignore  

def main():
    finaldata = FinalData(data_dir='source/data/')
    attrdataset = attrDataset(finaldata, negtsample_num=5)
    for i in range(10):
        print(attrdataset[i])



