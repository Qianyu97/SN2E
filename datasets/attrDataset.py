from random import sample
import numpy as np
from pandas import DataFrame

from finaldata import FinalData
from utils.unit import Indexer_SN2E
from torch.utils.data import Dataset

class attrDataset(Dataset):
    def __init__(self, 
                 nodeList_idx:list[int],
                 attrDF_idx:DataFrame,
                 upperDF_idx:DataFrame,
                 negtDict_idx:dict[int, set[int]], 
                 negtsample_num=10):
        super(Dataset, self).__init__()
        self.len = len(nodeList_idx)
        self.attrArray = attrDF_idx.sort_index().to_numpy()
        self.upperArray = upperDF_idx.sort_index().to_numpy()
        self.nodeList_idx  = nodeList_idx
        self.negtDict_idx  = negtDict_idx
        self.negtsample_num = negtsample_num
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, items):
        #attrdata = self.attrNumpy[items]
        #upperdata = self.upperNumpy[items]
        negdata = self.negtDict_idx[items]
        negdata = np.asarray(sample(list(negdata), self.negtsample_num)) 
        return [np.asarray(items),negdata] # type: ignore  

def main():
    finaldata = FinalData(data_dir='source/data/')
    myIndex = Indexer_SN2E(
        nodeList=finaldata.nodeList,
        attrList=finaldata.attrList
        )
    attrdataset = attrDataset(
        myIndex.str2num(finaldata.nodeList), 
        myIndex.str2num(finaldata.negtDict),
        negtsample_num = 5
        )
    for i in range(10):
        print(attrdataset[i])
    a = 0



