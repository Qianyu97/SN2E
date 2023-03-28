from random import sample
import numpy as np

from config import ModelArg, DataloaderArg
from refine import FineData
from torch.utils.data import Dataset
from utils import *

class attrDataset(Dataset):
    def __init__(self, indexdata:FineData) -> None:
        super(Dataset, self).__init__()
        self.len = len(indexdata.defilist)
        self.indexdata  = indexdata
        self.attrDF     = indexdata.homoDF.T 
        self.negtDF     = indexdata.negtDF.T
        self.negtnum    = indexdata.negtnum 
        self.negtsample_num = DataloaderArg.negtsamplenum
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, items):
        items += (ModelArg.model.num_nodefi)
        attrdata = self.attrDF[items]
        negtdata = self.negtDF[items]
        negt_num = self.negtnum[items]
        if negt_num > DataloaderArg.negtsamplenum:
            negtdata = negtdata[:negt_num].sample(n = self.negtsample_num) 
        else:
            negtdata = negtdata[:self.negtsample_num]
        return [np.asarray(items), np.asarray(negtdata), np.asarray(attrdata)] # type: ignore  
        



