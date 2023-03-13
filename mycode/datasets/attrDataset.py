from random import sample
import numpy as np

from config import ModelArg, DataloaderArg
from refine import FineData
from torch.utils.data import Dataset
from utils import *

class attrDataset(Dataset):
    def __init__(self, indexdata:FineData) -> None:
        super(Dataset, self).__init__()
        self.len = len(indexdata.fulllist)
        self.indexdata  = indexdata
        self.padAttrs   = [0]*DataloaderArg.len_attr
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, items):
        items += 1
        posdata = self.indexdata.homoDF.get(items, self.padAttrs)
        negdata = sample(self.indexdata.negtdict[items], DataloaderArg.negtsamplenum)
        return [np.asarray(items), np.asarray(negdata), np.asarray(posdata)] # type: ignore  
        



