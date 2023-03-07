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
        self.indexdata = indexdata
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, items):
        items += 1
        posdata = self.indexdata.attrDF[items]
        negdata = sample(self.indexdata.negtdict[items], DataloaderArg.negtnum)
        return [np.array(items), np.array(posdata), np.array(negdata)]
    



