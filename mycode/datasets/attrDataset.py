from random import sample
import numpy as np
import pandas as pd

from config import ModelArg, DataloaderArg
from refine import FineData
from torch.utils.data import Dataset
from utils import *

class attrDataset(Dataset):
    def __init__(self, indexdata:FineData) -> None:
        super(Dataset, self).__init__()
        ModelArg.__setattr__('fullnum', len(indexdata.fulllist))# type: ignore
        ModelArg.__setattr__('primnum', len(indexdata.primlist))# type: ignore
        ModelArg.__setattr__('definum', len(indexdata.defilist))# type: ignore
        self.len = len(indexdata.fulllist)
        self.indexdata = indexdata
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, items):
        posdata = self.indexdata.attrDF[items]
        negdata = sample(self.indexdata.negtdict[items], DataloaderArg.negtnum)
        return [np.array(items), np.array(posdata), np.array(negdata)]
    



