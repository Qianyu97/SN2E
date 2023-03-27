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
        self.padAttr    = [0]*self.attrDF.shape[0]
        self.padPare    = [0]
        self.padNegt    = [1]    
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, items):
        items += (ModelArg.model.num_nodefi)
        attrdata = self.attrDF.get(items, self.padAttr)
        #paredata = self.indexdata.paredict.get(items, self.padPare)
        negdata = self.indexdata.negtdict[items]
        len_negdata = len(negdata)
        if len_negdata > DataloaderArg.negtsamplenum:
            negdata = sample(negdata, DataloaderArg.negtsamplenum) 
        else:
            negdata.extend(self.padNegt*(DataloaderArg.negtsamplenum - len_negdata))
        return [np.asarray(items), np.asarray(negdata), np.asarray(attrdata)] # type: ignore  
        



