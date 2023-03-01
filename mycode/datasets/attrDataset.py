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
        ModelArg.__setattr__('fullnum', len(indexdata.fulllist))
        ModelArg.__setattr__('primnum', len(indexdata.primlist))
        ModelArg.__setattr__('definum', len(indexdata.defilist))
        self.trainList = indexdata.fulllist
        self.loaderFlag = False
        self.indexdata = indexdata
    
    def __len__(self):
        return len(self.trainList)
    def __getitem__(self, items):
        if self.loaderFlag:
            index = self.indexdata.defilist[items]
            return np.array(self.indexdata.attrdict.get(index))
        else:
            index = items
            return [np.array(index), np.array(sample(self.indexdata.negtdict[index], DataloaderArg.negtnum))]
    
    def transformLoader(self):
        self.loaderFlag = not self.loaderFlag
        self.trainList =self.numData.defiConcepts if self.loaderFlag else self.numData.conceptList
            
    @staticmethod
    def translationToNum(sourse:pd.DataFrame, conceptDict:dict, ):
        sourse = sourse.applymap(lambda x: conceptDict[x])
        sourse.Name = sourse.Name.map(lambda x: conceptDict[x])



