from config import Config
from random import sample
import numpy as np
import pandas as pd
from generateSource import fineData
from torch.utils.data import Dataset
from mcode.utils.utils import *

class attrDataset(Dataset):
    def __init__(self, configs:Config) -> None:
        super(Dataset, self).__init__()
        self.strData = loadpickle(configs.fineDataPath)
        self.numData = translate(self.strData, self.strData.conceptDict)
        
        configs.conceptNum = len(self.numData.conceptList)
        configs.primConNum = len(self.numData.primConcepts)
        configs.defiConNum = len(self.numData.defiConcepts)
        self.negsampleNum = configs.negsampleNum
        self.trainList = self.numData.conceptList
        self.loaderFlag = False
    
    def __len__(self):
        return len(self.trainList)

    def __getitem__(self, items):
        if self.loaderFlag:
            index = self.numData.defiConcepts[items]
            return np.array(self.numData.attrDF.get(index))
        else:
            index = items
            return [np.array(index), np.array(sample(self.numData.negtDict.get(index), self.negsampleNum))]
    
    def transformLoader(self):
        self.loaderFlag = not self.loaderFlag
        self.trainList =self.numData.defiConcepts if self.loaderFlag else self.numData.conceptList
            
    @staticmethod
    def translationToNum(sourse:pd.DataFrame, conceptDict:dict, ):
        sourse = sourse.applymap(lambda x: conceptDict[x])
        sourse.Name = sourse.Name.map(lambda x: conceptDict[x])



