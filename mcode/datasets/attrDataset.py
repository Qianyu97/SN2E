from config import Config
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from mcode.utils.utils import *

class attrDataset(Dataset):
    def __init__(self, configs:Config) -> None:
        super(Dataset, self).__init__()
        self.negsampleNum = configs.negsampleNum
        self.attrDict = loadpickle(configs.attrDictPath)
        self.homoDict = loadpickle(configs.homoDictPath)
        self.sonDict  = loadpickle(configs.sonDictPath)
        self.primConcepts   = loadpickle(configs.primConceptsPath)
        self.defiConcepts   = loadpickle(configs.defiConceptsPath)

        self.conceptIndex   = loadpickle(configs.conceptIndexPath)
        self.conceptList = self.conceptIndex.get('num2str')
        self.conceptDict = self.conceptIndex.get('str2num')

        self.primNumConcepts = translateIndex(self.primConcepts, self.conceptDict)
        self.defiNumConcepts = translateIndex(self.defiConcepts, self.conceptDict)
        self.attrDF = translateIndex(pd.DataFrame.from_dict(self.attrDict, orient='index').T, self.conceptDict)
        self.homoDF = translateIndex(pd.DataFrame.from_dict(self.homoDict, orient='index').T, self.conceptDict)
        self.negtDF = self.generateNegtSample()

        configs.model.conceptNum = len(self.conceptList)

        self.trainList = self.defiNumConcepts
        self.loaderFlag = False
    
    def generateNegtSample(self):
        primConceptSet = set(self.primConcepts)
        negtDF = pd.DataFrame.from_dict({x : primConceptSet - {x} for x in primConceptSet}, orient='index').T
        negtDF = translateIndex(negtDF, self.conceptDict)
        return negtDF
    
    def __len__(self):
        return len(self.trainList)

    def __getitem__(self, items):
        if self.loaderFlag:
            index = self.defiNumConcepts[items]
            return np.array(self.homoDF.get(index))
        else:
            index = self.primNumConcepts[items]
            return [np.array(index), np.array(self.negtDF.get(index))]
    
    def transformLoader(self):
        self.loaderFlag = not self.loaderFlag
        self.trainList =self.defiNumConcepts if self.loaderFlag else self.primNumConcepts
            
    @staticmethod
    def translationToNum(sourse:pd.DataFrame, conceptDict:dict, ):
        sourse = sourse.applymap(lambda x: conceptDict[x])
        sourse.Name = sourse.Name.map(lambda x: conceptDict[x])



