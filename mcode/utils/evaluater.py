from numpy.core.fromnumeric import sort
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import dataset
from mcode.models.base import Module
from config import Config
from mcode.datasets import attrDataset, tripleDataset


class Evaluater():
    def __init__(self, configs:Config, dataset:attrDataset.attrDataset) -> None:
        self.configs = configs
        self.dataset = dataset
        self.sourceData = dataset.strData.defiConcepts
        self.targetData = dataset.strData.conceptList
        self.sourceLen = len(self.sourceData)
        self.targetLen = len(self.targetData)

        #self.homoGT, self.homoGTflatIndex = self.creatGT(self.dataset.homoNumDict)
        
    
    def creatGT(self, truthNumDict):
        GT = np.zeros((self.sourceLen, self.targetLen))
        for row, columns in truthNumDict.items():
            GT[row, list(columns)] = True
            GTflatIndex = GT.reshape(-1,1).nonzero()[:,0]
        return GT, GTflatIndex


    def HrankEvaluate(self, model:Module):
        evalResult = model.evaluate(self.sourceData, self.targetData)
        _ , sortIndex = evalResult.reshape(-1, 1).sort(descending=True)
        _ , rank= sortIndex.sort()
        GTRank = rank[self.homoGTflatIndex]
        return GTRank.mean().item()
    
    def Hf1Evaluate(self, model:Module, rangeLower, rangeUpper, step):
        threshold = np.arange(rangeLower, rangeUpper, step)
        evalResult = model.evaluate(self.sourceData, self.targetData)
        judgement = evalResult > threshold
        tp = ( self.homoGT *  judgement).sum(1).sum(2).item()
        fp = ( self.homoGT * ~judgement).sum(1).sum(2).item()
        fn = (~self.homoGT *  judgement).sum(1).sum(2).item()
        precision   = (tp + 1) / ( tp + fp + 1)
        recall      = (tp + 1) / ( tp + fn + 1) 
        F1score = 2 * precision * recall / (precision + recall)
        print(F1score)
        return F1score

    def CREvaluate(self):
        return