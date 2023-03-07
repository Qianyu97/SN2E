import numpy as np
import torch

from mycode.models.SN2E import SN2E
from finaldata import FinalData
from config import TestArg


class Evaluater():
    def __init__(self, finaldata:FinalData, model:SN2E) -> None:
        self.finaldata = finaldata
        self.sourceData = finaldata.indexdata.defilist
        self.targetData = finaldata.indexdata.fulllist
        self.sourceLen = len(self.sourceData)
        self.targetLen = len(self.targetData)
        self.model = model
        threshold = np.arange(
            TestArg.threshold_lower, 
            TestArg.threshold_Upper, 
            TestArg.step)
        groundtruth = self.creatGroundTruth(finaldata.indexdata.homodict) 
        a = 0
        
    
    def creatGroundTruth(self, truthdict:dict):
        GT = np.zeros((self.sourceLen, self.targetLen), dtype = bool)
        for row, columns in truthdict.items():
            GT[row - self.targetLen + self.sourceLen - 1, list(columns)] = True
        GTflatIndex = list(GT.reshape(-1,1).nonzero())[0]
        return GT


    def HrankEvaluate(self,  groundtruth_flat):
        evalResult = self.model.evaluate(self.sourceData, self.targetData)
        _ , sortIndex = evalResult.reshape(-1, 1).sort(descending=True)
        _ , rank= sortIndex.sort()
        GTRank = rank[groundtruth_flat]
        return GTRank.mean().item()
    
    def Hf1Evaluate(self, groundtruth, threshold):
        evalResult = self.model.evaluate(self.sourceData, self.targetData)
        judgement = evalResult > threshold
        tp = (   groundtruth *   judgement).sum(1).sum(2).item()
        fp = (   groundtruth * ~ judgement).sum(1).sum(2).item()
        fn = ( ~ groundtruth *   judgement).sum(1).sum(2).item()
        precision   = (tp + 1) / ( tp + fp + 1)
        recall      = (tp + 1) / ( tp + fn + 1) 
        F1score = 2 * precision * recall / (precision + recall)
        print(F1score)
        return F1score

    def checklambd(self, concept):
        attributes  = self.finaldata.finedata.attrdict[concept]
        attrIndex   = list(self.finaldata.indexconvert(attributes))
        lambd       = self.model.scorePos(attrIndex)
        return lambd
    
    def checkgap(self, concept0, conceptN):
        index0  = self.model.variable(self.finaldata.indexconvert(concept0))
        indexN  = self.model.variable(self.finaldata.indexconvert(conceptN))
        gap     = self.model.scoreNeg(index0, indexN)
        return gap