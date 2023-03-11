import numpy as np
import torch
from random import sample

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
        self.redundlen = self.targetLen - self.sourceLen + 1
        self.model = model
        self.threshold = 0.5
        '''torch.arange(
            TestArg.threshold_lower, 
            TestArg.threshold_Upper, 
            TestArg.step)'''
        self.groundtruth = self.creatGroundTruth(finaldata.indexdata.homodict) 
        a = 0
        
    
    def creatGroundTruth(self, truthdict:dict):
        groundtruth = torch.zeros([self.sourceLen, self.targetLen], dtype = bool)  # type: ignore
        for row, columns in truthdict.items():
            groundtruth[row - self.redundlen, list(columns)] = True
        #groundtruth_flat = list(groundtruth.reshape(-1,1).nonzero())[0]
        return groundtruth


    def HrankEvaluate(self,  groundtruth_flat):
        evalResult = self.model.evaluate(self.sourceData, self.targetData)
        _ , sortIndex = evalResult.reshape(-1, 1).sort(descending=True)
        _ , rank= sortIndex.sort()
        GTRank = rank[groundtruth_flat]
        return GTRank.mean().item()
    
    def calcF1score(self):
        evalResult = torch.Tensor()
        chunknum = 2
        chunklen = int(self.sourceLen / chunknum)
        chunkdata = sample(self.sourceData, chunklen) 
        evalResult = self.model.evaluate(chunkdata, self.targetData)
        judgement = evalResult < self.threshold #.unsqueeze(-1).unsqueeze(-1)
        groundtruth = self.groundtruth[[i - self.redundlen for i in chunkdata]]
        tp = (   groundtruth *   judgement).sum(-1).sum(-1)
        fp = (   groundtruth * ~ judgement).sum(-1).sum(-1)
        fn = ( ~ groundtruth *   judgement).sum(-1).sum(-1)
        precision   = (tp + 1) / ( tp + fp + 1)
        recall      = (tp + 1) / ( tp + fn + 1) 
        F1score = (2 * precision * recall / (precision + recall)).max().item()
        print('F1score: %.2f ' % F1score)
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

    def findworstlambd(self):
        attrDF = self.finaldata.indexdata.attrDF.sort_index(axis = 1)
        lambd = self.model.scorePos(np.asarray(attrDF).T)
        worstlambd, indexes = lambd.sort(descending=True)
        showname    = self.finaldata.indexconvert((indexes +  1).tolist()[:5], 'num2str')
        showvalue  = worstlambd.tolist()[:5]
        namestring  = '{}, {}, {}, {}, {} have the worst lambd, which are '.format(*showname)
        valuestring = '{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} '.format(*showvalue)
        print(namestring + valuestring)