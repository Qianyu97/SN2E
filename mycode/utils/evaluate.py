import numpy as np
import torch
from random import sample

from mycode.models.SN2E import SN2E
from finaldata import FinalData
from config import TestArg, TrainArg


class Evaluater():
    def __init__(self, finaldata:FinalData, model:SN2E) -> None:
        self.finaldata = finaldata
        self.sourceData = finaldata.indexdata.defilist
        self.targetData = finaldata.indexdata.fulllist
        self.sourceLen = len(self.sourceData)
        self.targetLen = len(self.targetData)
        self.redundlen = self.targetLen - self.sourceLen + 2
        self.model = model
        self.threshold = 1
        '''torch.arange(
            TestArg.threshold_lower, 
            TestArg.threshold_Upper, 
            TestArg.step)'''
        self.groundtruth = self.creatGroundTruth() 
        a = 0
        
    
    def creatGroundTruth(self):
        homodict_kt = self.finaldata.indexconvert(self.finaldata.finedata.creat_homodict_kt())
        groundtruth = torch.zeros([self.sourceLen, self.targetLen], dtype = bool)  # type: ignore
        for row in self.sourceData:
            columns = homodict_kt[row]
            groundtruth[row - self.redundlen, [i - 2 for i in columns]] = True
        #groundtruth_flat = list(groundtruth.reshape(-1,1).nonzero())[0]
        return groundtruth


    def HrankEvaluate(self,  groundtruth_flat):
        evalResult = self.model.evaluate(self.sourceData, self.targetData)
        _ , sortIndex = evalResult.reshape(-1, 1).sort(descending=True)
        _ , rank= sortIndex.sort()
        GTRank = rank[groundtruth_flat]
        return GTRank.mean().item()
    
    def calcF1score(self, chunknum = 3):
        chunklen = int(self.sourceLen / chunknum)
        chunkdata = sample(self.sourceData, chunklen) 
        evalResult = self.model.evaluate(chunkdata, sorted(self.targetData), 'KL')
        judgement = evalResult < self.threshold #.unsqueeze(-1).unsqueeze(-1)
        groundtruth = self.groundtruth[[i - self.redundlen for i in chunkdata]]
        if TrainArg.usegpu:
            groundtruth = groundtruth.cuda(TrainArg.gpunum)
        tp = (   groundtruth *   judgement).sum(-1).sum(-1)
        fp = (   groundtruth * ~ judgement).sum(-1).sum(-1)
        fn = ( ~ groundtruth *   judgement).sum(-1).sum(-1)
        precision   = (tp + 1) / ( tp + fp + 1)
        recall      = (tp + 1) / ( tp + fn + 1) 
        F1score = (2 * precision * recall / (precision + recall)).max().item()
        print('F1score: %.4f ' % F1score)
        print('wrong fn: ')
        for i in ( ~ groundtruth *   judgement)[:100].nonzero().tolist():
            sourcename = self.finaldata.indexconvert(chunkdata[i[0]], 'num2str')
            targetname = self.finaldata.indexconvert(i[1] + 2, 'num2str')
            print('%s - %s'%(sourcename, targetname))
        
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
        attrDF = self.finaldata.indexdata.homoDF.sort_index()
        pareDF = self.finaldata.indexdata.pareDF.sort_index()
        lambd = self.model.scorePos(np.asarray(attrDF)) # type: ignore
        worstlambd, indexes = lambd.sort(descending=True)
        showname    = self.finaldata.indexconvert((indexes + self.redundlen).tolist()[:5], 'num2str')
        showvalue  = worstlambd.tolist()[:5]
        namestring  = '{}, {}, {}, {}, {} have the worst lambd, which are '.format(*showname)
        valuestring = '{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} '.format(*showvalue)
        print(namestring + valuestring)
    
    def findworstgap(self, mode = 'gap'):
        import pandas as pd
        shownum = 5
        negtArray = np.asarray(self.finaldata.indexconvert(pd.DataFrame.from_dict(self.finaldata.finedata.negtdict, orient='index').applymap(lambda x: 'negtpad' if x is None else x)).sort_index())
        gap = self.model.evaluate(np.asarray(sorted(self.sourceData))[:1000], negtArray[:1000], mode)
        gap_flat = gap.flatten()
        row, colomn = gap.shape
        worstgap, index = gap_flat.sort()
        showname_s = self.finaldata.indexconvert(((index / colomn).int() + self.redundlen).tolist()[:shownum], 'num2str')
        showname_t = self.finaldata.indexconvert(list(negtArray[(index / colomn).int().tolist()[:shownum], (index % colomn).int().tolist()[:shownum]]), 'num2str')
        showvalue  = worstgap.tolist()[:shownum]
        print("the worst gap coups are:")
        for i, value in enumerate(showvalue):
            print(" ---- {source:15} - {target:5}: gap = {value:.2f}".format(source = showname_s[i], target = showname_t[i].name, value = value))
        a = 0
    
    def findworstgap_homo(self, mode = 'gap'):
        import pandas as pd
        shownum = 5
        homoDF_kt = self.finaldata.indexconvert(self.finaldata.finedata.creat_DataFrame(self.finaldata.finedata.creat_homodict_kt()))
        homoArray = np.asarray(homoDF_kt.sort_index())
        gap = self.model.evaluate(np.asarray(sorted(self.sourceData)), homoArray, mode)
        gap_flat = gap.flatten()
        row, colomn = gap.shape
        worstgap, index = gap_flat.sort(descending=True)
        showname_s = self.finaldata.indexconvert(((index / colomn).int() + self.redundlen).tolist()[:shownum], 'num2str')
        showname_t = self.finaldata.indexconvert(list(homoArray[(index / colomn).int().tolist()[:shownum], (index % colomn).int().tolist()[:shownum]]), 'num2str')
        showvalue  = worstgap.tolist()[:shownum]
        print("the worst homo coups are:")
        for i, value in enumerate(showvalue):
            print(" ---- {source:15} - {target:5}: gap = {value:.2f}".format(source = showname_s[i], target = showname_t[i].name, value = value))
        a = 0
        