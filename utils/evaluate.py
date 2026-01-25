import numpy as np
import torch
from random import sample

from utils.treeunit import NodeUnit, AttributeUnit
from models.SN2E import SN2E
from finaldata import FinalData


class Evaluater():
    def __init__(self, finalData:FinalData, model:SN2E) -> None:
        self.finalData = finalData
        self.model = model
        self.sourceData = finalData.defiList
        self.targetData = finalData.attrList
        self.sourceData_idx = [i.index for i in self.sourceData]
        self.targetData_idx = [i.index for i in self.targetData]
        self.sourceLen = len(self.sourceData)
        self.targetLen = len(self.targetData)
        self.groundTruth = finalData.homoDict
        self.groundTruth_matrx = self.matrixGroungTruth()
        
        self.threshold = 1
        '''torch.arange(
            TestArg.threshold_lower, 
            TestArg.threshold_Upper, 
            TestArg.step)'''
        a = 0
        
    
    def matrixGroungTruth(self)->torch.Tensor:
        groundtruth_matrix = torch.zeros([self.sourceLen, self.targetLen], dtype = bool, device=self.model.device)  
        for concept in self.sourceData:
            oneGroundTruth_idx = [attribute.index - 1 for attribute in self.groundTruth[concept]] 
            groundtruth_matrix[concept.index - 1, oneGroundTruth_idx] = True
        #groundtruth_flat = list(groundtruth.reshape(-1,1).nonzero())[0]
        return groundtruth_matrix


    def HrankEvaluate(self,  groundtruth_flat):
        evalResult = self.model.evaluate(self.sourceData, self.targetData)
        _ , sortIndex = evalResult.reshape(-1, 1).sort(descending=True)
        _ , rank= sortIndex.sort()
        GTRank = rank[groundtruth_flat]
        return GTRank.mean().item()
    
    def calcF1score(self, chunknum = 1):
        chunklen = int(self.sourceLen / chunknum)
        chunkedData_idx = sample(self.sourceData_idx, chunklen) 
        evalResult = self.model.evaluate(chunkedData_idx, self.targetData_idx, mode='gap', type2='attr')
        judgement = evalResult < self.threshold #.unsqueeze(-1).unsqueeze(-1)
        groundtruth = self.groundTruth_matrx[chunkedData_idx]
        tp = (   groundtruth *   judgement).sum(-1).sum(-1)
        fp = (   groundtruth * ~ judgement).sum(-1).sum(-1)
        fn = ( ~ groundtruth *   judgement).sum(-1).sum(-1)
        precision   = (tp + 1) / ( tp + fp + 1)
        recall      = (tp + 1) / ( tp + fn + 1) 
        F1score = (2 * precision * recall / (precision + recall)).max().item()
        print('F1score: %.4f ' % F1score)
        '''print('wrong fn: ')
        for i in ( ~ groundtruth *   judgement).nonzero().tolist():
            sourcename = self.finaldata.indexconvert(chunkdata[i[0]], 'num2str')
            targetname = self.finaldata.indexconvert(i[1] + 2, 'num2str')
            print('%s - %s'%(sourcename, targetname))'''
        
        return F1score

    def checklambd(self, name0:str):
        concept0 = self.finalData.defiUnitDict[name0]
        conceptA = concept0.attributes
        conceptF = concept0.father
        indexA   = [i.index for i in conceptA]
        indexF   = conceptF.index
        lambd = self.model.scorePos(indexA, indexF, ifdetach=True)
        return lambd
    
    def checkgap(self, name0:str, nameN:list[str], typeN = 'attr'):
        concept0 = self.finalData.defiUnitDict[name0]
        index0 = concept0.index
        if typeN == 'attr':
            conceptN = [self.finalData.attrUnitDict[i] for i in nameN]
            indexN = [i.index for i in conceptN]
        elif typeN == 'defi':
            conceptN = [self.finalData.defiUnitDict[i] for i in nameN]
            indexN = [i.index for i in conceptN]
        else:
            raise Exception('Arg typeN should be attr or defi')
        gap     = self.model.scoreNeg(index0, indexN, ifdetach=True)
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
    
    def lookupEmbedding(self, nameList:list[str], type0='attr'):
        if type(nameList) == str:
            nameList = [nameList]
        conceptList = [self.finalData.unitDict[type0][i] for i in nameList]
        index = [concept.index for concept in conceptList]
        return self.model.lookupEmbedding(index, type0, ifdetach=True)
    
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
        