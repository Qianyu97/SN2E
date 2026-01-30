import numpy as np
import torch
from random import sample

from utils.unit import NodeUnit, AttrUnit, Indexer_SN2E
from utils import evalTool
from models import initModule
from models.SN2E import SN2E
from finaldata import FinalData

import pandas as pd

def matrixHomoGroungTruth(homoDict:dict[NodeUnit, set[AttrUnit]], node_num, attr_num)->torch.Tensor:
        groundtruth_matrix = torch.zeros([node_num, attr_num], dtype = bool)  
        for concept in homoDict.keys():
            attributes_index = [attribute.index for attribute in homoDict[concept]] 
            groundtruth_matrix[concept.index, attributes_index] = True
        return groundtruth_matrix

class Evaluater():
    def __init__(self, finaldata:FinalData, indexer:Indexer_SN2E, model:SN2E) -> None:
        self.finaldata = finaldata
        self.indexer = indexer
        self.model = model

    def HrankEvaluate(self,  groundtruth_flat):
        evalResult = self.model.evaluate(self.sourceData, self.targetData)
        _ , sortIndex = evalResult.reshape(-1, 1).sort(descending=True)
        _ , rank= sortIndex.sort()
        GTRank = rank[groundtruth_flat]
        return GTRank.mean().item()
    
    def evaluateF1score(self, threshold:int|list[int], mode = 'entail'):
        groundTruth = self.finaldata.homoDict
        source_num = len(self.finaldata.nodeList)
        target_num = len(self.finaldata.attrList)
        groundTruth_matrix = matrixHomoGroungTruth(groundTruth, source_num, target_num).to(self.model.device)
        sourceData_idx = list(range(source_num))
        targetData_idx = list(range(target_num))
        evalValue = self.model.evaluate((sourceData_idx, targetData_idx), mode=mode)
        prediction = evalValue > threshold
        F1score, precision, recall = evalTool.calcF1score(prediction, groundTruth_matrix)
        return F1score

    def checkgamma(self, name0:str):
        concept0 = self.finaldata.lookupConcept(name0)
        conceptA = concept0.attributes
        conceptF = concept0.father
        indexA   = [i.index for i in conceptA]
        indexF   = conceptF.index
        gamma = self.model.scorePos(indexA, indexF, ifdetach=True)
        return gamma.item()
    
    def checkgap(self, name0:str, nameN:list[str], typeN = 'node'):
        index0 = self.indexer.str2num(name0)
        indexN = self.indexer.str2num(nameN)
        gap    = self.model.checkGap(index0, indexN, typeN = typeN)
        return gap.tolist()

    def findWorstGamma(self, gamma:torch.Tensor = None):
        if gamma == None:
            pass
        attrDF = self.finaldata.attrDF
        upperDF = self.finaldata.upperDF
        attrDF_idx = self.indexer.str2num_DataFrame(attrDF).sort_index()
        upperDF_idx = self.indexer.str2num_DataFrame(upperDF).sort_index()
        gamma = self.model.scorePos(np.asarray(attrDF_idx), np.asarray(upperDF_idx), ifdetach=True) 
        worstgamma, indexes = gamma.sort(descending=False)
        showname   = self.indexer.num2str((indexes).tolist()[:5], dtype='node')
        showvalue  = worstgamma.tolist()[:5]
        namestring  = '{}, {}, {}, {}, {} have the worst gamma, which are '.format(*showname)
        valuestring = '{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} '.format(*showvalue)
        print(namestring + valuestring)

    def findWorstNegt(self, 
                      negtensor:torch.Tensor, indexN:torch.Tensor, k = 5):
        '''
        find k worst negtive loss and print them
        
        negtensor   : [num0, numN]
        indexN      : [num0, numN]
        '''
        if negtensor == None:
            print("no negtensor input, return")
            return
        indexN_new = indexN.to(negtensor.device)
        num0, numN = negtensor.shape
        worstgap, indices_flat = torch.topk(negtensor.view(-1), k)
        worst0_idx = indices_flat // numN
        worstN_indice = indices_flat % numN
        worsyN_idx = indexN_new[worst0_idx, worstN_indice]
        showname0 = self.finaldata.num2str((worst0_idx).tolist(), dtype='node')
        shownameN = self.finaldata.num2str(worsyN_idx.tolist(), dtype='attr')
        showvalue  = worstgap.tolist()
        print("the worst gap coups are:")
        for i, value in enumerate(showvalue):
            print(" ---- {source:10} - {target:15}: {value:.2f}".format(source = showname0[i], target = shownameN[i], value = value))
    
    def lookupEmbedding(self, nameList:list[str], type0='attr'):
        if type(nameList) == str:
            nameList = [nameList]
        conceptList = [self.finaldata.unitDict[type0][i] for i in nameList]
        index = [concept.index for concept in conceptList]
        return self.model.lookupEmbedding(index, type0, ifdetach=True)
    
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

if __name__ == "__main__":
    from config import PathArg, TestArg
    from config_model import SN2EArg
    ModelArg = SN2EArg
    finaldata = FinalData(PathArg["dataDirectory"])
    myIndex = Indexer_SN2E(
        nodeList=finaldata.nodeList,
        attrList=finaldata.attrList,
        load_dir=PathArg["indexDirectory"]
        )
    finaldata.indexConceptUnit(myIndex)
    model = initModule.initModel(
        ModelArg, finaldata.returnDataParams(),
        modelPath=PathArg["modelDirectory"] + ModelArg["name"] + '.ckpt', 
        usegpu=TestArg["usegpu"], 
        gpunum=TestArg["gpunum"]
        )
    evaluater = Evaluater(finaldata, myIndex, model)
    f1score = evaluater.evaluateF1score(threshold=0)
    evaluater.findWorstGamma()
    d = evaluater.checkgamma('Cat')
    b = evaluater.checkgap('Cat', ['Dog', 'Fish'], typeN='node')
    c = evaluater.checkgap('Dog', ['has_id Bird', 'has_id Bee'], typeN='attr')
    a=0


        