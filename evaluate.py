import numpy as np
import torch

from torch import Tensor
from random import sample

import finaldata
from utils.unit import NodeUnit, AttrUnit, BaseUnit, Indexer_SN2E
from utils import evalTool, drawerTool
from models import initModule
from models.SN2E import SN2E
from finaldata import FinalData

import pandas as pd
def geneGroundTruth(num_source, num_target, groundtruth:dict[NodeUnit, list[BaseUnit]] = None)->Tensor:
    groundtruth_matrix = torch.zeros([num_source, num_target], dtype = bool)  
    for one_source in groundtruth.keys():
        targets_index = [one_target.index for one_target in groundtruth[one_source]] 
        groundtruth_matrix[one_source.index, targets_index] = True
    return groundtruth_matrix

def geneMasking(num_source, num_target, attrDict:dict[NodeUnit, list[BaseUnit]], homoDict:dict[NodeUnit, list[BaseUnit]])->Tensor:
    masking_matrix = torch.ones([num_source, num_target], dtype = bool)  
    for one_source in homoDict.keys():
        masking = set(homoDict[one_source]) - set(attrDict[one_source])
        masking_matrix[one_source.index, [one_target.index for one_target in masking]] = False
    return masking_matrix     

def evaluateF1score(
        source_idx:list[int], target_idx:list[int], 
        groundtruth:Tensor, model:SN2E, 
        threshold:list[int], type_target='attr')->Tensor:
    with torch.no_grad():
        evalValues = model.evaluate((source_idx, target_idx), type2=type_target)
    threshold_tensor = torch.tensor(threshold, device=evalValues.device).view(-1, 1, 1)
    predictions = evalValues.unsqueeze(0) > threshold_tensor
    F1score, precision, recall = evalTool.calcF1score(predictions, groundtruth)
    best_F1score, best_threshold_index = F1score.max(dim=-1)
    best_threshold = threshold_tensor[best_threshold_index]
    return best_F1score.item(), best_threshold.item()

def evaluateAUC(source_idx:list[int], target_idx:list[int], groundtruth:Tensor, model:SN2E, type_target='node'):
    model.eval()
    with torch.no_grad():
        evalValues = model.evaluate((source_idx, target_idx), type2=type_target)
    auc_score = drawerTool.plot_roc(
        groundtruth.view(-1), 
        evalValues.view(-1), 
        title="KGE Model ROC Curve", 
        savepath="./KGE_ROC_Curve.png"
        )
    return auc_score

class Evaluater():
    def __init__(self, finaldata:FinalData, indexer:Indexer_SN2E, model:SN2E) -> None:
        self.finaldata = finaldata
        self.indexer = indexer
        self.model = model

    def HrankEvaluate(self,  groundtruth_flat):
        with torch.no_grad():
            evalResult = self.model.evaluate(self.sourceData, self.targetData)
        _ , sortIndex = evalResult.reshape(-1, 1).sort(descending=True)
        _ , rank= sortIndex.sort()
        GTRank = rank[groundtruth_flat]
        return GTRank.mean().item()
    
    def evaluateF1score(self, threshold:int|list[int], evalmode = 'intrinsic'):
        assert evalmode in ['inherited', 'intrinsic', 'node'], "evalmode should be either 'inherited' or 'intrinsic' or 'node'"
        if type(threshold) == int:
            threshold = [threshold]
        threshold_tensor = torch.tensor(threshold, device=self.model.device).view(-1, 1)
        source = self.finaldata.nodeList
        target = self.finaldata.nodeList if evalmode == 'node' else self.finaldata.attrList
        num_source, num_target = len(source), len(target)
        source_idx, target_idx = list(range(num_source)), list(range(num_target))
        with torch.no_grad():
            if evalmode == 'intrinsic':
                masking = geneMasking(num_source, num_target, self.finaldata.attrDict, self.finaldata.homoDict)
                groundtruth = geneGroundTruth(num_source, num_target, self.finaldata.attrDict)[masking]
                evalValues:torch.Tensor = self.model.evaluate((source_idx, target_idx), type2='attr')[masking]
            elif evalmode == 'inherited':
                groundtruth = geneGroundTruth(num_source, num_target, self.finaldata.homoDict).view(-1)
                evalValues:torch.Tensor = self.model.evaluate((source_idx, target_idx), type2='attr').view(-1)
            elif evalmode == 'node':
                groundtruth = geneGroundTruth(num_source, num_target, self.finaldata.anceDict).view(-1)
                evalValues:torch.Tensor = self.model.evaluate((source_idx, target_idx), type2='node').view(-1)
        predictions = evalValues.unsqueeze(0) > threshold_tensor
        F1score, precision, recall = evalTool.calcF1score(predictions, groundtruth)
        best_F1score, best_threshold_index = F1score.max(dim=-1)
        best_threshold = threshold_tensor[best_threshold_index]
        pr_auc  = drawerTool.plot_pr_auc(groundtruth, evalValues, title="KGE Model Precision-Recall Curve", draw=False)
        roc_auc = drawerTool.plot_roc(groundtruth, evalValues, title="KGE Model ROC Curve", draw=False)
        return best_F1score.item(), best_threshold.item(), pr_auc, roc_auc 
    
    def evaluateF1score_node(self, threshold:int|list[int] = 0):
        if type(threshold) == int:
            threshold = [threshold]
        source = self.finaldata.nodeList
        target = self.finaldata.nodeList
        num_source, num_target = len(source), len(target)
        source_idx, target_idx = list(range(num_source)), list(range(num_target))
        groundtruth = geneGroundTruth(num_source, num_target, self.finaldata.anceDict)
        best_F1score, best_threshold = evaluateF1score(source_idx, target_idx, groundtruth, self.model, threshold, type_target='node')
        return best_F1score.item(), best_threshold.item()

    def checkgamma(self, name0:str):
        concept0 = self.finaldata.lookupConcept(name0)
        indexA   = [i.index for i in concept0.attributes]
        indexF   = concept0.father.index
        with torch.no_grad():
            gamma = self.model.scorePos(indexA, indexF, train_mode=False)
        return gamma.item()
    
    def checkentail(self, name0:str, nameN:list[str], typeN = 'node'):
        index0 = self.indexer.str2num(name0)
        indexN = self.indexer.str2num(nameN)
        with torch.no_grad():
            gap = self.model.checkEntail(index0, indexN, typeN = typeN)
        return gap.tolist()
    
    def checkgap(self, name0:str, nameN:list[str], typeN = 'node'):
        index0 = self.indexer.str2num(name0)
        indexN = self.indexer.str2num(nameN)
        with torch.no_grad():
            gap = self.model.checkGap(index0, indexN, typeN = typeN)
        return gap.tolist()

    def findWorstGamma(self, gamma:Tensor = None):
        if gamma == None:
            pass
        attrDF = self.finaldata.attrDF
        upperDF = self.finaldata.upperDF
        attrDF_idx = self.indexer.str2num_DataFrame(attrDF).sort_index()
        upperDF_idx = self.indexer.str2num_DataFrame(upperDF).sort_index()
        with torch.no_grad():
            gamma = self.model.scorePos(np.asarray(attrDF_idx), np.asarray(upperDF_idx), train_mode=False)
        worstgamma, indexes = gamma.sort(descending=False)
        showname   = self.indexer.num2str((indexes).tolist()[:5], dtype='node')
        showvalue  = worstgamma.tolist()[:5]
        namestring  = '{}, {}, {}, {}, {} have the worst gamma, which are '.format(*showname)
        valuestring = '{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} '.format(*showvalue)
        print(namestring + valuestring)
    
    def findWorstEntail(self):
        upperDF = self.finaldata.upperDF
        concept_idx = list(range(len(self.finaldata.nodeList)))
        upperDF_idx = torch.tensor(list(range(len(self.finaldata.nodeList)))).view(1, -1)
        with torch.no_grad():
            entail = self.model.scoreEntail(np.asarray(concept_idx), upperDF_idx, type2='node')
        worstentail, indices_flat = torch.topk(entail.view(-1), k)
        showname   = self.indexer.num2str((indexes).tolist()[:5], dtype='node')
        showvalue  = worstentail.tolist()[:5]
        namestring  = '{}, {}, {}, {}, {} have the worst entail, which are '.format(*showname)
        valuestring = '{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} '.format(*showvalue)
        print(namestring + valuestring)

    #TODO
    def findWorstNegt(self, negtensor:Tensor, indexN:Tensor, k = 5):
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
    #TODO
    def lookupEmbedding(self, nameList:list[str], type0='attr'):
        if type(nameList) == str:
            nameList = [nameList]
        conceptList = [self.finaldata.unitDict[type0][i] for i in nameList]
        index = [concept.index for concept in conceptList]
        return self.model.lookupEmbedding(index, type0)
    
    #TODO
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
    f1score_intrinsic, best_threshold_intrinsic, pr_auc_intrinsic, roc_auc_intrinsic = evaluater.evaluateF1score(threshold=np.linspace(-5, 5, 21), evalmode='intrinsic')
    f1score_inherited, best_threshold_inherited, pr_auc_inherited, roc_auc_inherited = evaluater.evaluateF1score(threshold=np.linspace(-5, 5, 21), evalmode='inherited')
    f1score_node, best_threshold_node, pr_auc_node, roc_auc_node = evaluater.evaluateF1score(threshold=np.linspace(-5, 5, 21), evalmode='node')
    print(f"The f1score for attributes (intrinsic) is {f1score_intrinsic:.3f}, its threshold is {best_threshold_intrinsic}")
    print(f"The PR-AUC for attributes (intrinsic) is {pr_auc_intrinsic:.3f}, the ROC-AUC is {roc_auc_intrinsic:.3f}")
    print(f"The f1score for attributes (inherited) is {f1score_inherited:.3f}, its threshold is {best_threshold_inherited}")
    print(f"The PR-AUC for attributes (inherited) is {pr_auc_inherited:.3f}, the ROC-AUC is {roc_auc_inherited:.3f}")
    print(f"The f1score for nodes is {f1score_node:.3f}, its threshold is {best_threshold_node}")
    print(f"The PR-AUC for nodes is {pr_auc_node:.3f}, the ROC-AUC is {roc_auc_node:.3f}")
    evaluater.findWorstGamma()
    finaldata.print_tree()
    #d = evaluater.checkgamma('Cat')
    #b = evaluater.checkgap('Cat', ['Dog', 'Fish'], typeN='node')
    #c = evaluater.checkgap('Dog', ['has_id Bird', 'has_id Bee'], typeN='attr')
    a=0


        