
import torch
import torch.nn as nn
import torch.nn.parameter as pm
import numpy as np
import os

#from utils.treeunit import Embedding
from models.base import Module
from utils.embedding import Embedding
from utils.embedding import EmbeddingOperator
from config import TrainArg
from config_model import SN2E_Arg


        
class SN2E(Module):
    def __init__(self, ModelArg:SN2E_Arg, device):
        super(SN2E, self ).__init__()
        self.modelName  = ModelArg.name
        self.dim        = ModelArg.dim
        self.device     = device
        self.attr_num   = ModelArg.attr_num
        self.defi_num   = ModelArg.defi_num
        self.depth_range_record = ModelArg.depth_range_record
        self.lambdaMax  = ModelArg.lambdaMax
        self.gapMax     = ModelArg.gapMax
        self.logv_max   = ModelArg.logv_max
        self.logv_min   = ModelArg.logv_min
        self.alpha      = ModelArg.alpha
        self.attrMeanEmbedding = torch.nn.Embedding(self.attr_num, self.dim, padding_idx = 0, max_norm=1, device=device)  
        self.attrLogvEmbedding  = torch.nn.Embedding(self.attr_num, self.dim, padding_idx = 0, device=device) 
        self.attrInvaEmbedding  = self.logv2inva(self.attrLogvEmbedding) #invaision matrix, the inverse of covariance matrix
        self.defaultMean, self.defaultLogv  = 0, -20
        self.NoneMean, self.NoneInva = torch.empty([0]), torch.empty([0])
        self.defiMeanEmbedding, self.defiInvaEmbedding = torch.empty([0]), torch.empty([0])
        self.defiMeanList, self.defiInvaList = [], []
        self.initNoneEmbedding()
        self.initDefiEmbedding()
        self.initUpperMap()
        if ModelArg.gapmode == 'gap':
            self.calcNeg = self.calcGap 
        elif ModelArg.gapmode == 'entail':
            self.calcNeg = self.calcEntailProb
        else:
            raise Exception('gap mode should be \'gap\' or \'entail\'')
        
    def initEmbedding(self, varInitMode = 'const'):
        nn.init.uniform_(self.attrMeanEmbedding.weight, -1, 1)
        nn.init.uniform_(self.attrLogvEmbedding.weight, -5, 0)
        self.attrMeanEmbedding.weight.data[0] = self.defaultMean
        self.attrLogvEmbedding.weight.data[0] = self.defaultLogv
    
    def lookupEmbedding(self, index, type0 = 'attr', ifdetach=False):
        if type0 == 'attr':
            return self.lookupAttrEmbedding(index, ifdetach)
        elif type0 == 'defi':
            return self.lookupDefiEmbedding(index, ifdetach)
        else:
            raise Exception('type0 should be attr or defi when looking up embedding')

    def lookupAttrEmbedding(self, index, ifdetach=False)->Embedding:
        '''
        index:  (index)[batchNum, indexNum]
        mean:   (torch.tensor)[batchNum, indexNum, dim]
        varInv: (torch.tensor)[batchNum, indexNum, dim]
        '''
        index = self.variable(index)
        mean, inva = self.attrMeanEmbedding(index), self.attrInvaEmbedding[index]
        if ifdetach:
            return mean.detach(), inva.detach()
        else:
            return mean, inva

    def lookupDefiEmbedding(self, index, ifdetach=False)-> Embedding:
        index = self.variable(index)
        index_mapped = self.uppermapList[index]
        mean, inva = self.defiMeanEmbedding[index_mapped], self.defiInvaEmbedding[index_mapped]
        if ifdetach:
            return mean.detach(), inva.detach()
        else:
            return mean, inva

    def calcIntersection(self, e_a:Embedding)->Embedding:
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varLogv    : (torch.tensor)[num0, numA, dim]

        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]
        '''
        mean_a, inva_a = e_a
        varInvU = inva_a.sum(-2)
        meanU = (inva_a * mean_a).sum(-2) / varInvU
        return meanU, varInvU

    def calcLambda(self, e_a:Embedding, e_u:Embedding)-> torch.Tensor:
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varLogv    : (torch.tensor)[num0, numA, dim]
        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]

        lambd      : (torch.tensor)[num0]
        '''
        mean_a, inva_a = e_a
        mean_u, inva_u = e_u
        return (- (mean_a.pow(2) * inva_a).sum(-2) + mean_u.pow(2) * inva_u).sum(-1)

    def calcGap(self, e_s:Embedding, e_n:Embedding) -> torch.Tensor:
        '''
        mean0      : (torch.tensor)[num0, dim] 
        varInv0    : (torch.tensor)[num0, dim]
        meanN      : (torch.tensor)[num0, numN, dim]
        varInvN    : (torch.tensor)[num0, numN, dim]

        gap        : (torch.tensor)[num0, numN]
        '''
        mean_s, inva_s = e_s
        mean_n, logv_n = e_n
        mean_s, inva_s = mean_s.unsqueeze(-2), inva_s.unsqueeze(-2)
        return ( (mean_s - mean_n).pow(2).div(inva_s.reciprocal() + logv_n.reciprocal())).sum(-1)
    
    def calcEntailProb(self, e_s:Embedding, e_t:Embedding)->torch.Tensor:
        '''
        e_t被e_s包围的情况
        '''
        mean_s, inva_s = e_s
        mean_t, inva_t = e_t
        mean_s, inva_s = mean_s.unsqueeze(-2), inva_s.unsqueeze(-2)
        return (inva_s.log() - (inva_s + inva_t).log() - (mean_s - mean_t).pow(2).div(inva_s.reciprocal()+inva_t.reciprocal())).sum(-1)
    
    def calcKL(self, e_s:Embedding, e_t:Embedding)->torch.Tensor:
        mean_s, inva_s = e_s
        mean_t, inva_t = e_t
        mean_s, inva_s = mean_s.unsqueeze(-2), inva_s.unsqueeze(-2)
        vardiv = inva_s.div(inva_t)
        return 1/2 * ( - 1 + vardiv  + (mean_s - mean_t).pow(2).div(inva_t)).sum(-1)

    def scorePos(self, indexA, indexF, ifdetach=False) -> torch.Tensor:
        '''
        indexA    : (index)[num0, numA]
        '''
        e_f = self.lookupDefiEmbedding(indexF, ifdetach)
        e_a = self.lookupAttrEmbedding(indexA, ifdetach)
        e_a = EmbeddingOperator.cat([e_a, e_f])
        e_u = self.calcIntersection(e_a)
        if not ifdetach:
            self.stackDefiEmbedding(e_u)
        return self.calcLambda(e_a, e_u)
    
    def scoreNeg(self, index0, indexN, ifdetach=False) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        e_0 = self.lookupDefiEmbedding(index0, ifdetach)
        e_n = self.lookupAttrEmbedding(indexN, ifdetach)
        return self.calcNeg(e_0, e_n)
    
    def forward(self, data:tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        '''
        index0    :(index)[num0]
        indexA    :(index)[num0, numA]
        indexN    :(index)[num0, numN]
        loss      :(torch.scale)
        '''
        index0, indexN, indexA, indexF  = data  
        self.updateUpperMap(index0)
        lambd   = self.scorePos(indexA, indexF)
        gap     = self.scoreNeg(index0, indexN)  
        posloss = - torch.where(lambd<self.lambdaMax, lambd, 0).sum() 
        negloss = torch.where(gap>self.gapMax, gap, 0).sum()
        loss    = posloss + self.alpha * negloss
        showgap = gap[gap<10000]
        return loss, lambd.sum().item(), showgap.sum().item(), lambd.min().item(), showgap.max().item()
        
    def batchEndWork(self):
        self.attrLogvEmbedding.weight[1:].data.copy_(
            torch.clamp(
                input=self.attrLogvEmbedding.weight[1:].data,
                min=self.logv_min,
                max=self.logv_max))
        self.attrInvaEmbedding = self.logv2inva(self.attrLogvEmbedding)
    
    def epochStartWork(self):
        self.initDefiEmbedding()
        self.initUpperMap()

    def generateWholeEmbedding(self):
        homoEmbedding = self.lookupEmbedding(self.homoIndex)
        defiEmebdding = self.calcIntersection(homoEmbedding)
        self.conceptmeanEmbedding_whole = torch.cat([self.conceptMeanEmbedding.weight, defiEmebdding.m])
        self.conceptvariEmbedding_whole = torch.cat([self.conceptVariEmbedding.weight, defiEmebdding.logv])
    
    def initNoneEmbedding(self) :
        NoneMean = torch.zeros([1, self.dim], requires_grad=False, device=self.device)
        NoneLogv = torch.ones([1, self.dim], requires_grad=False, device=self.device) * self.defaultLogv
        NoneInva = self.logv2inva(NoneLogv)
        self.NoneMean, self.NoneInva = NoneMean, NoneInva

    def initDefiEmbedding(self):
        self.defiMeanList = []
        self.defiInvaList = []
        self.stackDefiEmbedding((self.NoneMean, self.NoneInva))
    
    def stackDefiEmbedding(self, embedding:Embedding):
        mean, inva = embedding
        self.defiMeanList.append(mean)
        self.defiInvaList.append(inva)
        self.defiMeanEmbedding = torch.vstack(self.defiMeanList)
        self.defiInvaEmbedding = torch.vstack(self.defiInvaList)

    def initUpperMap(self):
        self.uppermapList = torch.zeros([self.defi_num], dtype=torch.long, device=self.device)
        self.nowupper = 1

    def updateUpperMap(self, newIndex:torch.Tensor):
        self.uppermapList[newIndex] = torch.arange(len(newIndex), device=self.device) + self.nowupper
        self.nowupper += len(newIndex)

    def variable(self, data):
        if not type(data) == torch.Tensor:
            if type(data) == int:
                data = torch.LongTensor([data])
            else:
                data = torch.LongTensor(data)
        return data.to(self.device)
    
    def init(self):
        self.initEmbedding()
        self.batchEndWork()
        
    
    def logv2inva(self, logv:torch.Tensor|torch.nn.Embedding) -> torch.Tensor:
        if type(logv) == torch.nn.Embedding:
            a = logv.weight
        elif type(logv) == torch.Tensor:
            a = logv
        else:
            raise Exception('input should be tensor or embedding')
        return (-a).exp()
    
    def inva2logv(self, inva:torch.Tensor) -> torch.Tensor:
        return -torch.log(inva)

    def invPermutation(self, list):
        for idx, val in enumerate(list):
            self.uppermapList[val] = idx
        
    def evaluate(self, index1, index2, mode = 'gap', type2 = 'attr'):
        embedding1 = self.lookupDefiEmbedding(index1)
        if type2 == 'attr':
            embedding2 = self.lookupAttrEmbedding(index2)
        elif type2 == 'defi':
            embedding2 = self.lookupDefiEmbedding(index2)
        else:
            raise Exception("type_2 should be attr or defi")
        if mode == 'KL':
            eval = self.calcKL
        elif mode == 'gap':
            eval = self.calcGap
        elif mode == 'entail':
            eval = self.calcEntailProb
        else:
            raise Exception('eval mode should be \'kl\', \'gap\' or \'entail\'')
        return eval(embedding1, embedding2)
    
    def loadCheckpoint(self, path, device):
        ckpt = torch.load(path, device)
        self.load_state_dict(ckpt["model"])
        self.defiMeanEmbedding, self.defiInvaEmbedding = ckpt["defiEmbedding"]

    def saveCheckpoint(self, path):
        torch.save({
            "model":self.state_dict(),
            "defiEmbedding": (self.defiMeanEmbedding, self.defiInvaEmbedding)
            }, path)
    
    
            