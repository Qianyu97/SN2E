
import torch
import torch.nn as nn
import torch.nn.parameter as pm
import numpy as np
import os

#from utils.treeunit import Embedding
from models.base import Module
from utils.unit import Embedding
from utils.unit import EmbeddingOperator
        
class SN2E(Module):
    def __init__(self, modelArg, dataArg):
        super(SN2E, self ).__init__()
        self.modelName  = modelArg["name"]
        self.dim        = modelArg["dim"]
        self.device     = modelArg["device"]
        self.gammaMax   = modelArg["gammaMax"]
        self.gapMax     = modelArg["gapMax"]
        self.logv_max   = modelArg["logv_max"]
        self.logv_min   = modelArg["logv_min"]
        self.alpha      = modelArg["alpha"]
        self.attr_num   =  dataArg["attr_num"]
        self.node_num   =  dataArg["node_num"]
        self.depthRangeRecord = dataArg["depth_range_record"]

        self.attrMeanEmbedding  = torch.nn.Embedding(self.attr_num + 1, self.dim, padding_idx = -1, max_norm=1, device=self.device)  
        self.attrLogvEmbedding  = torch.nn.Embedding(self.attr_num + 1, self.dim, padding_idx = -1, device=self.device) 
        self.attrInvaEmbedding  = self.logv2inva(self.attrLogvEmbedding) #invaision matrix, the inverse of covariance matrix
        self.defaultMean, self.defaultLogv  = 0, 20
        self.NoneMean, self.NoneInva = torch.empty([0]), torch.empty([0])
        self.nodeMeanEmbedding, self.nodeInvaEmbedding = torch.empty([0]), torch.empty([0])
        self.nodeMeanList, self.nodeInvaList = [], []
        self.initNoneEmbedding()
        self.initNodeEmbedding()
        self.uppermapList = torch.arange(self.node_num, dtype=torch.long, device=self.device)
        self.nowupper = 1
        self.attrmapList = torch.arange(self.attr_num, dtype=torch.long, device=self.device)
        if modelArg["gapmode"] == 'gap':
            self.calcNeg = self.calcGap 
        elif modelArg["gapmode"] == 'entail':
            self.calcNeg = self.calcEntailProb
        else:
            raise Exception('gap mode should be \'gap\' or \'entail\'')
        
    def initEmbedding(self, varInitMode = 'const'):
        nn.init.uniform_(self.attrMeanEmbedding.weight, -1, 1)
        nn.init.uniform_(self.attrLogvEmbedding.weight, -5, 0)
        self.attrMeanEmbedding.weight.data[-1] = self.defaultMean
        self.attrLogvEmbedding.weight.data[-1] = self.defaultLogv
    
    def lookupEmbedding(self, index, dtype = 'attr'):
        if dtype == 'attr':
            return self.lookupAttrEmbedding(index)
        elif dtype == 'node':
            return self.lookupNodeEmbedding(index)
        else:
            raise Exception('dtype should be \'attr\' or \'node\' when looking up an embedding')

    def lookupAttrEmbedding(self, index)->Embedding:
        '''
        index:  (index)[batchNum, indexNum]
        mean:   (torch.tensor)[batchNum, indexNum, dim]
        varInv: (torch.tensor)[batchNum, indexNum, dim]
        '''
        index = self.variable(index)
        if index.shape[0] == 0:
            mean, inva = self.attrMeanEmbedding.weight[index], self.attrInvaEmbedding[index]
        else:
            index_mapped = self.attrmapList[index]
            mean, inva = self.attrMeanEmbedding(index_mapped), self.attrInvaEmbedding[index]
        return mean, inva

    def lookupNodeEmbedding(self, index)-> Embedding:
        index = self.variable(index)
        mean, inva = self.nodeMeanEmbedding[index], self.nodeInvaEmbedding[index]
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

    def calcGamma(self, e_a:Embedding, e_u:Embedding)-> torch.Tensor:
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varLogv    : (torch.tensor)[num0, numA, dim]
        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]

        gamma      : (torch.tensor)[num0]
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
        return - ( (mean_s - mean_n).pow(2).div(inva_s.reciprocal() + logv_n.reciprocal())).sum(-1)
    
    def calcEntailProb(self, e_s:Embedding, e_t:Embedding)->torch.Tensor:
        '''
        e_t被e_s包围的情况
        '''
        mean_s, inva_s = e_s
        mean_t, inva_t = e_t
        mean_s, inva_s = mean_s.unsqueeze(-2), inva_s.unsqueeze(-2)
        v_s, v_t = inva_s.reciprocal(), inva_t.reciprocal()
        return ( - inva_t.log() - (v_s + v_t).log() + 0.593147 - (mean_s - mean_t).pow(2).div(v_s + v_t)).sum(-1)
    
    def calcKL(self, e_s:Embedding, e_t:Embedding)->torch.Tensor:
        mean_s, inva_s = e_s
        mean_t, inva_t = e_t
        mean_s, inva_s = mean_s.unsqueeze(-2), inva_s.unsqueeze(-2)
        vardiv = inva_s.div(inva_t)
        return 1/2 * ( - 1 + vardiv  + (mean_s - mean_t).pow(2).div(inva_t)).sum(-1)

    def scorePos(self, indexA, indexF, train_mode=True) -> torch.Tensor:
        '''
        indexA    : (index)[num0, numA]
        '''
        e_f = self.lookupNodeEmbedding(indexF)
        e_a = self.lookupAttrEmbedding(indexA)
        e_a = EmbeddingOperator.cat([e_a, e_f])
        e_u = self.calcIntersection(e_a)
        if train_mode:
            self.stackNodeEmbedding(e_u)
        return self.calcGamma(e_a, e_u)
    
    def scoreNeg(self, index0, indexN) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        e_0 = self.lookupNodeEmbedding(index0)
        e_n = self.lookupAttrEmbedding(indexN)
        return self.calcNeg(e_0, e_n)
    
    def forward(self, data:tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        '''
        index0    :(index)[num0]
        indexA    :(index)[num0, numA]
        indexN    :(index)[num0, numN]
        loss      :(torch.scale)
        '''
        index0, indexA, indexF, indexN  = data  
        self.updateUpperMap(index0)
        gamma   = self.scorePos(indexA, indexF)
        gap     = self.scoreNeg(index0, indexN)  
        posloss = - torch.where(gamma<self.gammaMax, gamma, 0).sum() 
        negloss = torch.where(gap>self.gapMax, gap, 0).sum()
        loss    = posloss + self.alpha * negloss
        showgap = gap[gap<10000]
        return loss, gamma.sum().item(), showgap.sum().item(), gamma.min().item(), showgap.max().item()
    
        
    def batchStartWork(self):
        self.attrLogvEmbedding.weight[:-1].data.copy_(
            torch.clamp(
                input=self.attrLogvEmbedding.weight[:-1].data,
                min=self.logv_min,
                max=self.logv_max))
        self.attrInvaEmbedding = self.logv2inva(self.attrLogvEmbedding)
    
    def epochStartWork(self):
        self.initNodeEmbedding()

    def initNoneEmbedding(self) :
        NoneMean = torch.zeros([1, self.dim], requires_grad=False, device=self.device)
        NoneLogv = torch.ones([1, self.dim], requires_grad=False, device=self.device) * self.defaultLogv
        NoneInva = self.logv2inva(NoneLogv)
        self.NoneMean, self.NoneInva = [NoneMean], [NoneInva]

    def initNodeEmbedding(self):
        self.nodeMeanList = []
        self.nodeInvaList = []
        self.nodeMeanEmbedding = torch.vstack(self.nodeMeanList + self.NoneMean)
        self.nodeInvaEmbedding = torch.vstack(self.nodeInvaList + self.NoneInva)

    def stackNodeEmbedding(self, embedding:Embedding):
        mean, inva = embedding
        self.nodeMeanList.append(mean)
        self.nodeInvaList.append(inva)
        self.nodeMeanEmbedding = torch.vstack(self.nodeMeanList + self.NoneMean)
        self.nodeInvaEmbedding = torch.vstack(self.nodeInvaList + self.NoneInva)

    def variable(self, data):
        if type(data) is not torch.Tensor:
            if type(data) == int:
                data = [data]
            return torch.LongTensor(data).to(device=self.device)
        else:
            return data.to(device=self.device)
    
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
        
    def evaluate(self, inputSamples)->torch.Tensor:
        index1, index2 = inputSamples
        embedding1 = self.lookupNodeEmbedding(index1)
        embedding2 = self.lookupAttrEmbedding(index2)
        return self.calcNeg(embedding1, embedding2)
    
    def checkEntail(self, index0, indexN, typeN):
        '''
        index0: num0 
        indexN: [numN]
        '''
        e_0 = self.lookupNodeEmbedding(index0)
        e_n = self.lookupEmbedding(indexN, dtype=typeN)
        return self.calcEntailProb(e_0, e_n)
    
    def checkGap(self, index0, indexN, typeN):
        '''
        index0: num0 
        indexN: [numN]
        '''
        e_0 = self.lookupNodeEmbedding(index0)
        e_n = self.lookupEmbedding(indexN, dtype=typeN)
        return self.calcGap(e_0, e_n)

    
    def loadCheckpoint(self, path, device):
        ckpt = torch.load(path, device)
        self.load_state_dict(ckpt["model"])
        self.nodeMeanEmbedding, self.nodeInvaEmbedding = ckpt["nodeEmbedding"]
        #self.uppermapList = ckpt["uppermap"]
        self.attrInvaEmbedding  = self.logv2inva(self.attrLogvEmbedding)

    def saveCheckpoint(self, path):
        torch.save({
            "model":self.state_dict(),
            "nodeEmbedding": (self.nodeMeanEmbedding, self.nodeInvaEmbedding)
            #"uppermap":self.uppermapList
            }, path)
        
    
    
            