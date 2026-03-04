
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
        '''
        Initialize hyperparameters, embedding tables, and caches.
        modelArg/dataArg: config dicts.
        '''
        super(SN2E, self ).__init__()
        self.modelName  = modelArg["name"]
        self.dim        = modelArg["dim"]
        self.device     = modelArg["device"]
        self.delta_obj  = modelArg["delta_obj"]
        self.h_obj      = modelArg["h_obj"]
        self.logv_max   = modelArg["logv_max"]
        self.logv_min   = modelArg["logv_min"]
        self.alpha      = modelArg["alpha"]
        self.attr_num   =  dataArg["attr_num"]
        self.node_num   =  dataArg["node_num"]
        self.depthRangeRecord = dataArg["depth_range_record"]

        self.attrMeanEmbedding  = torch.nn.Embedding(self.attr_num + 1, self.dim, padding_idx = -1, max_norm=1, device=self.device)  
        self.attrLogvEmbedding  = torch.nn.Embedding(self.attr_num + 1, self.dim, padding_idx = -1, device=self.device) 
        self.attrInvrEmbedding  = self.logv2invr(self.attrLogvEmbedding) #invrision matrix, the inverse of covariance matrix
        self.defaultMean, self.defaultLogv  = 0, 20
        self.paddingMean, self.paddingInvr = torch.empty([0]), torch.empty([0])
        self.nodeMeanEmbedding, self.nodeInvrEmbedding = torch.empty([0]), torch.empty([0])
        self.nodeMeanList, self.nodeInvrList = [], []
        self.initPaddingEmbedding()
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
        '''
        Initialize attribute mean and log-variance embeddings.
        '''
        nn.init.uniform_(self.attrMeanEmbedding.weight, -1, 1)
        nn.init.uniform_(self.attrLogvEmbedding.weight, -5, 0)
        #nn.init.constant_(self.attrLogvEmbedding.weight, 0)
        self.attrMeanEmbedding.weight.data[-1] = self.defaultMean
        self.attrLogvEmbedding.weight.data[-1] = self.defaultLogv
    
    def lookupEmbedding(self, index, dtype = 'attr'):
        '''
        Unified embedding lookup entry, dispatch by dtype.
        index: (index)[batch, ...]
        '''
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
        invr: (torch.tensor)[batchNum, indexNum, dim]
        '''
        index = self.variable(index)
        if index.shape[0] != 0:
            assert index.max() < self.attrMeanEmbedding.num_embeddings, f'index {index.max()} out of range for attr embedding with shape {self.attrMeanEmbedding.weight.shape}'
        if index.shape[0] == 0:
            mean, invr = self.attrMeanEmbedding.weight[index], self.attrInvrEmbedding[index]
        else:
            index_mapped = self.attrmapList[index]
            mean, invr = self.attrMeanEmbedding(index_mapped), self.attrInvrEmbedding[index]
        return mean, invr

    def lookupNodeEmbedding(self, index)-> Embedding:
        '''
        Lookup node embedding (mean, invr).
        index: (index)[batch, ...]
        '''
        index = self.variable(index)
        if index.shape[0] != 0:
            assert index.max() < self.nodeMeanEmbedding.shape[0], f'index {index.max()} out of range for node embedding with shape {self.nodeMeanEmbedding.shape}'
        mean, invr = self.nodeMeanEmbedding[index], self.nodeInvrEmbedding[index]
        return mean, invr

    def calcIntersection(self, e_a:Embedding)->Embedding:
        '''
        e_a: (mean, invr) as (torch.tensor)[num0, numA, dim]

        meanU, invrU     : (torch.tensor)[num0, dim]
        '''

        mean_a, invr_a = e_a
        invrU = invr_a.sum(-2)
        meanU = (invr_a * mean_a).sum(-2) / invrU
        return meanU, invrU

    def calcGamma(self, e_a:Embedding, e_u:Embedding)-> torch.Tensor:
        '''
        e_a: (mean, invr) as (torch.tensor)[num0, numA, dim]
        e_u: (mean, invr) as (torch.tensor)[num0, dim]
        meanA      : (torch.tensor)[num0, numA, dim] 
        logvA    : (torch.tensor)[num0, numA, dim]
        meanU      : (torch.tensor)[num0, dim]
        invrU    : (torch.tensor)[num0, dim]

        gamma      : (torch.tensor)[num0]
        '''
        mean_a, invr_a = e_a
        mean_u, invr_u = e_u
        return (- (mean_a.pow(2) * invr_a).sum(-2) + mean_u.pow(2) * invr_u).sum(-1)

    def calcGap(self, e_s:Embedding, e_n:Embedding) -> torch.Tensor:
        '''
        e_s: (mean, invr) as (torch.tensor)[num0, dim]
        e_n: (mean, invr) as (torch.tensor)[num0, numN, dim]
        mean0      : (torch.tensor)[num0, dim] 
        invr0    : (torch.tensor)[num0, dim]
        meanN      : (torch.tensor)[num0, numN, dim]
        invrN    : (torch.tensor)[num0, numN, dim]

        gap        : (torch.tensor)[num0, numN]
        '''
        mean_s, invr_s = e_s
        mean_n, logv_n = e_n
        mean_s, invr_s = mean_s.unsqueeze(-2), invr_s.unsqueeze(-2)
        return - ( (mean_s - mean_n).pow(2).div(invr_s.reciprocal() + logv_n.reciprocal())).sum(-1)
    
    def calcEntailProb(self, e_s:Embedding, e_t:Embedding)->torch.Tensor:
        '''
        e_t被e_s包围的情况
        e_s: (mean, invr) as (torch.tensor)[num0, dim]
        e_t: (mean, invr) as (torch.tensor)[num0, numT, dim]
        '''
        mean_s, invr_s = e_s
        mean_t, invr_t = e_t
        mean_s, invr_s = mean_s.unsqueeze(-2), invr_s.unsqueeze(-2)
        v_s, v_t = invr_s.reciprocal(), invr_t.reciprocal()
        return ( - invr_t.log() - (v_s + v_t).log() + 0.593147 - (mean_s - mean_t).pow(2).div(v_s + v_t)).sum(-1)
    
    def scoreEntailProb(self, indexD, indexF, type2='node')->torch.Tensor:
        '''
        indexD: (index)[numD]
        indexF: (index)[numF] or (index)[numD, numF]
        '''
        assert type2 in ['node', 'attr']
        embedding1 = self.lookupNodeEmbedding(indexD)
        embedding2 = self.lookupEmbedding(indexF, type2)
        return self.calcNeg(embedding1, embedding2)
    
    #TODO: KL算式不对
    def calcKL(self, e_s:Embedding, e_t:Embedding)->torch.Tensor:
        '''
        Compute KL divergence (formula pending correction).
        e_s: (mean, invr) as (torch.tensor)[num0, dim]
        e_t: (mean, invr) as (torch.tensor)[num0, numT, dim]
        '''
        mean_s, invr_s = e_s
        mean_t, invr_t = e_t
        mean_s, invr_s = mean_s.unsqueeze(-2), invr_s.unsqueeze(-2)
        vardiv = invr_s.div(invr_t)
        return 1/2 * ( - 1 + vardiv  + (mean_s - mean_t).pow(2).div(invr_t)).sum(-1)

    def scorePos(self, indexA, indexF, train_mode=True) -> torch.Tensor:
        '''
        indexA    : (index)[num0, numA]
        indexF    : (index)[num0, numF]
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
        index0    : (index)[num0]
        indexN    : (index)[num0, numN]
        '''
        e_0 = self.lookupNodeEmbedding(index0)
        e_n = self.lookupAttrEmbedding(indexN)
        return self.calcNeg(e_0, e_n)
    
    def forward(self, data:tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        '''
        index0    : (index)[num0]
        indexA    : (index)[num0, numA]
        indexF    : (index)[num0, numF]
        indexN    : (index)[num0, numN]
        loss      :(torch.scale)
        '''
        index0, indexA, indexF, indexN  = data  
        #self.updateUpperMap(index0)
        gamma   = self.scorePos(indexA, indexF)
        gap     = self.scoreNeg(index0, indexN)  
        posloss = torch.where(gamma<self.delta_obj, gamma, 0).sum().neg()
        negloss = torch.where(gap>self.h_obj, gap, 0).sum()
        loss    = posloss + self.alpha * negloss
        showgap = gap[gap<10000]
        return loss, gamma.sum().item(), showgap.sum().item(), gamma.min().item(), showgap.max().item()
    
        
    def batchStartWork(self):
        '''
        Clamp and refresh caches at the start of each batch.
        '''
        self.attrLogvEmbedding.weight[:-1].data.copy_(
            torch.clamp(
                input=self.attrLogvEmbedding.weight[:-1].data,
                min=self.logv_min,
                max=self.logv_max))
        self.attrInvrEmbedding = self.logv2invr(self.attrLogvEmbedding)
    
    def epochStartWork(self):
        '''
        Reset node embedding cache at the start of each epoch.
        '''
        self.initNodeEmbedding()

    def initPaddingEmbedding(self) :
        '''
        Initialize padding mean/variance embeddings.
        '''
        paddingMean = torch.zeros([1, self.dim], requires_grad=False, device=self.device)
        paddingLogv = torch.ones([1, self.dim], requires_grad=False, device=self.device) * self.defaultLogv
        paddingInvr = self.logv2invr(paddingLogv)
        self.paddingMean, self.paddingInvr = [paddingMean], [paddingInvr]

    def initNodeEmbedding(self):
        '''
        Clear and rebuild node embedding cache (with padding).
        '''
        self.nodeMeanList = []
        self.nodeInvrList = []
        self.nodeMeanEmbedding = torch.vstack(self.nodeMeanList + self.paddingMean)
        self.nodeInvrEmbedding = torch.vstack(self.nodeInvrList + self.paddingInvr)

    def stackNodeEmbedding(self, embedding:Embedding):
        '''
        Append node embedding to cache.
        '''
        mean, invr = embedding
        self.nodeMeanList.append(mean)
        self.nodeInvrList.append(invr)
        self.nodeMeanEmbedding = torch.vstack(self.nodeMeanList + self.paddingMean)
        self.nodeInvrEmbedding = torch.vstack(self.nodeInvrList + self.paddingInvr)

    def variable(self, data):
        '''
        Convert input to LongTensor on target device.
        data: (index)[...] or int/list
        '''
        if type(data) is not torch.Tensor:
            if type(data) == int:
                data = [data]
            return torch.LongTensor(data).to(device=self.device)
        else:
            return data.to(device=self.device)
    
    def logv2invr(self, logv:torch.Tensor|torch.nn.Embedding) -> torch.Tensor:
        '''
        Convert log-variance to inverse-variance.
        logv: (torch.tensor)[..., dim] or Embedding.weight
        '''
        if type(logv) == torch.nn.Embedding:
            a = logv.weight
        elif type(logv) == torch.Tensor:
            a = logv
        else:
            raise Exception('input should be tensor or embedding')
        return (-a).exp()
    
    def invr2logv(self, invr:torch.Tensor) -> torch.Tensor:
        '''
        Convert inverse-variance to log-variance.
        invr: (torch.tensor)[..., dim]
        '''
        return -torch.log(invr)

    def invPermutation(self, list):
        '''
        Invert mapping: map values in list back to indices.
        '''
        for idx, val in enumerate(list):
            self.uppermapList[val] = idx
        
    def evaluate(self, inputSamples, type2='node')->torch.Tensor:
        '''
        Evaluation entry: compute entailment score.
        inputSamples: (index)[numD], (index)[numF]
        '''
        assert type2 in ['node', 'attr']
        index1, index2 = inputSamples
        return self.scoreEntailProb(index1, index2, type2)

    def scoreGap(self, index0, indexN, typeN):
        '''
        index0: (index)[num0]
        indexN: (index)[numN] or (index)[num0, numN]
        '''
        e_0 = self.lookupNodeEmbedding(index0)
        e_n = self.lookupEmbedding(indexN, dtype=typeN)
        return self.calcGap(e_0, e_n)

    def loadCheckpoint(self, path, device):
        '''
        Load checkpoint and restore model and caches.
        path: (str)
        device: (str|torch.device)
        '''
        ckpt = torch.load(path, device)
        self.load_state_dict(ckpt["model"])
        self.nodeMeanEmbedding, self.nodeInvrEmbedding = ckpt["nodeEmbedding"]
        #self.uppermapList = ckpt["uppermap"]
        self.attrInvrEmbedding  = self.logv2invr(self.attrLogvEmbedding)

    def saveCheckpoint(self, path):
        '''
        Save checkpoint (model params and node embedding cache).
        path: (str)
        '''
        torch.save({
            "model":self.state_dict(),
            "nodeEmbedding": (self.nodeMeanEmbedding, self.nodeInvrEmbedding)
            #"uppermap":self.uppermapList
            }, path)
        
    
    
            