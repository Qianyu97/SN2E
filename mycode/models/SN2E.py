
import torch
import torch.nn as nn
import torch.nn.parameter as pm
import numpy as np
from mycode.models.base import Module
from config import ModelArg
class SN2E(Module):
    def __init__(self, config:ModelArg.model): #conceptNum, dim, lambdaMax, gapMax, vmax, vmin, alpha):
        super(SN2E, self).__init__()
        self.modelName  = config.name
        self.dim        = config.dim
        self.num_defi, self.num_prim = config.num_defi, config.num_prim
        self.num_nodefi = self.num_prim + 1
        self.lambdaMax  = pm.Parameter(torch.Tensor([config.lambdaMax]), requires_grad = False)
        self.gapMax_prim = pm.Parameter(torch.Tensor([config.gapMax_prim]), requires_grad = False)
        self.gapMax_defi = pm.Parameter(torch.Tensor([config.gapMax_defi]), requires_grad = False)
        self.invmax     = 1/config.vmin
        self.invmin     = 1/config.vmax
        self.alpha      = config.alpha
        self.defaultNoneMean   = 0
        self.defaultNoneInvar  = 0.00000001
        self.conceptMeanEmbedding = torch.nn.Embedding(self.num_nodefi, config.dim, padding_idx = 0)
        self.conceptVariEmbedding = torch.nn.Embedding(self.num_nodefi, config.dim, padding_idx = 0)
        self.isCuda     = False
        
    def initEmbedding(self, varInitMode = 'const'):
        nn.init.uniform_(self.conceptMeanEmbedding.weight, -5, 5)
        nn.init.uniform_(self.conceptVariEmbedding.weight, 0.1, 10)
        self.conceptMeanEmbedding.weight.data[0] = self.defaultNoneMean
        self.conceptVariEmbedding.weight.data[0] = self.defaultNoneInvar

    def lookupEmbedding(self, index):
        '''
        index:  (index)[batchNum, indexNum]
        mean:   (torch.tensor)[batchNum, indexNum, dim]
        varInv: (torch.tensor)[batchNum, indexNum, dim]
        '''
        index = self.variable(index)
        return self.conceptMeanEmbedding(index), self.conceptVariEmbedding(index)

    def loaddefiEmbedding(self, index):
        homos = self.homoIndex[index]
        homosEmbedding  = self.lookupEmbedding(homos)
        return self.calcIntersection(homosEmbedding)

    def calcIntersection(self, embeddingA):
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varInvA    : (torch.tensor)[num0, numA, dim]

        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]
        '''
        meanA, varInvA = embeddingA
        varInvU = varInvA.sum(-2)
        meanU = (varInvA * meanA).sum(-2) / varInvU
        return meanU, varInvU

    def calcLambda(self, embeddingA, embeddingU)-> torch.Tensor:
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varInvA    : (torch.tensor)[num0, numA, dim]
        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]

        lambd      : (torch.tensor)[num0]
        '''
        meanA, varInvA = embeddingA
        meanU, varInvU = embeddingU
        lambd = - 1/2 * ( - (meanA.pow(2) * varInvA).sum(-2) + meanU.pow(2) * varInvU).sum(-1)
        return lambd

    def calcGap(self, embedding0, embeddingN) -> torch.Tensor:
        '''
        mean0      : (torch.tensor)[num0, dim] 
        varInv0    : (torch.tensor)[num0, dim]
        meanN      : (torch.tensor)[num0, numN, dim]
        varInvN    : (torch.tensor)[num0, numN, dim]

        gap        : (torch.tensor)[num0, numN]
        '''
        mean0, varInv0 = embedding0
        meanN, varInvN = embeddingN
        mean0, varInv0 = mean0.unsqueeze(-2), varInv0.unsqueeze(-2)
        gap = 1/2 * ( - (mean0 - meanN).pow(2) * (varInv0 * varInvN) / (varInv0 + varInvN) ).sum(-1)
        return gap
    
    def calcEntailProb(self, embedding1, embedding2):
        mean1, varInv1 = embedding1
        mean2, varInv2 = embedding2
        mean1, varInv1 = mean1.unsqueeze(-2), varInv1.unsqueeze(-2)
        mean2, varInv2 = mean2.unsqueeze(-3), varInv2.unsqueeze(-3)
        EntailProb = 1/2 * ( (1 + varInv2 / varInv1).log() + (mean1 - mean2).pow(2) * varInv1 * varInv2 / (varInv1 + varInv2)).sum(-1)
        return EntailProb.squeeze()

    def scorePos(self, indexA) -> torch.Tensor:
        '''
        indexA    : (index)[num0, numA]
        '''
        embeddingA = self.lookupEmbedding(indexA)
        embeddingU = self.calcIntersection(embeddingA)
        return self.calcLambda(embeddingA, embeddingU)

    def scoreNeg_prim(self, index0, indexN) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        embedding0 = self.lookupEmbedding(index0)
        embeddingN = self.lookupEmbedding(indexN)
        return self.calcGap(embedding0, embeddingN)
    
    def scoreNeg_defi(self, index0, indexN) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        embedding0 = self.loaddefiEmbedding(index0)
        embeddingN = self.lookupEmbedding(indexN)
        return self.calcGap(embedding0, embeddingN)
    
    def forward(self, data):
        '''
        index0    :(index)[num0]
        indexA    :(index)[num0, numA]
        indexN    :(index)[num0, numN]

        loss      :(torch.scale)
        '''
        [index0, indexN, indexA]  = data
        seppoint = (index0 <= self.num_nodefi).sum()
        lambd   = self.scorePos(indexA[seppoint:])
        gap_prim     = self.scoreNeg_prim(index0[:seppoint], indexN[:seppoint])
        gap_defi     = self.scoreNeg_defi(index0[seppoint:], indexN[seppoint:])
        posloss = torch.max(lambd, self.lambdaMax).sum() 
        negloss_prim = - (self.alpha / torch.max(gap_prim, self.gapMax_prim)).sum()
        negloss_defi = - (self.alpha / torch.max(gap_defi, self.gapMax_defi)).sum()
        loss = posloss + negloss_prim + negloss_defi
        return loss, lambd.sum().item(), (- gap_prim.sum().item(), - gap_defi.sum().item()), \
            lambd.max().item(), (- gap_prim.max().item(), -gap_defi.max().item())
        
    def variable(self, data):
        if self.isCuda:
            return data.cuda(self.gpunum)
        else:
            return data
    
    def tailingWorks(self):
        self.conceptVariEmbedding.weight.data.copy_(
            torch.clamp(
                input=self.conceptVariEmbedding.weight.data,
                min=self.invmin,
                max=self.invmax))

    def evaluate(self, index1, index2):
        embedding1 = self.lookupEmbedding(index1)
        embedding2 = self.lookupEmbedding(index2)
        return self.calcEntailProb(embedding1, embedding2)
    
    def sethomoIndex(self, homoDF):
        self.homoIndex = self.variable(torch.tensor(np.asarray(homoDF).T))
        
    def cudaModel(self, gpunum):
        self.cuda(gpunum)
        self.isCuda = True
        self.gpunum = gpunum

    def cpuModel(self):
        self.cpu()
        self.isCuda = False
    
    
            