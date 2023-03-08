
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
        self.lambdaMax  = pm.Parameter(torch.Tensor([config.lambdaMax]), requires_grad = False)
        self.gapMax     = pm.Parameter(torch.Tensor([config.gapMax]), requires_grad = False)
        self.invmax     = 1/config.vmin
        self.invmin     = 1/config.vmax
        self.alpha      = config.alpha
        self.defaultNoneMean   = 0
        self.defaultNoneInvar  = 0.00000001
        self.primMeanEmbedding = pm.Parameter(torch.empty([self.num_prim, self.dim]))
        self.primVariEmbedding = pm.Parameter(torch.empty([self.num_prim, self.dim]))
        self.NoneMeanEmbedding = pm.Parameter(torch.empty([1, self.dim]), requires_grad=False)
        self.NoneVariEmbedding = pm.Parameter(torch.empty([1, self.dim]), requires_grad=False)
        self.conceptMeanEmbedding = torch.empty([self.num_prim + self.num_defi + 1, self.dim])
        self.conceptVariEmbedding = torch.empty([self.num_prim + self.num_defi + 1, self.dim])
        self.isCuda     = False
        
    def initEmbedding(self, varInitMode = 'const'):
        nn.init.constant_(self.NoneMeanEmbedding, self.defaultNoneMean)
        nn.init.constant_(self.NoneVariEmbedding, self.defaultNoneInvar)
        nn.init.uniform_(self.primMeanEmbedding, -5, 5) 
        nn.init.constant_(self.primVariEmbedding, 0.1)

    def lookupEmbedding(self, index):
        '''
        index:  (index)[batchNum, indexNum]
        mean:   (torch.tensor)[batchNum, indexNum, dim]
        varInv: (torch.tensor)[batchNum, indexNum, dim]
        '''
        mean, varInv = self.conceptMeanEmbedding[index], self.conceptVariEmbedding[index]
        return mean, varInv

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

    def calcLambda(self, embeddingA, embeddingU):
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

    def calcGap(self, embedding0, embeddingN):
        '''
        mean0      : (torch.tensor)[num0, dim] 
        varInv0    : (torch.tensor)[num0, dim]
        meanN      : (torch.tensor)[num0, numN, dim]
        varInvN    : (torch.tensor)[num0, numN, dim]

        gap        : (torch.tensor)[num0, numN]
        '''
        mean0, varInv0 = embedding0
        meanN, varInvN = embeddingN
        mean0, varInv0 = [mean0.unsqueeze(-2), varInv0.unsqueeze(-2)]
        gap = 1/2 * ( - (mean0 - meanN).pow(2) * (varInv0 * varInvN) / (varInv0 + varInvN) ).sum(-1)
        return gap
    
    def calcEntailProb(self, embedding1, embedding2):
        mean1, varInv1 = embedding1
        mean2, varInv2 = embedding2
        mean1, varInv1 = mean1.unsqueeze(-2), varInv1.unsqueeze(-2)
        mean2, varInv2 = mean2.unsqueeze(-3), varInv2.unsqueeze(-3)
        EntailProb = 1/2 * ( (varInv1 / (varInv1 + varInv2)).log() -  (mean1 - mean2).pow(2) * varInv1 * varInv2 * (varInv1 + varInv2)).sum(-1)
        return EntailProb.squeeze()

    def scorePos(self, indexA) -> torch.Tensor:
        '''
        indexA    : (index)[num0, numA]
        '''
        embeddingA = self.lookupEmbedding(indexA)
        embeddingU = self.calcIntersection(embeddingA)
        return self.calcLambda(embeddingA, embeddingU)

    def scoreNeg(self, index0, indexN) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        embedding0 = self.lookupEmbedding(index0)
        embeddingN = self.lookupEmbedding(indexN)
        return self.calcGap(embedding0, embeddingN)
    
    def forward(self, data):
        '''
        index0    :(index)[num0]
        indexA    :(index)[num0, numA]
        indexN    :(index)[num0, numN]

        loss      :(torch.scale)
        '''
        [index0, indexA, indexN]  = data
        lambd   = self.scorePos(indexA)
        gap     = self.scoreNeg(index0, indexN)
        posloss = torch.max(lambd, self.lambdaMax).sum() 
        negloss = - (self.alpha / torch.max(gap, self.gapMax)).sum()
        loss = posloss + negloss
        return loss, posloss.item(), negloss.item(), lambd.max().item(), -gap.max().item()
        
    def variable(self, data):
        if self.isCuda:
            return data.cuda(self.gpunum)
        else:
            return data
    
    def tailingWorks(self):
        self.primVariEmbedding.data.copy_(
            torch.clamp(
                input=self.primVariEmbedding.detach(),
                min=self.invmin,
                max=self.invmax))
        self.catEmbeddding()
        self.setDefiConceptEmbedding()

    def evaluate(self, index1, index2):
        embedding1 = self.lookupEmbedding(index1)
        embedding2 = self.lookupEmbedding(index2)
        return self.calcEntailProb(embedding1, embedding2)
    
    def setDefiConceptEmbedding(self):
        homoEmbedding = self.lookupEmbedding(self.homoIndex)
        [defiMean, defiVari] = self.calcIntersection(homoEmbedding)
        self.conceptMeanEmbedding = torch.cat([self.conceptMeanEmbedding, defiMean], 0)
        self.conceptVariEmbedding = torch.cat([self.conceptVariEmbedding, defiVari], 0)

    
    def catEmbeddding(self):
        self.conceptMeanEmbedding = torch.cat([self.NoneMeanEmbedding, self.primMeanEmbedding], 0)
        self.conceptVariEmbedding = torch.cat([self.NoneVariEmbedding, self.primVariEmbedding], 0)
    
    def sethomoIndex(self, homoDF):
        self.homoIndex = self.variable(torch.tensor(np.asarray(homoDF).T))
        
    def cudaModel(self, gpunum):
        self.cuda(gpunum)
        self.isCuda = True
        self.gpunum = gpunum

    def cpuModel(self):
        self.cpu()
        self.isCuda = False
    
    
            