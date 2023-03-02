import torch
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
        self.NoneIndex  = 0
        self.isCuda     = False
        self.primMeanEmbedding = pm.Parameter(torch.empty([self.num_prim, self.dim]))
        self.primVariEmbedding = pm.Parameter(torch.empty([self.num_prim, self.dim]))
        self.defiMeanEmbedding = pm.Parameter(torch.empty([self.num_defi + 1, self.dim]), requires_grad = False)
        self.defiVariEmbedding = pm.Parameter(torch.empty([self.num_defi + 1, self.dim]), requires_grad = False)
        self.catTogether()
        

    def lookupEmbedding(self, index, detach = False):
        '''
        index:  (index)[BatchNum, indexNum]
        mean:   (torch.tensor)[BatchNum, indexNum, dim]
        varInv: (torch.tensor)[BatchNum, indexNum, dim]
        '''
        index = self.variable(index)
        mean, varInv = self.conceptMeanEmbedding[index], self.conceptVariEmbedding[index]
        #if detach:
        #    mean, varInv = mean.detach(), varInv.detach()
        return [mean, varInv]

    def calcIntersection(self, embeddingA):
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varInvA    : (torch.tensor)[num0, numA, dim]

        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]
        '''
        [meanA, varInvA] = embeddingA
        varInvU = varInvA.sum(-2)
        meanU = (varInvA * meanA).sum(-2) / varInvU
        return [meanU, varInvU]

    def calcLambda(self, embeddingA, embeddingU):
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varInvA    : (torch.tensor)[num0, numA, dim]
        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]

        lambd      : (torch.tensor)[num0]
        '''
        [meanA, varInvA] = embeddingA
        [meanU, varInvU] = embeddingU
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
        [mean0, varInv0] = embedding0
        [meanN, varInvN] = embeddingN
        [mean0, varInv0] = [mean0.unsqueeze(1), varInv0.unsqueeze(1)]
        gap = 1/2 * ( - (mean0 - meanN).pow(2) * (varInv0 * varInvN) / (varInv0 + varInvN) ).sum(-1)
        return gap
    
    def calcSbProb(self, embedding1, embedding2):
        [mean1, varInv1] = embedding1
        [mean2, varInv2] = embedding2
        sbProb = 1/2 * ( (varInv1 / (varInv1 + varInv2)).log() -  (mean1 - mean2).pow(2) * varInv1 * varInv2 * (varInv1 + varInv2)).sum(-1)
        return sbProb

    def scorePos(self, indexA):
        '''
        indexA    : (index)[num0, numA]
        '''
        embeddingA = self.lookupEmbedding(indexA)
        embeddingU = self.calcIntersection(embeddingA)
        return self.calcLambda(embeddingA, embeddingU)

    def scoreNeg(self, index0, indexN):
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        embedding0 = self.lookupEmbedding(index0)
        embeddingN = self.lookupEmbedding(indexN)
        return self.calcGap(embedding0, embeddingN)
    
    def forward(self, data, trainMode):
        '''
        index0    :(index)[num0]
        indexA    :(index)[num0, numA]
        indexN    :(index)[num0, numN]

        loss      :(torch.scale)
        '''
        loss = torch.Tensor(0)
        if trainMode == 'defimode':
            [index0, indexA, indexN]  = data
            lambd   = self.scorePos(indexA)
            gap     = self.scoreNeg(index0, indexN)
            posloss = torch.max(lambd, self.lambdaMax).sum() 
            negloss = - (self.alpha / torch.max(gap, self.gapMax)).sum()
            loss = posloss + negloss
        elif trainMode == 'primmode':
            [index0, indexN] = data
            gap     = self.scoreNeg(index0, indexN)
            negloss    = - (1 / torch.max(gap, self.gapMax)).sum() * self.alpha
            loss = negloss
        return loss
    
    def initEmbedding(self, varInitMode = 'const'):
        nn.init.uniform_(self.primMeanEmbedding, -5, 5) 
        nn.init.constant_(self.defiMeanEmbedding, 0) 
        nn.init.constant_(self.primVariEmbedding, 0.1)
        nn.init.constant_(self.defiVariEmbedding, 0.1)
        self.tailingWorks()
        self.catTogether()
        
    
    def variable(self, data):
        if self.isCuda:
            return data.cuda()
        else:
            return data
    
    def tailingWorks(self):
        def resetNoneEmbedding():
            self.primMeanEmbedding[self.NoneIndex][:] = 0
            self.primVariEmbedding[self.NoneIndex][:] = 0
        def limitVarRange():
            self.primVariEmbedding.data.copy_(torch.clamp(input=self.primVariEmbedding.detach(),
                                                       min=self.invmin,
                                                       max=self.invmax))
        limitVarRange()
        #resetNoneEmbedding()
    
    def tempProcess(self, homoDF):
        self.catTogether()
        self.setDefiConceptEmbedding(homoDF)

    def evaluate(self, index1, index2):
        embedding1 = self.lookupEmbedding(index1)
        embedding2 = self.lookupEmbedding(index2)
        return self.calcSbProb(embedding1, embedding2)
    
    def setDefiConceptEmbedding(self, homoDF):
        homoTensor = self.variable(torch.tensor(np.asarray(homoDF).T))
        homoEmbedding = self.lookupEmbedding(homoTensor)
        [defiMean, defiVari] = self.calcIntersection(homoEmbedding)
        self.conceptMeanEmbedding[self.num_prim : -1] = defiMean
        self.conceptVariEmbedding[self.num_prim : -1] = defiVari
    
    def catTogether(self):
        self.conceptMeanEmbedding = torch.cat([self.primMeanEmbedding, self.defiMeanEmbedding], 0)
        self.conceptVariEmbedding = torch.cat([self.primVariEmbedding, self.defiVariEmbedding], 0)
        
    def cudaModel(self):
        self.cuda()
        self.isCuda = True

    def cpuModel(self):
        self.cpu()
        self.isCuda = False
    
    
            