from unicodedata import name
from numpy.core.fromnumeric import var
import torch
from torch.autograd.variable import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mcode.models.base import Module

class SN2E(Module):
    def __init__(self, config): #conceptNum, dim, lambdaMax, gapMax, vmax, vmin, alpha):
        super(SN2E, self).__init__()
        self.modelName  = config.name
        self.dim        = config.Dim
        self.lambdaMax  = nn.Parameter(torch.Tensor([config.LambdaMax]))
        self.gapMax     = nn.Parameter(torch.Tensor([config.GapMax]))
        self.invmax     = 1/config.Vmin
        self.invmin     = 1/config.Vmax
        self.alpha      = config.Alpha
        self.NoneIndex  = config.conceptNum
        self.isCuda     = False

        self.conceptMeanEmbedding = nn.Embedding(num_embeddings=config.conceptNum + 1,
                                            embedding_dim=self.dim)
        self.conceptVariEmbedding = nn.Embedding(num_embeddings=config.conceptNum + 1,
                                            embedding_dim=self.dim)

        self.lambdaMax.requires_grad    = False
        self.gapMax.requires_grad       = False
        

    def lookupEmbedding(self, index, detach = False):
        '''
        index:  (index)[BatchNum, indexNum]

        mean:   (torch.tensor)[BatchNum, indexNum, dim]
        varInv: (torch.tensor)[BatchNum, indexNum, dim]
        '''
        index = self.variable(index)
        mean, varInv = self.conceptMeanEmbedding(index), self.conceptVariEmbedding(index)
        if detach:
            mean, varInv = mean.detach(), varInv.detach()
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
        if trainMode == 'posMode':
            indexA  = data
            lambd   = self.scorePos(indexA)
            loss    = torch.max(lambd, self.lambdaMax).sum()
        elif trainMode == 'negMode':
            [index0, indexN] = data
            gap     = self.scoreNeg(index0, indexN)
            loss    = - (1 / torch.max(gap, self.gapMax)).sum() * self.alpha
        return loss
    
    def initEmbedding(self, varInitMode = 'const'):
        nn.init.uniform_(self.conceptMeanEmbedding.weight.data, -1, 1)
        if varInitMode == 'const':
            nn.init.constant_(self.conceptVariEmbedding.weight.data, 1)
        self.tailingWorks()
    
    def variable(self, data):
        if self.isCuda:
            return data.cuda()
        else:
            return data
    
    def tailingWorks(self):
        def resetNoneEmbedding():
            self.conceptMeanEmbedding.weight.data[self.NoneIndex][:] = 0
            self.conceptVariEmbedding.weight.data[self.NoneIndex][:] = 0
        def limitVarRange():
            self.conceptVariEmbedding.weight.data.copy_(torch.clamp(input=self.conceptVariEmbedding.weight.detach(),
                                                       min=self.invmin,
                                                       max=self.invmax))
        limitVarRange()
        resetNoneEmbedding()

        return
        
    def evaluate(self, index1, index2):
        embedding1 = self.lookupEmbedding(index1)
        embedding2 = self.lookupEmbedding(index2)
        return self.calcSbProb(embedding1, embedding2)
    
    def setDefiConceptEmbedding(self, defiNumConcept, homoDF):
        homoTensor = self.variable(torch.tensor(np.asarray(homoDF).T))
        homoEmbedding = self.lookupEmbedding(homoTensor)
        [defiMean, defiVari] = self.calcIntersection(homoEmbedding)
        self.conceptMeanEmbedding.weight.data[defiNumConcept] = defiMean
        self.conceptVariEmbedding.weight.data[defiNumConcept] = defiVari

    
    
            