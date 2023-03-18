
import torch
import torch.nn as nn
import torch.nn.parameter as pm
import numpy as np

from mycode.utils.treeunit import Embedding
from mycode.models.base import Module
from config import ModelArg


        

class SN2E(Module):
    def __init__(self): #conceptNum, dim, lambdaMax, gapMax, vmax, vmin, alpha):
        super(SN2E, self).__init__()
        self.modelName  = ModelArg.model.name
        self.dim        = ModelArg.model.dim
        #self.num_defi, self.num_prim = ModelArg.model.num_defi, ModelArg.model.num_prim
        self.num_nodefi = ModelArg.model.num_nodefi
        self.lambdaMax  = pm.Parameter(torch.Tensor([ModelArg.model.lambdaMax]), requires_grad = False)
        self.gapMax_prim = pm.Parameter(torch.Tensor([ModelArg.model.gapMax_prim]), requires_grad = False)
        self.gapMax_defi = pm.Parameter(torch.Tensor([ModelArg.model.gapMax_defi]), requires_grad = False)
        self.invmax     = 1/ModelArg.model.vmin
        self.invmin     = 1/ModelArg.model.vmax
        self.alpha      = ModelArg.model.alpha
        self.defaultNonemean   = 0
        self.defaultNoneInvar  = 0.00000001
        self.conceptMeanEmbedding = torch.nn.Embedding(self.num_nodefi, ModelArg.model.dim, padding_idx = 0)
        self.conceptVariEmbedding = torch.nn.Embedding(self.num_nodefi, ModelArg.model.dim, padding_idx = 0)
        self.isCuda     = False
        
    def initEmbedding(self, varInitMode = 'const'):
        nn.init.uniform_(self.conceptMeanEmbedding.weight, -1, 1)
        nn.init.uniform_(self.conceptVariEmbedding.weight, 0.1, 10)
        self.conceptMeanEmbedding.weight.data[0] = self.defaultNonemean
        self.conceptVariEmbedding.weight.data[0] = self.defaultNoneInvar
        self.conceptMeanEmbedding.weight.data[1] = 1000
        self.conceptVariEmbedding.weight.data[1] = 1000

    def lookupEmbedding(self, index):
        '''
        index:  (index)[batchNum, indexNum]
        mean:   (torch.tensor)[batchNum, indexNum, dim]
        varInv: (torch.tensor)[batchNum, indexNum, dim]
        '''
        index = self.variable(index)
        return Embedding(self.conceptMeanEmbedding(index), self.conceptVariEmbedding(index))

    def loaddefiEmbedding(self, index):
        homos = self.homoIndex[index - self.num_nodefi]
        homosEmbedding  = self.lookupEmbedding(homos)
        return self.calcIntersection(homosEmbedding)

    def calcIntersection(self, a:Embedding):
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varInvA    : (torch.tensor)[num0, numA, dim]

        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]
        '''
        varInvU = a.v.sum(-2)
        meanU = (a.v * a.m).sum(-2) / varInvU
        return Embedding(meanU, varInvU)

    def calcLambda(self, a:Embedding, u:Embedding)-> torch.Tensor:
        '''
        meanA      : (torch.tensor)[num0, numA, dim] 
        varInvA    : (torch.tensor)[num0, numA, dim]
        meanU      : (torch.tensor)[num0, dim]
        varInvU    : (torch.tensor)[num0, dim]

        lambd      : (torch.tensor)[num0]
        '''
        return - 1/2 * ( - (a.m.pow(2) * a.v).sum(-2) + u.m.pow(2) * u.v).sum(-1)

    def calcGap(self, s:Embedding, n:Embedding) -> torch.Tensor:
        '''
        mean0      : (torch.tensor)[num0, dim] 
        varInv0    : (torch.tensor)[num0, dim]
        meanN      : (torch.tensor)[num0, numN, dim]
        varInvN    : (torch.tensor)[num0, numN, dim]

        gap        : (torch.tensor)[num0, numN]
        '''
        return 1/2 * ( - (s.m.unsqueeze(-2) - n.m).pow(2).div(s.v.reciprocal().unsqueeze(-2) + n.v.reciprocal())).sum(-1)
    
    def calcEntailProb(self, s:Embedding, t:Embedding):
        return 1/2 * ( (1 + t.v.unsqueeze(-3) / s.v.unsqueeze(-2)).log() \
            + (s.m.unsqueeze(-2) - t.m.unsqueeze(-3)).pow(2).div(s.v.reciprocal().unsqueeze(-2)+t.v.reciprocal().unsqueeze(-3))).sum(-1)
    
    def calcKL(self, s:Embedding, t:Embedding):
        s = s.inv().unsqueeze(-2)
        t = t.inv().unsqueeze(-3)
        vardiv = s.v.div(t.v)
        return 1/2 * ( - vardiv.log() - 1 + vardiv + (s.m - t.m).pow(2).div(t.v)).sum(-1)

    def scorePos(self, indexA, indexP = None) -> torch.Tensor:
        '''
        indexA    : (index)[num0, numA]
        '''
        if not (indexP is None):
            e_p = self.loaddefiEmbedding(indexP)
            e_a = self.lookupEmbedding(indexA).cat(e_p)
        else:
            e_a = self.lookupEmbedding(indexA)
        e_u = self.calcIntersection(e_a)
        return self.calcLambda(e_a, e_u)

    def scoreNeg_prim(self, index0, indexN) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        e_0 = self.lookupEmbedding(index0)
        e_n = self.lookupEmbedding(indexN)
        return self.calcGap(e_0, e_n)
    
    def scoreNeg_defi(self, index0, indexN) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        e_0 = self.loaddefiEmbedding(index0)
        e_n = self.lookupEmbedding(indexN)
        return self.calcGap(e_0, e_n)
    
    def forward(self, data):
        '''
        index0    :(index)[num0]
        indexA    :(index)[num0, numA]
        indexN    :(index)[num0, numN]
        loss      :(torch.scale)
        '''
        [index0, indexN, indexA, indexP]  = data  #seppoint = (index0 < self.num_nodefi).sum()
        lambd   = self.scorePos(indexA, indexP)
        gap     = self.scoreNeg_defi(index0, indexN)  #gap_prim = self.scoreNeg_prim(index0[:seppoint], indexN[:seppoint])
        posloss = torch.max(lambd, self.lambdaMax).sum() 
        negloss = - (self.alpha / torch.max(gap, self.gapMax_defi)).sum()
        loss    = posloss + negloss
        showgap = - gap[gap>-10000]
        return loss, lambd.sum().item(), showgap.sum().item(), lambd.max().item(), showgap.min().item()
        
    def tailingWorks(self):
        self.conceptVariEmbedding.weight[1:].data.copy_(
            torch.clamp(
                input=self.conceptVariEmbedding.weight[1:].data,
                min=self.invmin,
                max=self.invmax))
    
    def scorePos_test(self, indexA) -> torch.Tensor:
        '''
        indexA    : (index)[num0, numA]
        '''
        e_a = self.lookupEmbedding_whole(indexA)
        embeddingU = self.calcIntersection(e_a)
        return self.calcLambda(e_a, embeddingU)
    
    def scoreNeg_test(self, index0, indexN) ->torch.Tensor:
        '''
        index0    :(index)[num0]
        indexN    :(index)[num0, numN]
        '''
        e_0 = self.lookupEmbedding_whole(index0)
        e_n = self.lookupEmbedding_whole(indexN)
        return self.calcGap(e_0, e_n)
    
    def generateWholeEmbedding(self):
        homoEmbedding = self.lookupEmbedding(self.homoIndex)
        defiEmebdding = self.calcIntersection(homoEmbedding)
        self.conceptmeanEmbedding_whole = torch.cat([self.conceptMeanEmbedding.weight, defiEmebdding.m])
        self.conceptvariEmbedding_whole = torch.cat([self.conceptVariEmbedding.weight, defiEmebdding.v])
    
    def lookupEmbedding_whole(self, index):
        index = self.variable(index)
        return Embedding(self.conceptmeanEmbedding_whole[index], self.conceptvariEmbedding_whole[index])
    
    def sethomoIndex(self, homoDF):
        self.homoIndex = self.variable(torch.tensor(np.asarray(homoDF).T))
    
    def variable(self, data):
        if not type(data) == torch.Tensor:
            if type(data) == int:
                data = torch.LongTensor([data])
            else:
                data = torch.LongTensor(data)
        if self.isCuda:
            return data.cuda(self.gpunum)
        else:
            return data
        
    def cudaModel(self, gpunum):
        self.cuda(gpunum)
        self.isCuda = True
        self.gpunum = gpunum

    def cpuModel(self):
        self.cpu()
        self.isCuda = False
        
    def evaluate(self, index1, index2):
        embedding1 = self.lookupEmbedding_whole(index1)
        embedding2 = self.lookupEmbedding_whole(index2)
        return self.calcEntailProb(embedding1, embedding2)

    
    
            