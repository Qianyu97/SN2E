
# -*- coding: utf-8 -*-

import torch

class Config():
    def __init__(self):
        # Data arguments
        self.dataPath = 'source/data/'
        self.modelPath = 'source/model/'
        self.picturePath = 'source/picture/'
        self.attrDictPath       = self.dataPath + 'attrDict.pkl'
        self.homoDictPath       = self.dataPath  + 'homoDict.pkl'
        self.attrDFPath         = self.dataPath  + 'attrDF.csv'
        self.homoDFPath         = self.dataPath  + 'homoDF.csv'
        self.conceptIndexPath   = self.dataPath  + 'conceptIndex.pkl'
        self.defiConceptsPath   = self.dataPath  + 'defiConcepts.pkl'
        self.primConceptsPath   = self.dataPath  + 'primConcepts.pkl'
        self.sonDictPath        = self.dataPath  + 'sonDict.pkl'
        self.trunkDictPath      = self.dataPath  + 'trunkDict.pkl'
        self.trunkOrderListPath = self.dataPath  + 'trunkOrderList.pkl'
        

        # Dataloader arguments
        self.batchsize = 2
        self.shuffle = True
        self.numworkers = 0
        self.droplast = False
        self.repproba = 0.5
        self.exproba = 0.5
        self.negsampleNum = 10

        self.model = SN2E()
        self.modelPath = self.modelPath + self.model.name
        self.ifLoadModel = False

        # Model and training general arguments
        self.usegpu = torch.cuda.is_available()
        self.gpunum = 0
        self.lrdecay = 1
        self.lrdecayepoch = 5
        self.weightdecay = 0
        self.lrDecay = 0
        self.weightDecay = 0
        self.evalepoch = 1
        self.optimizer = "SGD"
        self.evalmethod = "MR"
        self.simmeasure = "L2"
        self.loadembed = False
        self.entityfile = "./source/embed/entityEmbedding.txt"
        self.relationfile = "./source/embed/relationEmbedding.txt"
        self.premodel = "./source/model/TransE_ent128_rel128.param"

        # Other arguments
        self.summarydir = "./source/summary/KG2E_EL/"

        # Check Path
        #self.CheckPath()

        # self.usePaperConfig()
'''
    def usePaperConfig(self):
        # Paper best params
        if self.modelname == "TransE":
            self.embeddingdim = 50
            self.learningrate = 0.01
            self.margin = 1.0
            self.distance = 1
            self.simmeasure = "L1"
        elif self.modelname == "TransH":
            self.batchsize = 1200
            self.embeddingdim = 50
            self.learningrate = 0.005
            self.margin = 0.5
            self.C = 0.015625
        elif self.modelname == "TransD":
            self.batchsize = 4800
            self.entitydim = 100
            self.relationdim = 100
            self.margin = 2.0

    def CheckPath(self):
        # Check files
        CheckPath(self.pospath)
        CheckPath(self.validpath)

        # Check dirs
        CheckPath(self.modelpath, raise_error=False)
        CheckPath(self.summarydir, raise_error=False)
        CheckPath(self.logpath, raise_error=False)
        CheckPath(self.embedpath, raise_error=False)
        '''
class Optimizer():
    def __init__(self) -> None:
        pass

class SN2E():
    def __init__(self) -> None:
        self.name       = 'SN2E'
        self.epochs     = 1000
        self.learningrate = 0.01
        self.Dim        = 8
        self.LambdaMax  = 0
        self.GapMax     = -5
        self.Vmax       = 10
        self.Vmin       = 0.1
        self.Alpha      = 0.1
        self.posCheckTurn = 2
        self.negCheckTurn = 2
        self.NoneIndex = None

class TransE():
    def __init__(self) -> None:
        self.name   = 'TransE'
        self.Dim    = 100,
        self.Margin = 1.0,
        self.L      = 2

class TransH():
    def __init__(self) -> None:
        self.name   = 'TransH'
        self.Dim    = 100,
        self.Margin = 1.0,
        self.L      = 2,
        self.C      = 0.01,
        self.Eps    = 0.001

class TransD():
    def __init__(self) -> None:
        self.name   = 'TransD'
        self.EntDim =  100,
        self.RelDim =  100,
        self.Margin =  2.0,
        self.L      =  2

class TransA():
    def __init__(self) -> None:
        self.name   = 'TransA'
        self.Dim    =  100,
        self.Margin =  3.2,
        self.L      =  2,
        self.Lamb   =  0.01,
        self.C      =  0.2

class KG2E():
    def __init__(self) -> None:
        self.name   = 'KG2E'
        self.Dim    = 100,
        self.Margin = 4.0,
        self.Sim    = "EL",
        self.Vmin   = 0.03,
        self.Vmax   = 3.0