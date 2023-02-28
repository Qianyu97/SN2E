
# -*- coding: utf-8 -*-

import torch

class Config():
    origname = 'animal'
    depth = 8
    path_wntree = 'source/data/wntree.pkl'
    path_rawdata   = 'source/data/preData.pkl'
    path_model     = 'source/model/'
    path_picture   = 'source/picture/'
    path_finedata  = 'source/data/fineData.pkl'
    def __init__(self):
        # Data arguments
        
        
        # Dataloader arguments
        self.batchsize = 2
        self.shuffle = True
        self.numworkers = 0
        self.droplast = False
        self.repproba = 0.5
        self.exproba = 0.5
        self.negsampleNum = 10

        self.model = SN2E()
        self.path_model = self.path_model + self.model.name
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
        # Other arguments

class Optimizer():
    def __init__(self) -> None:
        pass

class SN2E():
    def __init__(self) -> None:
        self.name       = 'SN2E'
        self.epochs     = 1000
        self.learningrate = 0.075
        self.Dim        = 8
        self.LambdaMax  = 1.0
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