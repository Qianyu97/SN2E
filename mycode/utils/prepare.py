import torch
from torch import optim
import pandas as pd

from mycode.models import SN2E, TransA, TransD, TransE, TransH, KG2E
from mycode.models.base import Module
from config import ModelArg, TrainArg

def prepareModel(homoDF:pd.DataFrame,ifLoadModel = False):
    print("INFO -- prepare : Init model %s" % ModelArg.model.name)
    if ModelArg.model.name == "TransE":
        model = TransE.TransE(ModelArg.model)
    elif ModelArg.model.name == "TransH":
        model = TransH.TransH(ModelArg.model)
    elif ModelArg.model.name == "TransA":
        model = TransA.TransA(ModelArg.model)
    elif ModelArg.model.name == "TransD":
        model = TransD.TransD(ModelArg.model)
    elif ModelArg.model.name == "KG2E":
        model = KG2E.KG2E(ModelArg.model)
    elif ModelArg.model.name == "SN2E":
        model:SN2E.SN2E = SN2E.SN2E()
    else:
        print("ERROR : No model named %s" % ModelArg.model.name)
        raise Exception("Model Setting Error")
    if TrainArg.usegpu:
        model.cudaModel(TrainArg.gpunum)
        device = torch.device('cuda:'+str(TrainArg.gpunum))
        print("INFO -- prepare : set model cuda: " + str(TrainArg.gpunum) )
    else:
        model.cpuModel()
        device = torch.device('cpu')
        print("INFO -- prepare : set model cpu ")
    if ifLoadModel:
        model.loadCheckpoint(ModelArg.path_model, device)
        model.sethomoIndex(homoDF.sort_index(axis = 1))
        model.tailingWorks()
        model.generateWholeEmbedding()
        print("INFO -- prepare : Model loading complete")
    else:
        model.initEmbedding()
        model.sethomoIndex(homoDF.sort_index(axis = 1))
        model.tailingWorks()
        print("INFO -- prepare : Model initialization complete")
    return model

def prepareOptimizer(model:Module):
    print("INFO -- prepare : Init optimizer")
    OPTIMIZER   = TrainArg.optimizer
    LR          = ModelArg.model.learningrate
    weightdecay = ModelArg.model.weightdecay
    lrdecay     = ModelArg.model.lrdecay
    momentum    = ModelArg.model.momentum
    if OPTIMIZER == "Adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=LR,
            lr_decay=lrdecay,
            weight_decay=weightdecay,
        )
    elif OPTIMIZER == "Adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr = LR,
            weight_decay=weightdecay,
        )
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr = LR,
            weight_decay=weightdecay,
        )
    elif OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr = LR,
            weight_decay = weightdecay,
            momentum = momentum 
        )
    else:
        print("ERROR : Optimizer %s is not supported."%OPTIMIZER)
        raise Exception('Optimizer Error')
    optimizer.zero_grad()
    print("INFO -- prepare : Finish preparing Optimizer...")
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size = ModelArg.model.lrdecayEpoch, 
        gamma = ModelArg.model.lrdecay)
    
    return optimizer, scheduler