import torch
from torch import optim
import pandas as pd

from models import SN2E, TransA, TransD, TransE, TransH, KG2E
from models.base import Module
from config import TrainArg
from config_model import ModelArg

def initModel(modelArg:ModelArg, ifloadmodel = False, modelDir=None, usegpu=True, gpunum=0)->Module:
    name = modelArg.name
    print(f"INFO -- prepare : Init model {name}")
    if usegpu:
        device = torch.device('cuda:'+str(gpunum))
        print(f"INFO -- prepare : set model cuda: {gpunum}")
    else:
        device = torch.device('cpu')
        print("INFO -- prepare : set model cpu ")
    if name == "TransE":
        model = TransE.TransE()
    elif name == "TransH":
        model = TransH.TransH()
    elif name == "TransA":
        model = TransA.TransA()
    elif name == "TransD":
        model = TransD.TransD()
    elif name == "KG2E":
        model = KG2E.KG2E()
    elif name == "SN2E":
        model = SN2E.SN2E(modelArg, device)
    else:
        print(f"ERROR : No model named {name}")
        raise Exception("Model Setting Error")
    
    if ifloadmodel:
        if modelDir is None:
            raise Exception("Model directory must be provided when loading a model.")
        model.loadCheckpoint(modelDir + name + '.ckpt', device)
        model.batchEndWork()
        print("INFO -- prepare : Model loading complete")
    else:
        model.initEmbedding()
        model.batchEndWork()
        print("INFO -- prepare : Model initialization complete")
    return model

def initOptimizer(model:Module, train_Arg:TrainArg):
    print("INFO -- prepare : Init optimizer")
    optimizer_name = train_Arg.optimizer
    learning_rate = train_Arg.learningrate
    weight_decay = train_Arg.weightdecay
    lr_decay = train_Arg.lrdecay
    lr_decay_epoch = train_Arg.lrdecayEpoch
    momentum = train_Arg.momentum
    if optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr = learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
            momentum = momentum 
        )
    else:
        print("ERROR : Optimizer %s is not supported."%optimizer_name)
        raise Exception('Optimizer Error')
    optimizer.zero_grad()
    print("INFO -- prepare : Finish preparing Optimizer...")
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size = lr_decay_epoch, 
        gamma = lr_decay)
    return optimizer, scheduler