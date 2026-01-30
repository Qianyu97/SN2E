import torch
from torch import optim
import pandas as pd

from models import SN2E, TransA, TransD, TransE, TransH, KG2E
from models.base import Module


def initModel(modelArg, dataArg, modelPath=None, usegpu=True, gpunum=0)->Module:
    name = modelArg["name"]
    print(f"INFO -- prepare : Init model {name}")
    if usegpu:
        device = torch.device('cuda:'+str(gpunum))
        modelArg['device'] = device
        print(f"INFO -- prepare : set model cuda: {gpunum}")
    else:
        device = torch.device('cpu')
        modelArg['device'] = device
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
        model = SN2E.SN2E(modelArg, dataArg)
    else:
        print(f"ERROR : No model named {name}")
        raise Exception("Model Setting Error")
    if modelPath is None:
        model.initEmbedding()
        print("INFO -- prepare : Model initialization complete")
    else:
        model.loadCheckpoint(modelPath, device)
        print("INFO -- prepare : Model loading complete")
    return model

def initOptimizer(model:Module, **trainArg):
    print("INFO -- prepare : Init optimizer")
    optimizer = trainArg["optimizer"]
    learning_rate = trainArg["learningrate"]
    weight_decay = trainArg["weightdecay"]
    lr_decay = trainArg["lrdecay"]
    lr_decay_epoch = trainArg["lrdecayEpoch"]
    momentum = trainArg["momentum"]
    if optimizer == "Adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
        )
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr = learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
            momentum = momentum 
        )
    else:
        print("ERROR : Optimizer %s is not supported."%optimizer)
        raise Exception('Optimizer Error')
    optimizer.zero_grad()
    print("INFO -- prepare : Finish preparing Optimizer...")
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size = lr_decay_epoch, 
        gamma = lr_decay)
    return optimizer, scheduler