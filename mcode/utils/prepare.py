import torch
from code import models

from mcode.models import SN2E
from mcode.datasets.attrDataset import attrDataset
from mcode.datasets.tripleDataset import tripleDataset

def prepareModel(configs,  ifLoadModel):
    
    print("INFO : Init model %s"%configs.model.name)
    if configs.model.name == "TransE":
        model = TransE.TransE(configs.model)
    elif configs.model.name == "TransH":
        model = TransH.TransH(configs.model)
    elif configs.model.name == "TransA":
        model = TransA.TransA(configs.model)
    elif configs.model.name == "TransD":
        model = TransD.TransD(configs.model)
    elif configs.model.name == "KG2E":
        model = KG2E.KG2E(configs.model)
    elif configs.model.name == "SN2E":
        model = SN2E.SN2E(configs.model, configs.defiConNum, configs.primConNum)
    else:
        print("ERROR : No model named %s"%configs.model.name)
        raise Exception("Model Setting Error")
    if configs.usegpu:
        model.cuda()
        model.isCuda = True
    if ifLoadModel:
        model.loadCheckpoint(configs.modelPath)
        model.catTogether()
    else:
        model.initEmbedding()
    return model

def prepareOptimizer(configs, model):
    
    OPTIMIZER   = configs.optimizer
    LR          = configs.model.learningrate
    weightDecay = configs.weightDecay
    lrDecay     = configs.lrDecay

    if OPTIMIZER == "Adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=LR,
            lr_decay=lrDecay,
            weight_decay=weightDecay,
        )
    elif OPTIMIZER == "Adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr = LR,
            weight_decay=weightDecay,
        )
    elif OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = LR,
            weight_decay=weightDecay,
        )
    elif OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr = LR,
            weight_decay=weightDecay,
        )
    else:
        print("ERROR : Optimizer %s is not supported."%OPTIMIZER)
        raise Exception('Optimizer Error')
    print("Finish preparing Optimizer...")
    return optimizer

def prepareDataSet(config):
    
    if config.model.name == 'SN2E':
        return attrDataset(config)
    else:
        return tripleDataset(config.entityDictPath,
                            config.relationDictPath,
                            config.posDataPath)