from torch import optim

from mycode.models import SN2E, TransA, TransD, TransE, TransH, KG2E
from mycode.models.base import Module
from config import ModelArg, TrainArg

def prepareModel(ifLoadModel = False):
    print("INFO : Init model %s" % ModelArg.model.name)
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
        model:SN2E.SN2E = SN2E.SN2E(ModelArg.model)
    else:
        print("ERROR : No model named %s" % ModelArg.model.name)
        raise Exception("Model Setting Error")
    if TrainArg.usegpu:
        model.cuda()
        #model.isCuda = True
    if ifLoadModel:
        model.loadCheckpoint(ModelArg.path_model)
        #model.catTogether()
    else:
        model.initEmbedding()
    return model

def prepareOptimizer(model:Module):
    
    OPTIMIZER   = TrainArg.optimizer
    LR          = ModelArg.model.learningrate
    weightdecay = ModelArg.model.weightdecay
    lrdecay     = ModelArg.model.lrdecay

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
            weight_decay=weightdecay,
        )
    else:
        print("ERROR : Optimizer %s is not supported."%OPTIMIZER)
        raise Exception('Optimizer Error')
    print("Finish preparing Optimizer...")
    return optimizer