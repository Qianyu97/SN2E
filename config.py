# Data arguments
PathArg= {
    "dataDirectory" :'./data_full/',
    "pictureDirectory" :'./source/picture/',
    "modelDirectory" :'./source/model/',
    "indexDirectory" : './source/index/'
    }

# Dataloader arguments
DataloaderArg = {
    "batchsize"   : 128,
    "shuffle"     : True,
    "numworkers"  : 0,
    "droplast"    : False,
    "len_attr"    : 0,
    }
        
    
# Training arguments
TrainArg = {
    "negtsample_num" : 100,
    "epochs"         : 10000,
    "usegpu"         : True,
    "gpunum"         : 1,
    "evalepoch"      : 1,
    "optimizer"      : "Adam",
    "evalmethod"     : "MR",
    "learningrate"   : 0.01,
    "weightdecay"    : 0,
    "lrdecay"        : 0,
    "lrdecayEpoch"   : 300,
    "momentum"       : 0,
    }

TestArg = {
    "usegpu"          : True, 
    "gpunum"          : 1,
    "evalmethod"      : "MR",
    "threshold_lower" : 0,
    "threshold_Upper" : 1,
    "step"            : 0.1,
    "num_showpicture" : 10,
    }


'''class validateArg():
    field = DataloaderArg
    name = 'batchsize'
    candidate = [8, 4]'''

'''class displayArg():
    epoch           = TrainArg.epochs
    optimizer       = TrainArg.optimizer
    batchsize       = DataloaderArg.batchsize
    negtsample_num  = TrainArg.negtsample_num
    modelname       = ModelArg.model.name
    dim             = ModelArg.model.dim
    gammaaMax       = ModelArg.model.gammaaMax
    gapMax          = ModelArg.model.gapMax
    gapmode         = ModelArg.model.gapmode
    vmax            = ModelArg.model.logv_max
    vmin            = ModelArg.model.logv_min
    alpha           = ModelArg.model.alpha
    learningrate    = TrainArg.learningrate
    weightdecay     = TrainArg.weightdecay
    lrdecay         = TrainArg.lrdecay
    lrdecayEpoch    = TrainArg.lrdecayEpoch
    momentum        = TrainArg.momentum'''
    
if __name__ == '__main__':
    pass



