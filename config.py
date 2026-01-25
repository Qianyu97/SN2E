

# Data arguments
class PathArg():
    def __init__(self):
        self.dataDirectory      = 'source/data/'
        self.pictureDirectory    = 'source/picture/'
        self.modelDirectory      = 'source/model/'

# Dataloader arguments
class DataloaderArg():
    def __init__(self):
        self.batchsize   = 128
        self.shuffle     = True
        self.numworkers  = 0
        self.droplast    = False
        self.len_attr    = 0
    
# Training arguments
class TrainArg():
    def __init__(self, 
                negtsample_num = 10,
                epochs         = 100,
                usegpu         = True,
                gpunum         = 1,
                evalepoch      = 1,
                optimizer      = "Adam",
                evalmethod     = "MR",
                learningrate   = 0.01,
                weightdecay    = 0,
                lrdecay        = 0.1,
                lrdecayEpoch   = 300,
                momentum       = 0,
                ):
        self.negtsample_num = negtsample_num
        self.epochs         = epochs 
        self.usegpu         = usegpu
        self.gpunum         = gpunum 
        self.evalepoch      = evalepoch
        self.optimizer      = optimizer
        self.evalmethod     = evalmethod
        self.learningrate   = learningrate
        self.weightdecay    = weightdecay
        self.lrdecay        = lrdecay
        self.lrdecayEpoch   = lrdecayEpoch
        self.momentum       = momentum
    

class TestArg():
    def __init__(self, 
                usegpu         = True,
                gpunum         = 1,
                evalmethod     = "MR",
                threshold_lower = 0,
                threshold_Upper = 1,
                step            = 0.1,
                num_showpicture = 10,
                ):
        self.usegpu         = usegpu
        self.gpunum         = gpunum 
        self.evalmethod     = evalmethod
        self.threshold_lower = threshold_lower
        self.threshold_Upper = threshold_Upper
        self.step            = step
        self.num_showpicture = num_showpicture

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
    lambdaMax       = ModelArg.model.lambdaMax
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



