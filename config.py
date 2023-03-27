
class DatapathArg():
    # Data arguments
    path_rawdata    = 'source/data/rawdata.pkl'
    path_finedata   = 'source/data/finedata.pkl'
    path_indexdict  = 'source/data/indexdict.pkl'
    path_picture    = 'source/picture/'
    path_profiler   = 'source/model/profiler.lprof'

class DataloaderArg():
    # Dataloader arguments
    batchsize   = 16
    shuffle     = True
    numworkers  = 0
    droplast    = False
    negtsamplenum = 128
    len_attr    = 0
    
class ModelArg():
    # Model and training general arguments
    class SN2E():
        name        = 'SN2E'
        dim         = 128
        lambdaMax   = 1
        gapMax      = 3
        gapmode     = 'gap'
        vmax        = 10000
        vmin        = 0.0001
        alpha       = 1
        NoneIndex   = None
        learningrate    = 0.004
        weightdecay     = 0
        lrdecay         = 0.5
        lrdecayEpoch    = 300
        momentum        = 0
        num_defi    = 0
        num_prim    = 0
        num_full    = 0
        num_nodefi  = 0
    class TransE():
        name   = 'TransE'
        Dim    = 100,
        Margin = 1.0,
        L      = 2
    class TransH():
        name   = 'TransH'
        Dim    = 100,
        Margin = 1.0,
        L      = 2,
        C      = 0.01,
        Eps    = 0.001
    class TransD():
        name   = 'TransD'
        EntDim =  100,
        RelDim =  100,
        Margin =  2.0,
        L      =  2
    class TransA():
        name   = 'TransA'
        Dim    =  100,
        Margin =  3.2,
        L      =  2,
        Lamb   =  0.01,
        C      =  0.2
    class KG2E():
        name   = 'KG2E'
        Dim    = 100,
        Margin = 4.0,
        Sim    = "EL",
        Vmin   = 0.03,
        Vmax   = 3.0
    model = SN2E
    path_model = 'source/model/' + model.name
    
class TrainArg():
    # Training arguments
    epochs      = 2000
    usegpu      = True
    gpunum      = 1
    evalepoch   = 1
    optimizer   = "Adam"
    evalmethod  = "MR"
    simmeasure  = "L2"
    ifloadmodel = False

class TestArg():
    threshold_lower = 0
    threshold_Upper = 1
    step = 0.1
    num_showpicture = 10
    
class WordnetArg():
    # Other arguments
    wordnet_depth = 9
    originWord = 'animal'
    a = TrainArg.epochs

class validateArg():
    field = DataloaderArg
    name = 'batchsize'
    candidate = [8, 4]

class displayArg():
    epoch           = TrainArg.epochs
    optimizer       = TrainArg.optimizer
    batchsize       = DataloaderArg.batchsize
    negtsamplenum   = DataloaderArg.negtsamplenum
    modelname       = ModelArg.model.name
    dim             = ModelArg.model.dim
    lambdaMax       = ModelArg.model.lambdaMax
    gapMax          = ModelArg.model.gapMax
    gapmode         = ModelArg.model.gapmode
    vmax            = ModelArg.model.vmax
    vmin            = ModelArg.model.vmin
    alpha           = ModelArg.model.alpha
    learningrate    = ModelArg.model.learningrate
    weightdecay     = ModelArg.model.weightdecay
    lrdecay         = ModelArg.model.lrdecay
    lrdecayEpoch    = ModelArg.model.lrdecayEpoch
    momentum        = ModelArg.model.momentum
    
if __name__ == '__main__':
    WordnetArg.a = 5
    print(TrainArg.epochs)




