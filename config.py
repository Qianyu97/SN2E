
class DatapathArg():
    # Data arguments
    path_rawdata    = 'source/data/rawdata.pkl'
    path_finedata   = 'source/data/finedata.pkl'
    path_indexdict  = 'source/data/indexdict.pkl'
    path_picture    = 'source/picture/'
    path_profiler   = 'source/model/profiler.lprof'

class DataloaderArg():
    # Dataloader arguments
    batchsize   = 128
    shuffle     = True
    numworkers  = 0
    droplast    = False
    negtsamplenum = 128
    len_attr    = 0
    
class ModelArg():
    # Model and training general arguments
    class SN2E():
        name        = 'SN2E'
        dim         = 64
        lambdaMax   = 2
        gapMax_prim = -0.5
        gapMax_defi = -1
        vmax        = 100
        vmin        = 0.01
        alpha       = 1
        NoneIndex   = None
        num_defi    = 0
        num_prim    = 0
        num_full    = 0
        
        
        learningrate    = 0.002
        weightdecay     = 0
        lrdecay         = 1
        lrdecayEpoch    = 250
        momentum        = 0
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
    epochs      = 500
    ifLoadModel = False
    usegpu      = True
    gpunum      = 0
    evalepoch   = 1
    optimizer   = "Adam"
    evalmethod  = "MR"
    simmeasure  = "L2"

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
    field = ModelArg.model
    name = 'learningrate'
    candidate = [0.00025, 0.0005, 0.001, 0.002, 0.004]

class displayArg():
    epoch           = TrainArg.epochs
    optimizer       = TrainArg.optimizer
    batchsize       = DataloaderArg.batchsize
    negtsamplenum   = DataloaderArg.negtsamplenum
    modelname       = ModelArg.model.name
    dim             = ModelArg.model.dim
    lambdaMax       = ModelArg.model.lambdaMax
    gapMax_prim     = ModelArg.model.gapMax_prim
    gapMax_defi     = ModelArg.model.gapMax_defi
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




