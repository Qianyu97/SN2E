
class DatapathArg():
    # Data arguments
    path_wntree     = 'source/data/wntree.pkl'
    path_rawdata    = 'source/data/preData.pkl'
    path_picture    = 'source/picture/'
    path_finedata   = 'source/data/fineData.pkl'

class DataloaderArg():
    # Dataloader arguments
    batchsize   = 2
    shuffle     = True
    numworkers  = 0
    droplast    = False
    negsampleNum = 10
    
class ModelArg():
    # Model and training general arguments
    class SN2E():
        name            = 'SN2E'
        learningrate    = 0.075
        weightdecay     = 0
        lrdecay         = 1
        Dim             = 8
        LambdaMax       = 1.0
        GapMax          = -5
        Vmax            = 10
        Vmin            = 0.1
        Alpha           = 0.1
        posCheckTurn    = 2
        negCheckTurn    = 2
        NoneIndex       = None
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
    epochs          = 1000
    ifLoadModel = False
    usegpu      = True
    gpunum      = 0
    evalepoch   = 1
    optimizer   = "SGD"
    evalmethod  = "MR"
    simmeasure  = "L2"
    
    
class OtherArg():
    # Other arguments
    wordnet_depth = 8
    originWord = 'animal'
    
        



