
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
    negtnum = 100
    
class ModelArg():
    # Model and training general arguments
    class SN2E():
        name            = 'SN2E'
        dim             = 64
        lambdaMax       = 0.5
        gapMax          = -5
        vmax            = 10
        vmin            = 0.1
        alpha           = 0.1
        posCheckTurn    = 2
        negCheckTurn    = 2
        NoneIndex       = None
        num_defi        = 0
        num_prim        = 0
        
        learningrate    = 0.075
        weightdecay     = 0
        lrdecay         = 0
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
    epochs      = 1000
    ifLoadModel = False
    usegpu      = True
    gpunum      = 0
    evalepoch   = 1
    optimizer   = "SGD"
    evalmethod  = "MR"
    simmeasure  = "L2"
    measuretime = True
    measurefunc = 'trainer.run()'
    

class TestArg():
    threshold_lower = 0
    threshold_Upper = 2
    step = 0.1
    num_showpicture = 10
    
class WordnetArg():
    # Other arguments
    wordnet_depth = 8
    originWord = 'animal'
    a = TrainArg.epochs

class validateArg():
    field = TrainArg
    name = 'optimizer'
    candidate = ["SGD", "Adam", "Adadelta", "Adagrad"]
    
if __name__ == '__main__':
    WordnetArg.a = 5
    print(TrainArg.epochs)




