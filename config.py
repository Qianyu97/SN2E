class Config():
    class SN2E():
        name       = 'SN2E'
        epochs     = 1000
        learningrate = 0.075
        Dim        = 8
        LambdaMax  = 1.0
        GapMax     = -5
        Vmax       = 10
        Vmin       = 0.1
        Alpha      = 0.1
        posCheckTurn = 2
        negCheckTurn = 2
        NoneIndex = None
            
    # Data arguments
    origname = 'animal'
    depth = 8
    path_wntree = 'source/data/wntree.pkl'
    path_rawdata   = 'source/data/preData.pkl'
    path_model     = 'source/model/'
    path_picture   = 'source/picture/'
    path_finedata  = 'source/data/fineData.pkl'
    # Dataloader arguments
    batchsize = 2
    shuffle = True
    numworkers = 0
    droplast = False
    negsampleNum = 10
    # Model and training general arguments
    model = SN2E
    path_model = path_model + model.name
    ifLoadModel = False
    usegpu = True
    gpunum = 0
    evalepoch = 1
    optimizer = "SGD"
    evalmethod = "MR"
    simmeasure = "L2"
    # Other arguments
        
class Optimizer():
    def __init__(self) -> None:
        pass



class TransE():
    def __init__(self) -> None:
        self.name   = 'TransE'
        self.Dim    = 100,
        self.Margin = 1.0,
        self.L      = 2

class TransH():
    def __init__(self) -> None:
        self.name   = 'TransH'
        self.Dim    = 100,
        self.Margin = 1.0,
        self.L      = 2,
        self.C      = 0.01,
        self.Eps    = 0.001

class TransD():
    def __init__(self) -> None:
        self.name   = 'TransD'
        self.EntDim =  100,
        self.RelDim =  100,
        self.Margin =  2.0,
        self.L      =  2

class TransA():
    def __init__(self) -> None:
        self.name   = 'TransA'
        self.Dim    =  100,
        self.Margin =  3.2,
        self.L      =  2,
        self.Lamb   =  0.01,
        self.C      =  0.2

class KG2E():
    def __init__(self) -> None:
        self.name   = 'KG2E'
        self.Dim    = 100,
        self.Margin = 4.0,
        self.Sim    = "EL",
        self.Vmin   = 0.03,
        self.Vmax   = 3.0