class ModelArg():
    def __init__(self, name=None, dim=0, device='cpu'):
        self.name = name
        self.dim = dim
    
class SN2E_Arg(ModelArg):
    def __init__(
            self,
            name        = 'SN2E',
            dim         = 64,
            lambdaMax   = -0.3,
            gapMax      = -3,
            gapmode     = 'entail',#'gap'
            logv_max    = 0,
            logv_min    = -5,
            alpha       = 1,
            ):
        super().__init__(name, dim)
        self.lambdaMax  = lambdaMax
        self.gapMax     = gapMax
        self.gapmode    = gapmode
        self.logv_max   = logv_max
        self.logv_min   = logv_min
        self.alpha      = alpha

        self.attr_num = 0
        self.defi_num = 0
        self.depth_range_record = dict()


class TransE(ModelArg):
    name   = 'TransE'
    Dim    = 100,
    Margin = 1.0,
    L      = 2
class TransH(ModelArg):
    name   = 'TransH'
    Dim    = 100,
    Margin = 1.0,
    L      = 2,
    C      = 0.01,
    Eps    = 0.001
class TransD(ModelArg):
    name   = 'TransD'
    EntDim =  100,
    RelDim =  100,
    Margin =  2.0,
    L      =  2
class TransA(ModelArg):
    name   = 'TransA'
    Dim    =  100,
    Margin =  3.2,
    L      =  2,
    Lamb   =  0.01,
    C      =  0.2
class KG2E(ModelArg):
    name   = 'KG2E'
    Dim    = 100,
    Margin = 4.0,
    Sim    = "EL",
    Vmin   = 0.03,
    Vmax   = 3.0