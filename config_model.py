

SN2EArg ={
    "name"        : 'SN2E',
    "dim"         : 32,
    "gammaMax"    : -1,
    "gapMax"      : -0.2,
    "gapmode"     : 'entail',
    "logv_max"    : 0,
    "logv_min"    : -5,
    "alpha"       : 1,
}
        


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