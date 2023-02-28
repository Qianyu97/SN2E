from mycode.utils.configUtils import attrDict
from mycode.utils.utils import savepickle
from config import Config
class preData():
    def __init__(self) -> None:    
        self.attrDict = {
                    'entity'        : {'thing', attrDict('can be_touch'), attrDict('real')},
                    'organism'      : {'entity', attrDict('living'), attrDict('can move'), },
                    'unorganism'    : {'entity', attrDict('television'), attrDict('cup')},
                    'nature_thing'  : {'entity', attrDict('cloud'), attrDict('river'), attrDict('mountain')},
                    'plankton'      : {'organism', attrDict('small'), attrDict('aggregateof'), },
                    'hybrid'        : {'organism', attrDict('offspring'), attrDict('stock'), }, 
                    'animal'        : {'organism', attrDict('can breathes'), attrDict('has live'), attrDict('can eat'), }, \
                    'fish'          : {'animal' , attrDict('has gill'),attrDict('live_in water')}, \
                    'bird'          : {'animal' , attrDict('can fly'), attrDict('has feather'), attrDict('has egg')},\
                    'mammal'        : {'animal' , attrDict('has stable_tempurature'), attrDict('can milk')},\
                    'salmon'        : {'fish', attrDict('yellow'), attrDict('small')},\
                    'shark'         : {'fish', attrDict('has big_gill'), attrDict('can meat'), attrDict('good')},\
                    'canary'        : {'bird', attrDict('yellow'), attrDict('can sing'), attrDict('small')},\
                    'dog'           : {'mammal', attrDict('live_in land'), attrDict('can meat'), attrDict('smart')},\
                    'dolphin'       : {'mammal', attrDict('smart'), attrDict('live_in water'), attrDict('fat')},
                    'penguin'       : {'bird', attrDict('has gill'),attrDict('fat')}}
        self.trunkDict = {
                    #'entity'        : 'thing',
                    'organism'      : 'entity',
                    'unorganism'    : 'entity',
                    'nature_thing'  : 'entity',
                    'plankton'      : 'organism',
                    'hybrid'        : 'organism', 
                    'animal'        : 'organism', 
                    'fish'          : 'animal', 
                    'bird'          : 'animal',
                    'mammal'        : 'animal',
                    'salmon'        :'fish',
                    'shark'         :'fish',
                    'canary'        :'bird',
                    'dog'           : 'mammal',
                    'dolphin'       : 'mammal',
                    'penguin'       : 'bird'
                }



if __name__ == '__main__':
    configs = Config()
    mPreData = preData() 
    savepickle(mPreData, configs.preDataPath)
    