import pickle
import copy
import pandas as pd

from mycode.utils.configUtils import attrDict

def loadpickle(filePath):
    try:
        with open(filePath, 'rb') as f:
            pickledata = pickle.load(f)
            f.close()
            return pickledata
    except Exception as e:
        print(e)
        return None
    finally:
        pass

def savepickle(data, filePath):
    try:
        with open(filePath, 'wb') as f:
            #f.truncate()
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    except Exception as e:
        print(e)

def translate(source, Index):
    if type(source) == dict:
        return {translate(k, Index): translate(v, Index) for k, v in source.items()}
    elif type(source) == pd.DataFrame:
        newSource = source.applymap(lambda x: Index[x])
        newSource.columns = source.columns.map(lambda x: Index[x])
        return newSource
    elif type(source) == list or type(source) == set:
        return type(source)([Index[item] for item in source])
    elif type(source) == str or type(source) == int or type(source) == attrDict:
        return Index[source]
    else:
        newSource = copy.deepcopy(source)
        for k, v in vars(newSource).items():
            if k == 'conceptDict':
                continue
            setattr(newSource, k, translate(v, Index))
        return newSource


    

    
    

