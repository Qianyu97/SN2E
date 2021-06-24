import pickle
import pandas as pd

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

def translateIndex(source, conceptIndex):
    if type(source) == pd.DataFrame:
        newSource = source.applymap(lambda x: conceptIndex[x])
        newSource.columns = source.columns.map(lambda x: conceptIndex[x])
        return newSource
    elif type(source) == list or set:
        return type(source)([conceptIndex[item] for item in source])
    else:
        return conceptIndex[source]


    

    
    

