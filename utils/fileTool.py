import pickle
import json
from pathlib import Path

def clear_jpg(path):
    for f in Path(path).glob("*.jpg"):
        f.unlink()

def loadpickle(filePath):
    try:
        with open(filePath, 'rb') as f:
            pickledata = pickle.load(f)
            f.close()
            return pickledata
    except Exception as e:
        print(e)
        return None

def savepickle(data, filePath):
    try:
        with open(filePath, 'wb') as f:
            #f.truncate()
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    except Exception as e:
        print(e)

def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except:
        raise Exception(f"Error loading JSON file: {path}")
    
def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


    

    
    

