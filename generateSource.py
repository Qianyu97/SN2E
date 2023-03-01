from preProcess import preData
import pandas as pd
import numpy as np

import utils
from config import Config
class fineData():
    def __init__(self, preDataPath) -> None:
        preData = utils.loadpickle(preDataPath)
        self.attrDict = preData.attrDict
        self.trunkDict = preData.trunkDict

    def geneExtra(self):
        self.primConcepts, self.defiConcepts = self.geneConcepts(self.attrDict)
        self.conceptList = self.primConcepts + self.defiConcepts
        self.conceptDict = {item: idx for idx, item in enumerate(self.conceptList)}
        self.homoDict = self.geneHomoDict(self.attrDict, self.trunkDict)
        self.cognDict = self.geneCognDict(self.attrDict, self.trunkDict, self.conceptList)
        self.negtDict = self.geneNegtDict(self.conceptList, self.cognDict)
        self.sonsDict = self.geneSonsDict(self.attrDict, self.conceptList)
        self.attrDF = pd.DataFrame.from_dict(self.attrDict, orient='index').T
        self.homoDF = pd.DataFrame.from_dict(self.homoDict, orient='index').T
        self.conceptDict[None] = len(self.conceptList)
        
        a = 0

    @staticmethod
    def geneConcepts(attrDict):
        primConcepts = set()
        defiConcepts = list(attrDict.keys())
        for _ , value in attrDict.items():
            primConcepts.update(value)
        return list(primConcepts - set(defiConcepts)), defiConcepts

    def geneIndexDict(self, num2strList:list):
        str2numDict = {item: idx for idx, item in enumerate(num2strList)}
        str2numDict[None] = len(num2strList)
        return str2numDict

    def geneNegtDict(self, conceptList, cognDict):
        negtDict = dict()
        conceptSet = set(conceptList)
        for concept, cognitions in cognDict.items():
            negtDict[concept] = list(conceptSet - cognitions)
        return negtDict

    def geneCognDict(self, attrDict, trunkDict, allConcepts):
        cognDict = dict()
        homoKTDict = self.geneHomoDict(attrDict, trunkDict, ifKeepTrunk=True)
        descDict   = self.geneDescDict(homoKTDict, allConcepts)
        for concept in allConcepts:
            cognDict[concept] = homoKTDict.get(concept, set()) | descDict.get(concept, set()) | {concept}
        return cognDict

    @staticmethod
    def geneHomoDict(attrDict, trunkDict, ifKeepTrunk = False):
        def iter(concept):
            if concept is None:
                return set()
            else:
                parentConcept = trunkDict.get(concept)
                iter(parentConcept)
                homoDict[concept] = attrDict[concept] | homoDict.get(parentConcept, set())
                if not ifKeepTrunk and parentConcept is not None:
                    homoDict[concept].remove(parentConcept)
                
        homoDict = dict()
        for concept in attrDict.keys():
            if concept not in homoDict:
                iter(concept)
        return homoDict

    @staticmethod
    def geneDescDict(homoDict, allConcepts):
        descDict = dict()
        homoFlatDF = pd.DataFrame(np.zeros([len(allConcepts), len(homoDict)], dtype=bool))
        homoFlatDF.index, homoFlatDF.columns = allConcepts, homoDict.keys()
        for concept, homos in homoDict.items():
            homoFlatDF[concept][homos] = True
        descFlatDF = homoFlatDF.T
        for concept, descs in descFlatDF.items():
            descDict[concept] = set(descFlatDF.index[descs])
        return descDict
    
    def geneSonsDict(self, attrDict, allConcepts):
        sonsDict = dict()
        attrFlatDF = pd.DataFrame(np.zeros([len(allConcepts), len(attrDict)], dtype=bool))
        attrFlatDF.index, attrFlatDF.columns = allConcepts, attrDict.keys()
        for concept, attrs in attrDict.items():
            attrFlatDF[concept][attrs] = True
        sonsFlatDF = attrFlatDF.T
        for concept, sons in sonsFlatDF.items():
            msons = set(sonsFlatDF.index[sons])
            if len(msons) < 2:
                continue
            sonsDict[concept] = msons
        return sonsDict
    

    

class TriplesGen():
    def fromAttrDict(attrDictPath):
        pass

if __name__ == "__main__":
    configs = Config()
    mfineData = fineData(configs.preDataPath)
    mfineData.geneExtra()
    utils.savepickle(mfineData, configs.fineDataPath)
