import collections
from pickle import GLOBAL
from mcode.utils import utils
from config import Config
class AttrDictGen():
    def __init__(self, config:Config) -> None:
        self.attrDict = utils.loadpickle(config.attrDictPath)
        self.trunkDict = utils.loadpickle(config.trunkDictPath)
        self.trunkOrderList = utils.loadpickle(config.trunkOrderListPath)
        self.conceptList = None
        self.conceptDict = None

    def fromAttrDict(self, attrDictPath):
        self.attrDict = utils.loadpickle(attrDictPath)
        
        self.sonDict  = self.geneSonDict(self.attrDict)
        self.defiConcepts = list(self.attrDict.keys())
        self.primConcepts = self.genePrimConcepts(self.attrDict)
        self.conceptList = list(set.union(set(self.defiConcepts), set(self.primConcepts)))
        self.conceptDict = self.geneIndexDict(self.conceptList)
        self.homoDict = self.geneHomoDict(self.attrDict, self.trunkDict, self.trunkOrderList)
        
    def geneIndexDict(self, indexList:list):
        indexDict = {item: idx for idx, item in enumerate(indexList)}
        indexDict[None] = len(indexList)
        return indexDict
    
    def genDrawSample(self):
        pass

    def geneSonDict(self, attrDict):
        sonDict = collections.defaultdict(set)
        for key, valueSet in attrDict.items():
            for value in valueSet:
                if value in self.trunkOrderList:
                    sonDict[value].add(key)
        return dict(sonDict)
    
    def geneCognDict(self, homoDict):

        pass
    
    def geneHomoDict(self, attrDict, trunkDict, trunkList, ifKeepTrunk = False):
        '''
        def iter(concept):
            if concept is None:
                return set()
            else:
                
                parentConcept = trunkDict.get(concept)
                homoDict[concept] = attrDict[concept] | iter(parentConcept)
                if ifKeepTrunk:
                    pass
                return homoDict
        homoDict = dict()
        for concept in attrDict.keys():
            if concept not in homoDict:
                iter(concept, homoDict, attrDict, trunkDict)
                '''

        for concept in self.trunkOrderList:
            fatherConcept = trunkDict.get(concept)
            homoDict[concept] = attrDict[concept]
            #if fatherConcept is not None:
            #    homoDict[concept] = set.union(homoDict[concept], homoDict.get(fatherConcept, set())) - {fatherConcept}
        return homoDict
        


    @staticmethod
    def genePrimConcepts(attrdict:dict):
        primConcepts = set()
        for _ , value in attrdict.items():
            primConcepts.update(value)
        return list(primConcepts)
    
    def save(self, configs):
        #self.attrDF.to_csv(configs.attrDFPath, sep='\t', encoding="utf-8")
        utils.savepickle({'str2num': self.conceptDict, 'num2str': self.conceptList}, configs.conceptIndexPath)
        utils.savepickle(self.defiConcepts, configs.defiConceptsPath)
        utils.savepickle(self.primConcepts, configs.primConceptsPath)
        utils.savepickle(self.homoDict, configs.homoDictPath)
        utils.savepickle(self.sonDict, configs.sonDictPath)

        



class TriplesGen():
    def fromAttrDict(attrDictPath):
        pass

if __name__ == "__main__":
    configs = Config()
    attrDictGen = AttrDictGen(configs)
    attrDictGen.fromAttrDict(configs.attrDictPath)
    attrDictGen.save(configs)
