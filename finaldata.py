import pandas as pd

from utils.unit import NodeUnit, AttrUnit, Indexer_SN2E
from rawdata import RawData

def create_DataFrame_fromdict(dictionary):
    return pd.DataFrame.from_dict(dictionary, orient='index')

class FinalData(RawData):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.attrDF   = create_DataFrame_fromdict(self.attrDict)
        self.homoDF   = create_DataFrame_fromdict(self.homoDict)
        self.upperDF  = create_DataFrame_fromdict(self.upperDict)
        
    def lookupConcept(self, concept:str):       
        try:
            return self.nodeUnitDict[concept]
        except :
            try:
                return self.attrUnitDict[concept]
            except:
                raise KeyError(f"string \'{concept}\' not found in node or attribute list")
    
    def indexConceptUnit(self, indexer:Indexer_SN2E):
        for concept in self.nodeList:
            concept.setIndex(indexer.str2num(concept.name)) 
        for attribute in self.attrList:
            attribute.setIndex(indexer.str2num(attribute.name)) 
    
    def returnDataParams(self):
        dataparams = {
            "attr_num" : len(self.attrList),
            "node_num" : len(self.nodeList),
            "depth_range_record" : self.depth_range_record}
        return dataparams

if __name__ == '__main__':
    from config import PathArg
    finaldata = FinalData(PathArg["dataDirectory"])
    myIndex = Indexer_SN2E(
        nodeList=finaldata.nodeList,
        attrList=finaldata.attrList
        )
    myIndex.saveIndex(PathArg['indexDirectory'])
    a = 0