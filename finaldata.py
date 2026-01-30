
import json
import pandas as pd
from collections import deque
from utils.unit import NodeUnit, AttrUnit, Indexer_SN2E
from utils.fileTool import load_json, save_json
from rawdata import RawData

def create_DataFrame_fromdict(dictionary):
    return pd.DataFrame.from_dict(dictionary, orient='index')

def create_homoDict(attrList:list[AttributeError], origin:NodeUnit):
    attrbute_set = set(attrList)
    homoDict:dict[NodeUnit, set[AttrUnit]] = {None: set()}
    queue = deque([origin])
    while queue:
        current_node = queue.popleft()
        queue.extend(current_node.children)
        homoDict[current_node] = homoDict[current_node.father] | current_node.attributes
    del homoDict[None]
    return homoDict

def create_negtDict(attrList:list[AttributeError], origin:NodeUnit):
    attrbute_set = set(attrList)
    negtDict:dict[NodeUnit, set[AttrUnit]] = {None: attrbute_set}
    queue = deque([origin])
    while queue:
        current_node = queue.popleft()
        queue.extend(current_node.children)
        negtDict[current_node] = negtDict[current_node.father] - current_node.attributes
    del negtDict[None]
    return negtDict

class FinalData(RawData):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.homoDict = create_homoDict(self.attrList, self.origin)
        self.negtDict = create_negtDict(self.attrList, self.origin)
        self.attrDF   = create_DataFrame_fromdict(self.attrDict)
        self.homoDF   = create_DataFrame_fromdict(self.homoDict)
        self.upperDF  = create_DataFrame_fromdict(self.upperDict)

    def lookupConcept(self, concept:str):       
        try:
            return self.nodeUnitDict.get(concept)
        except :
            try:
                return self.attrUnitDict.get(concept)
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