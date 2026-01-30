import torch
import pandas as pd
from utils import fileTool

#nodene embedding unit
type Embedding = tuple[torch.Tensor, torch.Tensor]
class EmbeddingOperator():
    @staticmethod
    def cat(embeddingList:list[Embedding]) -> Embedding:
        meanList, invaList = [], []
        for e in embeddingList:
            mean, invariance = e
            meanList.append(mean)
            invaList.append(invariance)
        newMean = torch.cat(meanList, dim=-2)
        newVar  = torch.cat(invaList, dim=-2)
        return newMean, newVar

#nodene concept unit
class BaseUnit(object):
    def __init__(self, name:str = ''):
        self.essence = name
        self.name = name if name is not None else 'None'
        self.index = 0
    
    def setIndex(self, index:int):
        self.index = index
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.essence)
    
    def __eq__(self, __o: object) -> bool:
        if __o is None:
            return __o == self.essence
        elif type(__o) == BaseUnit:
            return __o.essence == self.essence # type: ignore
        else:
            return __o == self.essence

class NodeUnit(BaseUnit):
    def __init__(self, 
                 name:str = '', 
                 father:'NodeUnit' = None, 
                 attributes:list['AttrUnit'] = None, 
                 children:list['NodeUnit'] = None,
                 depth:int = None):
        super().__init__(name)
        self.father = father
        self.attributes:set[AttrUnit] = set(attributes) if attributes is not None else set()
        self.children:set[NodeUnit] = set(children) if children is not None else set()
        self.depth = depth
        self.index = 0
    
    def setFather(self, father:'NodeUnit'):
        self.father = father
    
    def setChildren(self, children:list['NodeUnit']):
        self.children = set(children)
    
    def setAttributes(self, attributes:list['AttrUnit']):
        self.attributes = set(attributes)
    
    def setDepth(self, depth:int):
        self.depth = depth
    
    def addChild(self, child:'NodeUnit'):
        self.children.add(child)

    def addAttribute(self, attribute:'AttrUnit'):
        self.attributes.add(attribute)

class AttrUnit(BaseUnit):
    def __init__(self, name = '', children:list['NodeUnit'] = None):
        super().__init__(name)
        self.children = set(children) if children is not None else set()
        self.index = 0
    
    def addChild(self, child:NodeUnit):
        self.children.add(child)
    
    def setIndex(self, index:int):
        self.index = index

class Indexer_SN2E():
    def __init__(self, nodeList:list[NodeUnit]=[], attrList:list[AttrUnit]=[], load_dir=None):
        None_idx = -1
        abspath_nodeIndex = 'nodeIndex.json'
        abspath_attrIndex = 'attrIndex.json'
        nodeIndex = Indexer()
        attrIndex = Indexer()
        if load_dir is not None:
            nodeIndex.loadIndex(loadpath=load_dir + abspath_nodeIndex)
            attrIndex.loadIndex(loadpath=load_dir + abspath_attrIndex)
        else:
            nodeIndex.initIndex([i.name for i in nodeList])
            attrIndex.initIndex([i.name for i in attrList])
            nodeIndex.addSpecialIndex('None', None_idx)
            attrIndex.addSpecialIndex('None', None_idx)
        
        self.None_idx = None_idx
        self.abspath_nodeIndex = abspath_nodeIndex
        self.abspath_attrIndex = abspath_attrIndex
        self.nodeIndex = nodeIndex
        self.attrIndex = attrIndex
    
    def str2num_DataFrame(self, source:pd.DataFrame):
        newDataFrame:pd.DataFrame = source.map(lambda x: self.str2num(x))
        newDataFrame.index = source.index.map(lambda x: self.str2num(x))
        return newDataFrame

    def str2num(self, data):
        def translate(source):
            if type(source) == dict:
                return {translate(key) : translate(value) for key, value in source.items()}
            elif type(source) == list:
                return [translate(i) for i in source]
            elif type(source) == set:
                return {translate(i) for i in source}
            elif type(source) == NodeUnit:
                return self.nodeIndex.str2num(source.name)
            elif type(source) == AttrUnit:
                return self.attrIndex.str2num(source.name)
            elif type(source) == type(None):
                return self.None_idx
            elif type(source) == str:
                try:
                    return self.nodeIndex.index_str2num[source]
                except :
                    try:
                        return self.attrIndex.index_str2num[source]
                    except:
                        raise KeyError(f"dtype should be \'node\' or \'attr\'")
            else:
                raise TypeError(f"Unsupported index_str2num data type: {type(source)}")       
        output = translate(data)
        return output
    
    def num2str(self, data, dtype='node'):
        def translate(source):
            if type(source) == pd.DataFrame:
                newDataFrame:pd.DataFrame = source.map(lambda x: translate(x))
                newDataFrame.index = source.index.map(lambda x: translate(x))
                return newDataFrame
            elif type(source) == dict:
                return {translate(key) : translate(value) for key, value in source.items()}
            elif type(source) == list:
                return [translate(i) for i in source]
            elif type(source) == set:
                return {translate(i) for i in source}
            elif type(source) == int:
                if dtype == 'node':
                    return self.nodeIndex.num2str(source)
                elif dtype == 'attr':
                    return self.attrIndex.num2str(source)
                else:
                    Warning(f"dtype should be 'node' or 'attr', got {dtype}. Dedicate default \'node\'")
                    return self.nodeIndex.num2str[source]
            else:
                raise TypeError(f"Unsupported Index_num2str data type: {type(source)}")       
        output = translate(data)
        return output
    
    def saveIndex(self, data_dir):
        self.nodeIndex.saveIndex(data_dir + self.abspath_nodeIndex)
        self.attrIndex.saveIndex(data_dir + self.abspath_attrIndex)


class Indexer():
    def __init__(self):
        self.index_str2num:dict[str:int] = dict()
        self.index_num2str:list[str] = list()
    
    def num2str(self, number:int):
        return self.index_num2str[number]
    
    def str2num(self, string:str):
        return self.index_str2num[string]
    
    def initIndex(self, source):
        index_num2str = source
        index_str2num = {item: idx for idx, item in enumerate(index_num2str)}
        self.index_str2num = index_str2num
        self.index_num2str = index_num2str

    def loadIndex(self, loadpath):
        index_num2str = fileTool.load_json(loadpath)
        index_str2num = {item: idx for idx, item in enumerate(index_num2str)}
        self.index_str2num = index_str2num
        self.index_num2str = index_num2str

    def saveIndex(self, savepath):
        fileTool.save_json(savepath, self.index_num2str)
    
    def addSpecialIndex(self, specialToken:str, specialIndex:int):
        self.index_str2num[specialToken] = specialIndex
        self.index_num2str.append(specialToken)

    


    
        
