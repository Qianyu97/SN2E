
import json
import pandas as pd
from collections import deque

from config import PathArg
from config_model import SN2E_Arg
from utils.treeunit import NodeUnit, AttributeUnit
from rawdata import RawData
class DefiIndex():
    def __init__(self, attrlist:list[AttributeUnit], loadList=None):
        num2str = [i.name for i in attrlist] if loadList is None else loadList
        str2num = {item: idx for idx, item in enumerate(num2str)}
        self.str2num = str2num
        self.num2str = num2str
    
class AttrIndex():
    def __init__(self, defiList:list[NodeUnit], loadList=None):
        num2str = [i.name for i in defiList] if loadList is None else loadList
        str2num = {item: idx for idx, item in enumerate(num2str)}
        self.str2num = str2num
        self.num2str = num2str

class FinalData(RawData):
    def __init__(self, data_dir, updateArg=None, ifLoadIndex=False):
        super().__init__(data_dir)
        '''
        self.defiList
        self.attrList
        self.defiUnitDict
        self.attrUnitDict 
        self.upperDict 
        self.attrDict 
        self.rootConcept 
        self.edges 
        self.depth_range_record'''
        self.data_dir = data_dir
        self.attrDF = self.create_DataFrame_fromdict(self.attrDict)
        self.homoDict = self.create_negtDict()
        self.negtDict = self.create_negtDict()

        if ifLoadIndex:
            defiList_loaded = self.load_json(data_dir + 'defiIndex_num2str.json')
            attrList_loaded = self.load_json(data_dir + 'attrIndex_num2str.json')
            self.defiIndexdict = AttrIndex(self.defiList, defiList_loaded)
            self.attrIndexdict = DefiIndex(self.attrList, attrList_loaded)
        else:
            self.defiIndexdict = DefiIndex(self.defiList)
            self.attrIndexdict = AttrIndex(self.attrList)
        
        self.defiList_idx:list[int] = self.str2num(self.defiList)
        self.attrList_idx:list[int] = self.str2num(self.attrList)
        self.upperDict_idx:dict[int, int] = self.str2num(self.upperDict)
        self.attrDF_idx:pd.DataFrame = self.str2num(self.attrDF)
        self.negtDict_idx:dict[int, set[int]] = self.str2num(self.negtDict)   
        self.indexDeficoncept()
        self.indexAttribute()
        if updateArg is not None:
            self.updateArg(updateArg)

    def indexDeficoncept(self):
        for concept in self.defiList:
            concept.setIndex(self.defiIndexdict.str2num[concept.name]) 
    
    def indexAttribute(self):
        for attribute in self.attrList:
            attribute.setIndex(self.attrIndexdict.str2num[attribute.name]) 
    
    def str2num(self, data):
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
            elif type(source) == NodeUnit:
                return self.defiIndexdict.str2num[source.name]
            elif type(source) == AttributeUnit:
                return self.attrIndexdict.str2num[source.name]
            elif type(source) == type(None):
                return 0
            elif type(source) == str:
                if source in self.defiList:
                    return self.defiIndexdict.str2num[source]
                elif source in self.attrList:
                    return self.attrIndexdict.str2num[source]
                else:
                    raise KeyError(f"String {source} not found in either defiList or attrList.")
            else:
                raise TypeError(f"Unsupported str2num data type: {type(source)}")       
        output = translate(data)
        return output
    
    def num2str(self, data, dtype='defi'):
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
                if dtype == 'defi':
                    return self.defiIndexdict.num2str[source]
                elif dtype == 'attr':
                    return self.attrIndexdict.num2str[source]
                else:
                    raise ValueError(f"dtype must be 'defi' or 'attr', got {dtype}")
            else:
                raise TypeError(f"Unsupported num2str data type: {type(source)}")       
        output = translate(data)
        return output
    
    def create_DataFrame_fromdict(self, sourcedict):
        return pd.DataFrame.from_dict(sourcedict, orient='index')
    
    def create_homoDict(self):
        attrbute_set = set(self.attrList)
        homoDict:dict[NodeUnit, set[AttributeUnit]] = {None: set()}
        queue = deque([self.origin])
        while queue:
            current_node = queue.popleft()
            queue.extend(current_node.children)
            homoDict[current_node] = homoDict[current_node.father] | current_node.attributes
        return homoDict

    def create_negtDict(self):
        attrbute_set = set(self.attrList)
        negtDict:dict[NodeUnit, set[AttributeUnit]] = {None: attrbute_set}
        queue = deque([self.origin])
        while queue:
            current_node = queue.popleft()
            queue.extend(current_node.children)
            negtDict[current_node] = negtDict[current_node.father] - current_node.attributes
        return negtDict

    def updateArg(self, arg:SN2E_Arg):
        arg.attr_num = len(self.attrIndexdict.num2str)
        arg.defi_num = len(self.defiIndexdict.num2str)
        arg.depth_range_record = self.depth_range_record
    
    def saveIndexDictionary(self, path:str=None):
        if path is None:
            path = self.data_dir
        self.save_json(path + 'defiIndex_num2str.json', self.defiIndexdict.num2str)
        self.save_json(path + 'attrIndex_num2str.json', self.attrIndexdict.num2str)

    

    def old_str2num(self, data, index_dictionary, arrow = 'str2num'):
        def translate(source):
            sourcetype = type(source)
            if sourcetype == FineData:
                newsource = FineData()
                for a in dir(source):
                    ga = source.__getattribute__(a)
                    if not callable(ga) and not a.startswith('__'):
                        newsource.__setattr__(a, translate(ga))
                return newsource
            elif sourcetype == pd.DataFrame:
                newDataFrame:pd.DataFrame = source.map(lambda x: dictionary[x])
                newDataFrame.index = source.index.map(lambda x: dictionary[x])
                return newDataFrame
            elif sourcetype == dict:
                return  {dictionary[key] : translate(value) for key, value in source.items()}
            elif sourcetype == list:
                return [dictionary[i] for i in source]
            elif sourcetype == set:
                return {dictionary[i] for i in source}
            elif sourcetype == str or sourcetype == int:
                return dictionary[source]
            else:
                return source
        assert arrow == 'str2num' or arrow == 'num2str'
        dictionary = self.dictionary['str'] if arrow == 'str2num' else self.dictionary['num'] 
        output = translate(data)
        return output

if __name__ == '__main__':
    pathArg  = PathArg()
    modelArg = SN2E_Arg()
    finaldata = FinalData(pathArg.dataDirectory, toUpdateArg=modelArg)
    a = 0