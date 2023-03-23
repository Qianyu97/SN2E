import collections
import pandas as pd

from config import DatapathArg, ModelArg, DataloaderArg
from mycode.utils.treeunit import NodeUnit

from rawdata import RawData, BaseData

class FineData(BaseData):
    def __init__(self, path_rawdata = None):
        if path_rawdata:
            rawdata:RawData = self.load(path_rawdata) # type: ignore 
            fulllist, defilist, primlist = self.creat_baselist(rawdata.basetree)
            attrdict, homodict = self.creat_attrhomodict(rawdata.basetree)
            paredict = self.creat_paredict(rawdata.basedict)
            negtdict = self.creat_negtdict(defilist, primlist, rawdata.basetree)
            self.rawdata = rawdata
            self.fulllist = fulllist
            self.fullset  = set(fulllist)
            self.defilist = defilist
            self.primlist = primlist
            self.attrdict = attrdict
            self.homodict = homodict
            self.paredict = paredict
            self.negtdict = negtdict
            self.attrDF = self.creat_DataFrame(attrdict)
            self.homoDF = self.creat_DataFrame(homodict)
            self.pareDF = self.creat_DataFrame(paredict)
            ModelArg.SN2E.num_defi = len(defilist)
            ModelArg.SN2E.num_prim = len(primlist)
            ModelArg.SN2E.num_full = len(fulllist)
            ModelArg.SN2E.num_nodefi = ModelArg.SN2E.num_prim + 2
            #self.negtDF = self.creat_DataFrame(negtdict)
            a = 0
    
    def creat_baselist(self, basetree:NodeUnit):
        def iter(node:NodeUnit):
            primset.update(node.attributes) #[attribute.name for attribute in node.attributes]
            defiset.add(node.name)
            for son in node.sons:
                iter(son)
            return
        defiset, primset = set(), set()
        iter(basetree)
        defilist, primlist = list(defiset), list(primset)
        fulllist = primlist + defilist
        return fulllist, defilist, primlist

    def creat_attrhomodict(self, basetree):
        def iter(node:NodeUnit):
            attrdict[node.name].update(node.attributes)
            homodict[node.name].update(node.attributes)
            for son in node.sons:
                homodict[son.name].update(homodict[node.name])
                iter(son) 
            for attribute in node.attributes:
                #attrdict[attribute] = set()
                #homodict[attribute] = set()
                pass
            return
        attrdict = collections.defaultdict(set)
        homodict = collections.defaultdict(set)
        iter(basetree)
        #homodict['godfather'] = set()
        return dict(attrdict), dict(homodict)
    
    def creat_homodict_kt(self):
        def iter(node:NodeUnit):
            homodict[node.name].update(node.attributes)
            homodict[node.name].add(node.name)
            for son in node.sons:
                homodict[son.name].update(homodict[node.name])
                iter(son) 
            return
        homodict = collections.defaultdict(set)
        iter(self.rawdata.basetree)
        return dict(homodict)
    

    def creat_negtdict(self, defilist, primlist, basetree:NodeUnit):
        def creat_cogndict():
            def iter(node:NodeUnit):
                tempset = set()
                cogndict[node.name].update(node.attributes)
                for son in node.sons:
                    cogndict[son.name].update(cogndict[node.name])
                    iter(son)
                    tempset.update(cogndict[son.name]) 
                cogndict[node.name].update(tempset)
            cogndict = collections.defaultdict(set)
            iter(basetree)
            return cogndict
        cogndict = creat_cogndict()
        negtdict = {concept: list(set(primlist) - cogndict[concept]) for concept in defilist}
        return negtdict

    def creat_paredict(self, basedict:'dict[str, NodeUnit]'):
        return { k:list(v.fathers) for k, v in basedict.items()}
    
    def creat_DataFrame(self, sourcedict):
        return pd.DataFrame.from_dict(sourcedict, orient='index')
        
    def findMaxAttrnum(self, attrdict):
        maxAttrenum = 0
        itsname:str = ''
        attributes:list[str] = list()
        for key, value in attrdict.items():
            attrnum = len(value)
            if attrnum > maxAttrenum:
                itsname = key
                maxAttrenum = attrnum
                attributes = value
        print(itsname + ' has max attributes number, which is ' + str(maxAttrenum))
        print(str(attributes)) 
        

if __name__ == '__main__':
    finedata = FineData(DatapathArg.path_rawdata)
