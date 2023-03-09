import collections
import pandas as pd

from config import DatapathArg, ModelArg
from mycode.utils.treeunit import NodeUnit

from rawdata import RawData, BaseData

class FineData(BaseData):
    def __init__(self, path_rawdata = None):
        if path_rawdata:
            rawdata:RawData = self.load(path_rawdata) # type: ignore 
            fulllist, defilist, primlist = self.creat_baselist(rawdata.basetree)
            attrdict, homodict = self.creat_attrhomodict(rawdata.basetree)
            negtdict = self.creat_negtdict(fulllist, rawdata.basetree)
            self.rawdata = rawdata
            self.fulllist = fulllist
            self.fullset  = set(fulllist)
            self.defilist = defilist
            self.primlist = primlist
            self.attrdict = attrdict
            self.homodict = homodict
            self.negtdict = negtdict
            self.attrDF = self.creat_DataFrame(attrdict)
            self.homoDF = self.creat_DataFrame(homodict)
            ModelArg.SN2E.num_defi = len(defilist)
            ModelArg.SN2E.num_prim = len(primlist)
            ModelArg.SN2E.num_full = len(fulllist)
            #self.negtDF = self.creat_DataFrame(negtdict)
            a = 0
    
    def creat_baselist(self, basetree:NodeUnit):
        def iter(node:NodeUnit):
            primset.update(node.attributes)
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
            attrdict[node.name].update(node.fathers)
            attrdict[node.name].update(node.attributes)
            homodict[node.name].update(node.attributes)
            for son in node.sons:
                homodict[son.name].update(homodict[node.name])
                iter(son) 
            for attribute in node.attributes:
                attrdict[attribute] = set()
                #homodict[attribute] = set()
                pass
            return
        attrdict = collections.defaultdict(set)
        homodict = collections.defaultdict(set)
        iter(basetree)
        return dict(attrdict), dict(homodict)

    def creat_negtdict(self, fulllist, basetree:NodeUnit):
        def creat_cogndict():
            def iter(node:NodeUnit):
                homodict[node.name].add(node.name)
                homodict[node.name].update(node.attributes)
                descdict[node.name].update(node.sons)
                for son in node.sons:
                    homodict[son.name].update(homodict[node.name])
                    iter(son)
                    descdict[node.name].update(descdict[son.name]) 
                cogndict[node.name].update(descdict[node.name]|homodict[node.name])
                for attribute in node.attributes:
                    cogndict[attribute].add(attribute)
                    cogndict[attribute].add(node.name)
                    cogndict[attribute].update(descdict[node.name])
                a = 0
            cogndict = collections.defaultdict(set)
            homodict = collections.defaultdict(set)
            descdict = collections.defaultdict(set)
            iter(basetree)
            return cogndict
        cogndict = creat_cogndict()
        fullset = set(fulllist)
        negtdict = {concept: list(fullset - cogndict[concept]) for concept in fullset}
        return negtdict
    
    def creat_DataFrame(self, sourcedict):
        return pd.DataFrame.from_dict(sourcedict, orient='index').T
        
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
