import collections

from rawdata import RawData
from mcode.utils import utils
from mcode.utils.treeunit import NodeUnit
from config import Config

class FineData():
    def __init__(self, path_rawdata = None):
        if path_rawdata:
            rawdata:RawData = utils.loadpickle(path_rawdata)
            fulllist, defilist, primlist = self.creat_baselist(rawdata.basetree)
            attrdict, homodict = self.creat_attrhomodict(rawdata.basetree)
            self.fulllist = fulllist
            self.defilist = defilist
            self.primlist = primlist
            self.attrdict = attrdict
            self.homodict = homodict
    
    def creat_baselist(self, basetree):
        def iter(node:NodeUnit):
            primset.update(node.attributes)
            defiset.add(node.name)
            for son in node.sons:
                iter(son)
            return
        defiset, primset = set(), set()
        iter(basetree)
        defilist, primlist = list(defiset), list(primset)
        fulllist = defilist + primlist
        return fulllist, defilist, primlist

    def creat_attrhomodict(self, basetree):
        def iter(node:NodeUnit):
            attrdict[node.name].update(node.attributes)
            homodict[node.name].update(node.attributes)
            for son in node.sons:
                homodict[son.name].update(homodict[node.name])
                iter(son) 
            return
        attrdict = collections.defaultdict(set)
        homodict = collections.defaultdict(set)
        iter(basetree)
        return dict(attrdict), dict(homodict)

if __name__ == '__main__':
    finedata = FineData(Config.path_rawdata)
    a = 0
