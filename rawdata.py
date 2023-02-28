from mycode.utils.treeunit import NodeUnit
from mycode.utils.utils import savepickle, loadpickle
from config import Config

class RawData():
    def __init__(self, path_wntree = None) -> None: 
        try:   
            basetree = loadpickle(path_wntree)
        except:
            basetree = NodeUnit()
            print('load wntree error, keep a empty TreeData')
        basetree = self.enrichAttribute(basetree)
        basedict = self.creat_basedict(basetree)
        self.basetree = basetree
        self.basedict = basedict
    
    def enrichAttribute(self, basetree:NodeUnit):
        import random
        startmark = 'a'
        def iter(node:NodeUnit, id):
            attrnum = random.randint(minAttrnum, maxAttrnum)
            attributes = [startmark + str(i) for i in range(id, id + attrnum)]
            node.updateAttribute(attributes)
            id += attrnum
            for son in node.sons:
                id = iter(son, id)
            return id
        maxAttrnum = 4
        minAttrnum = 2
        self.attrstage = dict()
        iter(basetree, 0)
        return basetree
    
    def creat_basedict(self, basetree):
        def iter(node:NodeUnit):
            basedict[node.name] = node
            for son in node.sons:
                iter(son)
        basedict = dict()
        iter(basetree)
        return basedict

if __name__ == '__main__':
    configs = Config()
    rawdata = RawData(configs.path_wntree) 
    savepickle(rawdata, configs.path_rawdata)
    