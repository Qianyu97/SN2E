
from nltk.corpus import wordnet as wn

from mycode.utils.treeunit import NodeUnit
from config import DatapathArg, WordnetArg

class BaseData():
    def load(self, filePath):
        import pickle
        try:
            with open(filePath, 'rb') as f:
                pickledata = pickle.load(f)
                f.close()
                return pickledata
        except Exception as e:
            print(e)
            return None
    def save(self, filePath, data = ''):
        import pickle
        try:
            with open(filePath, 'wb') as f:
                if not data:
                    data = self
                elif type(data) == str:
                    data = self.__getattribute__(data)  # type: ignore
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                f.close()
                print('Info : save sucessfully')
        except Exception as e:
            print(e)

class RawData(BaseData):
    def __init__(self, origword, maxdepth) -> None:
        basetree, basedict = self.creat_basetree(origword, maxdepth)
        basetree = self.enrichAttribute(basetree)
        self.basetree = basetree
        self.basedict = basedict
        
    def creat_basetree(self, origword, maxdepth):
        def iter(node_father:NodeUnit, syns_father, depth):
            if depth < 0:
                return
            for syns_son in syns_father.hyponyms():
                name_son = syns_son.name()[:-5]
                while name_son in basedict:
                    name_son = name_son + '+'
                node_son = NodeUnit(name_son)  
                node_father.addSon(node_son)
                node_son.addFather(node_father)
                basedict[name_son] = node_son
                iter(node_son, syns_son, depth - 1)
        
        orignode = NodeUnit(origword)
        origsyns = wn.synsets(origword)[0] # type: ignore
        basedict:dict[str, NodeUnit] = {origword: orignode}
        iter(orignode, origsyns, maxdepth)
        return orignode, basedict
            
    def enrichAttribute(self, basetree):
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
        iter(basetree, 0)
        return basetree



if __name__ == "__main__":
    rawdata = RawData(WordnetArg.originWord, WordnetArg.wordnet_depth)
    rawdata.save(DatapathArg.path_rawdata)

