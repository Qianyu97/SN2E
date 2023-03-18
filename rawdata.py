
from nltk.corpus import wordnet as wn
import collections

from mycode.utils.treeunit import NodeUnit
from mycode.models.SN2E import SN2E
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
        basetree, attrbasedict = self.enrichAttribute(basetree)
        self.basetree = basetree
        self.basedict = basedict
        self.attrbasedict = attrbasedict
        
    def creat_basetree(self, origword, maxdepth):
        def iter(node_father:NodeUnit, syns_father, depth):  
            node_father.depth = maxdepth - depth
            if depth <= 0:
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
        orignode.addFather('godfather')
        return orignode, basedict
    
    def enrichAttribute(self, basetree):
        import random
        startmark = 'a'
        def iter(node:NodeUnit, id):
            attrnum = random.randint(minAttrnum, maxAttrnum)
            upperId = id + int(levelnumdict[node.depth])
            if upperId > attrnum:
                attributes = random.sample(range(id, upperId), attrnum) 
            else:
                upperId = attrnum
                attributes =range(id, id + attrnum)
            for i in attributes:
                name = startmark + str(i)
                if name in attrbasedict:
                    attribute = attrbasedict[name]
                else:
                    attribute = NodeUnit(name)
                    attrbasedict[name] = attribute
                attribute.addSon(node.name)
                node.addAttribute(attribute)
            for son in node.sons:
                iter(son, upperId)
            
        maxAttrnum = 4
        minAttrnum = 2
        levelnumdict  = self.countlevelnum(basetree)
        attrbasedict:dict[str,NodeUnit] = dict()
        iter(basetree, 0)
        return basetree, attrbasedict
    
    '''def enrichAttribute(self, basetree):
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
        return basetree'''
    
    def countlevelnum(self, tree:NodeUnit):
        def iter(node:NodeUnit):
            leveldict[node.depth] += 1
            for son in node.sons:
                iter(son)
        leveldict = collections.defaultdict(int)
        iter(tree)
        return leveldict
    
    def printree(self, startconcept = None, targetdepth = 1):
        def iter(node:NodeUnit, depth):
            if depth <= 0:
                return
            print('----' * node.depth + node.name)
            for son in node.sons:
                iter(son, depth - 1)
        if startconcept is None:
            startconcept = self.basetree
        if type(startconcept) == str:
            startconcept = self.basedict[startconcept] # type: ignore
        iter(startconcept, targetdepth) # type: ignore
    
    

if __name__ == "__main__":
    rawdata = RawData(WordnetArg.originWord, WordnetArg.wordnet_depth)
    rawdata.save(DatapathArg.path_rawdata)

