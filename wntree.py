
from nltk.corpus import wordnet as wn

from mycode.utils.treeunit import NodeUnit
from utils import savepickle
from config import DatapathArg, OtherArg

def creat_wntree(origname, maxdepth = 8):
    def iter(node_father:NodeUnit, syns_father, depth):
        if depth < 0:
            return
        for syns_son in syns_father.hyponyms():
            node_son = NodeUnit(syns_son.name())
            node_father.addSon(node_son)
            node_son.addFather(node_father)
            iter(node_son, syns_son, depth - 1)
        return
    origsyns = wn.synsets(origname)[0] # type: ignore
    orignode = NodeUnit(origname)
    iter(orignode, origsyns, maxdepth)
    return orignode

if __name__ == "__main__":
    wntree = creat_wntree(OtherArg.originWord, OtherArg.wordnet_depth)
    savepickle(wntree, DatapathArg.path_wntree)

