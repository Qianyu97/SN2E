import torch
from models import initModule
from evaluate import Evaluater
from gaussianDrawer import GaussianDrawer
from utils.unit import NodeUnit, AttrUnit
from finaldata import FinalData
from models.SN2E import SN2E

class Tester():
    def __init__(self,
                 model:SN2E, 
                 drawer:GaussianDrawer, 
                 evaluater:Evaluater) -> None:
        self.finaldata  = finaldata
        self.model = model
        self.drawer = drawer
        self.evaluater = evaluater
        a = 0

    def run(self):
        self.evaluater.calcF1score()
        a = self.evaluater.checkgamma('Animal')
        b = self.evaluater.checkgap('Bee', ['Animal', 'Car'], 'node')
        c = self.evaluater.checkgap('Bee', ['has_id Canary', 'has_id Bee'], 'attr')
        d = self.evaluater.lookupEmbedding('Animal', type0='node')
        #self.evaluater.findworstlambd()    
        #self.evaluater.findworstgap('KL')
        #self.evaluater.findworstgap_homo('KL')
        a = 0
        
    

    def findmostson(self):
        a = sorted(self.rawdata.basedict.keys(), key=lambda x:len(self.rawdata.basedict[x].sons), reverse=True)
        print(*a)

    def addLambd(self, model:SN2E, tree:NodeUnit,mode = 'attr'):
        def iter(node:NodeUnit):
            if mode == 'attr':
                attrIndex   = [list(self.finaldata.indexconvert(node.attributes|node.fathers))]
            elif mode == 'homo':
                attrIndex = [list(self.finaldata.indexconvert(self.finaldata.finedata.homodict[node.name]))]
            node.lambd = round(model.scorePos_test(attrIndex).item(), 2) # type: ignore
            for son in node.sons:
                iter(son)
        iter(tree)
        return tree
    
    def checknegt(self, concept):
        for v in self.finedata.negtdict.values():
            if concept in v:
                print('alert!!')
        print('end')

        
        

if __name__ == "__main__":
    from config import PathArg, TestArg
    from config_model import SN2EArg
    from utils.unit import Indexer_SN2E
    ModelArg = SN2EArg
    finaldata = FinalData(
        data_dir=PathArg["dataDirectory"]
        )
    myIndex = Indexer_SN2E(
        nodeList=finaldata.nodeList,
        attrList=finaldata.attrList,
        load_dir=PathArg["indexDirectory"]
        )
    finaldata.indexConceptUnit(myIndex)
    model = initModule.initModel(
        ModelArg, finaldata.returnDataParams(),
        modelPath=PathArg["modelDirectory"] + ModelArg["name"] + '.ckpt', 
        usegpu=TestArg["usegpu"], 
        gpunum=TestArg["gpunum"]
        )
    evaluater = Evaluater(finaldata, myIndex, model)
    drawer      = GaussianDrawer(finaldata, myIndex, model, PathArg["pictureDirectory"])
    print(f"The f1score is {evaluater.evaluateF1score()}")
    print(f"The auc score is {evaluater.evaluateAUC()}")
    #print(evaluater.checkgap('Cat', ['Feline', 'Mammal']))
    #print(evaluater.checkgap('Cat', ['has_id Bird'], 'attr'))
    #print(evaluater.checkgap('Cat', ['has_id Bee'], 'attr'))
    a = 0
    #mdrawer.drawSamples()

