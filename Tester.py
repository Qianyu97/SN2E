import torch
from utils import initModule
from utils.evaluate import Evaluater
from gaussianDrawer import GaussianDrawer
from utils.treeunit import NodeUnit, AttributeUnit
from finaldata import FinalData
from models.SN2E import SN2E
from config import PathArg, DataloaderArg, TestArg
from config_model import SN2E_Arg
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
        a = self.evaluater.checklambd('Animal')
        b = self.evaluater.checkgap('Bee', ['Animal', 'Car'], 'defi')
        c = self.evaluater.checkgap('Bee', ['has_id Canary', 'has_id Bee'], 'attr')
        self.drawer.drawOneConcept('Bee')
        d = self.evaluater.lookupEmbedding('Animal', type0='defi')
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
    pathArg         = PathArg()
    dataloaderArg  = DataloaderArg()
    modelArg      = SN2E_Arg()
    testArg       = TestArg()

    finaldata   = FinalData(pathArg.dataDirectory, modelArg, ifLoadIndex=True)
    model       = initModule.initModel(modelArg, ifloadmodel=True, modelDir=pathArg.modelDirectory, usegpu=testArg.usegpu, gpunum=testArg.gpunum)
    drawer      = GaussianDrawer(finaldata, model, pathArg.pictureDirectory)
    evaluater   = Evaluater(finaldata, model)
    test = Tester(model, drawer, evaluater)
    test.run()
    #mdrawer.drawSamples()