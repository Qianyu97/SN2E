import torch
from mycode.utils import evaluate, prepare
from mycode.utils.gaussianDrawer import GaussianDrawer
from finaldata import FinalData, RawData
from config import DatapathArg, TrainArg
from mycode.models.SN2E import SN2E
class Tester():
    def __init__(self,
                 finaldata:FinalData, 
                 model:SN2E, 
                 drawer:GaussianDrawer, 
                 evaluater:evaluate.Evaluater) -> None:
        self.finaldata  = finaldata
        self.finedata   = finaldata.finedata
        self.rawdata    = finaldata.finedata.rawdata
        self.model = model
        self.drawer = drawer
        self.evaluater = evaluater
        a = 0
            

    def run(self):
        #self.evaluater.calcF1score()
        #self.drawer.drawOneSample(['tiger'] + list(self.finaldata.finedata.attrdict['tiger']), 'tiger')
        #print('Lambd: %.2f' % self.checklambd('tiger').item())
        #print('Gap: ' + str(self.checkgap('a0', ['a1', 'a2']).tolist()))
        #self.drawpicture('animal')
        #m, v = self.lookupEmbedding(['a0', 'a1'])
        #self.draw('animal', 'attr')
        
        self.draw('animal', 'attr')
        self.draw('mammal', 'attr')
        self.draw('vertebrate', 'attr')
        self.draw('critter', 'attr')
        self.draw('lava', 'attr')
        self.draw('tiger', 'attr')
        self.draw(['vertebrate', 'reptile', 'bird'])
        self.evaluater.findworstlambd()
        a = 0
    
    def checklambd(self, concept, mode = 'attr'):
        if mode == 'attr':
            attributes  = self.finaldata.finedata.attrdict[concept]
        elif mode == 'homo':
            attributes  = self.finaldata.finedata.homodict[concept]
        else:
            raise Exception('lambd mode should be \'attr\' or \'homo\'')
        attrIndex   = [list(self.finaldata.indexconvert(attributes))]
        lambd       = self.model.scorePos_test(attrIndex)
        return lambd
    
    def checkgap(self, concept0, conceptN):
        index0  = self.finaldata.indexconvert(concept0)
        indexN  = list(self.finaldata.indexconvert(conceptN))
        gap     = self.model.scoreNeg_test(index0, indexN)
        return - gap
    
    def draw(self, name, partner = 'default'):
        if partner == 'attr':
            self.drawer.drawOneSample([name] + list(self.finaldata.finedata.attrdict[name]), name + '-attr')
        elif partner == 'sons':
            self.drawer.drawOneSample([name] + list(self.finaldata.finedata.rawdata.basedict[name].sons), name + '-son')
        elif partner == 'default':
            self.drawer.drawOneSample(name, 'default')
        else:
            self.drawer.drawOneSample(name, partner)
        
    def lookupEmbedding(self, name):
        index = self.finaldata.indexconvert(name)
        m, v  = self.model.lookupEmbedding_whole(index)
        return m, v

    def findmostson(self):
        a = sorted(self.rawdata.basedict.keys(), key=lambda x:len(self.rawdata.basedict[x].sons), reverse=True)
        print(*a)

        

if __name__ == "__main__":
    TrainArg.usegpu = False
    finaldata    = FinalData(DatapathArg.path_rawdata, DatapathArg.path_indexdict)
    model      = prepare.prepareModel(finaldata.indexdata.homoDF, ifLoadModel=True)
    drawer     = GaussianDrawer(finaldata, model)
    evaluater = evaluate.Evaluater(finaldata, model)
    test = Tester(finaldata, model, drawer, evaluater)
    test.run()
    #mdrawer.drawSamples()