import torch
from mycode.utils import evaluate, prepare
from mycode.utils.gaussianDrawer import GaussianDrawer
from mycode.utils.treeunit import NodeUnit
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
        
        #self.draw('animal', 'attr')
        #self.evaluater.calcF1score()
        self.draw(['bird', 'reptile'])
        self.draw('mammal', 'attr')
        self.draw('vertebrate', 'attr')
        self.draw('critter', 'attr')
        self.draw('larva', 'attr')
        self.draw('tiger', 'attr')
        self.draw('vertebrate', 'attr')
        e_animal = self.lookupEmbedding('animal')
        e_chordate = self.lookupEmbedding('chordate')
        e_vertebra = self.lookupEmbedding('vertebrate')
        e_mammal   = self.lookupEmbedding('mammal')
        e_bigcat = self.lookupEmbedding('big_cat')
        e_tiger = self.lookupEmbedding('tiger')
        basetree = self.addLambd(self.model, self.rawdata.basetree, mode = 'homo')
        self.evaluater.calcF1score()
        self.evaluater.findworstlambd()    
        self.evaluater.findworstgap('KL')
        self.evaluater.findworstgap_homo('KL')
        a = 0
    
    def checklambd(self, concept, mode = 'attr'):
        if mode == 'attr':
            attributes  = self.finaldata.finedata.attrdict[concept]|set(self.finaldata.finedata.paredict[concept])
        elif mode == 'homo':
            attributes  = self.finaldata.finedata.homodict[concept]
        else:
            raise Exception('lambd mode should be \'attr\' or \'homo\'')
        attrIndex   = [list(self.finaldata.indexconvert(attributes))]
        lambd       = self.model.scorePos(attrIndex)
        return lambd
    
    def checkgap(self, concept0, conceptN, mode = 'gap'):
        index0  = self.finaldata.indexconvert(concept0)
        indexN = self.finaldata.indexconvert(conceptN)
        if type(indexN) == set:
            indexN  = list(indexN)
        gap     = self.model.evaluate(index0, indexN, mode = mode)
        if type(indexN) == list:
            return gap.tolist()#[round(i, 2) for i in gap.tolist()]
        else:
            return round(gap.item(), 2)

    
    def draw(self, name, partner = 'default'):
        if partner == 'attr':
            self.drawer.drawOneSample([name] + list(self.finaldata.finedata.attrdict[name]) + list(self.finaldata.finedata.paredict[name]), name + '-attr')
        elif partner == 'sons':
            self.drawer.drawOneSample([name] + list(self.finaldata.finedata.rawdata.basedict[name].sons), name + '-son')
        elif partner == 'default':
            self.drawer.drawOneSample(name, 'default')
        else:
            self.drawer.drawOneSample(name, partner)
        
    def lookupEmbedding(self, name):
        index = self.finaldata.indexconvert(name)
        return self.model.lookupEmbedding_whole(index).inv()

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
    TrainArg.usegpu = False
    finaldata    = FinalData(ifloadDictionary=True)
    model      = prepare.prepareModel(finaldata.indexdata.homoDF, ifLoadModel=True)
    drawer     = GaussianDrawer(finaldata, model)
    evaluater = evaluate.Evaluater(finaldata, model)
    test = Tester(finaldata, model, drawer, evaluater)
    test.run()
    #mdrawer.drawSamples()