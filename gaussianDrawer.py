import random
from models.SN2E import SN2E
from finaldata import FinalData

from utils.unit import NodeUnit, AttrUnit, BaseUnit, Indexer_SN2E
from utils.unit import Embedding, EmbeddingOperator
from utils.drawerTool import gaussians_ellipse
from utils.fileTool import clear_jpg


class GaussianDrawer():
    def __init__(self, finaldata:FinalData, indexer:Indexer_SN2E, model:SN2E, pictureDirectory:str=''):
        self.finaldata = finaldata
        self.indexer = indexer
        self.baselist = finaldata.nodeList[:-1]
        self.model = model
        self.pictureDirectory = pictureDirectory
        clear_jpg(self.pictureDirectory)

    
    def drawsomeSamples(self, shownum = 10)->list[NodeUnit]:
        showTargets:list[NodeUnit] = list()
        showTargets = random.sample(self.baselist, shownum)
        for i, sample in enumerate(showTargets):
            self.drawOneConcept(sample.name)
    
    def drawOneConcept(self, name:str):
        concept = self.finaldata.nodeUnitDict[name]
        attrset = concept.attributes
        children = concept.children
        father = concept.father
        if len(attrset) > 0:
            if len(attrset) == 0:
                a = 0
            self.drawConcepts(
                nodename = [concept.name] + [father.name], 
                attrname = [i.name for i in attrset], 
                label = f"{concept.name} attributes")
        if len(children) > 0:
            self.drawConcepts(
                nodename = [concept.name] + [i.name for i in children],
                label = f"{concept.name} children")
        
    @DeprecationWarning
    def drawAttributeSample(self, concept:NodeUnit, attributes:list[BaseUnit], father:NodeUnit, label = 'default'):
        index0 = concept.index
        indexA = [item.index for item in attributes]
        indexF = father.index
        e_0 = self.model.lookupNodeEmbedding(index0, ifdetach=True)
        e_a = self.model.lookupAttrEmbedding(indexA, ifdetach=True)
        e_f = self.model.lookupNodeEmbedding(indexF, ifdetach=True)
        e_u:Embedding = EmbeddingOperator.cat([e_0, e_a, e_f])
        gaussians_ellipse(
            [concept.name] + [i.name for i in attributes] + [father], 
            e_u, facecolor = 'blue',
            saveDirectory = self.pictureDirectory + label + ".jpg")
        
    @DeprecationWarning
    def drawChildrenSample(self, concept:NodeUnit, children:list[BaseUnit], label = 'default'):
        if len(children) == 0:
            return
        index0 = concept.index
        children_idx = [item.index for item in children]
        e_0 = self.model.lookupNodeEmbedding(index0, ifdetach=True)
        e_a = self.model.lookupNodeEmbedding(children_idx, ifdetach=True)
        e_u = EmbeddingOperator.cat([e_0, e_a])
        gaussians_ellipse(
            [concept.name] + [i.name for i in children],
            e_u, facecolor = 'blue',
            saveDirectory = self.pictureDirectory + label + ".jpg")
    
    def drawConcepts(self, nodename:list[str]=[], attrname:list[str]=[], label='default'):
        if type(nodename) == str:
            nodename = [nodename]
        if type(attrname) == str:
            attrname == [attrname]
        indexD = self.indexer.str2num(nodename)
        indexA = self.indexer.str2num(attrname)
        e_d = self.model.lookupNodeEmbedding(indexD)
        e_a = self.model.lookupAttrEmbedding(indexA)
        e_u = EmbeddingOperator.cat([e_d, e_a])
        gaussians_ellipse(
            nodename + attrname,
            e_u, facecolor = 'blue',
            saveDirectory = self.pictureDirectory + label + ".jpg"
            )

if __name__ == "__main__":
    from models import initModule
    from config import PathArg, TestArg, SN2EArg
    DatapathArg = PathArg 
    TestArg  = TestArg
    ModelArg = SN2EArg
    finaldata  = FinalData(DatapathArg["dataDirectory"])
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
    drawer = GaussianDrawer(finaldata, myIndex, model, DatapathArg["pictureDirectory"])
    drawer.drawsomeSamples(3)
    a=0