from mcode import datasets
import torch
from config import Config
from mcode.utils import utils, prepare, evaluater
from mcode.utils.gaussianDrawer import GaussianDrawer
class Tester():
    def __init__(self, configs, dataset, model, drawer:GaussianDrawer, evaluater = None, ifLoadModel = True) -> None:
        self.configs = configs
        self.dataset = dataset
        self.model = model
        self.drawer = drawer
        self.evaluater = evaluater
            

    def run(self):
        #self.evaluater.Hf1Evaluate()
        self.drawer.drawSamples()
        a = 0

if __name__ == "__main__":
    configs     = Config()
    mDataset    = prepare.prepareDataSet(configs)
    mModel      = prepare.prepareModel(configs, ifLoadModel=True)
    mDrawer     = GaussianDrawer(configs, mDataset, mModel)
    mEvaluater = evaluater.Evaluater(configs, mDataset)
    mtest = Tester(configs, mDataset, mModel, mDrawer, mEvaluater)
    mtest.run()
    #mdrawer.drawSamples()