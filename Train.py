#from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch
import numpy as np
import re
import collections
import time
import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from finaldata import FinalData
from models import initModule
from evaluate import Evaluater
from datasets.attrDataset import attrDataset
from datasets.tripleDataset import tripleDataset
from models.SN2E import SN2E
from utils.unit import Indexer_SN2E

class RangeSampler(Sampler):
    #depth_range_record example:  {0: [0, 1], 1: [1, 3], 2: [3, 8], 3: [8, 44], 4: [44, 224], 5: [224, -1]}
    def __init__(self, start=0, end=1, shuffle=True):
        self.indices = list(range(start, end))
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



class Trainer():
    def __init__(self, dataLoader:DataLoader, model:SN2E, evaluater:Evaluater, dataset:attrDataset , **trainArg) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.model      = model
        self.epochs:int   = trainArg['epochs']
        self.optimizer, self.scheduler  = initModule.initOptimizer(self.model, **trainArg)
        self.dataset = dataset
    
    def train_one_batch(self, batchdata):
        loss, posloss, negloss, gamma_max, gap_min = self.model(batchdata)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.batchEndWork()
        return posloss, negloss, gamma_max, gap_min

    def run(self):
        pass

    def run_deprecated(self):
        print('Info -- : start model training')
        EPOCHS = self.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS))
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            worstgamma, worstgap = 0, float("-inf")
            posloss_sum, negloss_sum = 0, 0
            self.model.epochStartWork()
            for now_depth, depth_range in self.model.depthRangeRecord.items():
                start_idx, end_idx = depth_range
                self.dataLoader.sampler.indices = list(range(start_idx, end_idx))
                for batchdata in self.dataLoader:
                    posloss, negloss, gamma_max, gap_min = self.train_one_batch(batchdata)
                    posloss_sum += posloss
                    negloss_sum += negloss
                    worstgamma  = min(gamma_max, worstgamma)
                    worstgap    = max(gap_min, worstgap)
            aveposloss = posloss_sum / self.model.node_num
            avenegloss = negloss_sum / self.model.node_num/self.trainArg.negtsample_num
            EPOCHS_ITER.set_description("Epoch %d | postive loss : %.2f, negtive loss : %.2f, worst gamma: %.2f, worst gap: %.2f" \
                                % (epoch, aveposloss, avenegloss, worstgamma, worstgap))
        print('Info : finish model training')
        a = 0
        

def displayArgs(DataloaderArg:dict, TrainArg:dict, ModelArg:dict):
    print('Current training arguments:')
    for key, value in DataloaderArg.items():
        print(f'  {key} : {value}')
    for key, value in TrainArg.items():
        print(f'  {key} : {value}')
    for key, value in ModelArg.items():
        print(f'  {key} : {value}')
    print()

def main():
    from config import PathArg, DataloaderArg, TrainArg
    from config_model import SN2EArg
    displayArgs(DataloaderArg, TrainArg, SN2EArg)
    ModelArg = SN2EArg
    finaldata = FinalData(
        data_dir=PathArg['dataDirectory']
        )
    myIndex = Indexer_SN2E(
        nodeList=finaldata.nodeList,
        attrList=finaldata.attrList
        )
    finaldata.indexConceptUnit(myIndex)
    dataset = attrDataset(
        myIndex.str2num(finaldata.nodeList), 
        myIndex.str2num_DataFrame(finaldata.attrDF),
        myIndex.str2num_DataFrame(finaldata.upperDF),
        myIndex.str2num(finaldata.negtDict),
        TrainArg['negtsample_num']
        )
    dataloader = DataLoader(
        dataset     = dataset,              
        batch_size  = DataloaderArg["batchsize"],
        num_workers = DataloaderArg["numworkers"],
        drop_last   = DataloaderArg["droplast"],
        shuffle     = DataloaderArg["shuffle"]
    )
    model:SN2E = initModule.initModel(
        ModelArg, finaldata.returnDataParams(),
        usegpu=TrainArg["usegpu"], 
        gpunum=TrainArg["gpunum"]
        )
    optimizer, scheduler  = initModule.initOptimizer(model, **TrainArg)
    
    print('Info -- : start model training')
    EPOCHS_ITER = tqdm(range(TrainArg["epochs"]), miniters=10, mininterval=10)
    for epoch in EPOCHS_ITER:
        gammaList   = []
        negtDistList = []
        model.epochStartWork()
        model.batchStartWork()
        for now_depth, depth_range in model.depthRangeRecord.items():
            range_st, range_ed = depth_range
            now_range = list(range(range_st, range_ed))
            indexA = dataset.attrArray[range_st:range_ed]
            indexF = dataset.upperArray[range_st:range_ed]
            gamma = model.scorePos(indexA, indexF)
            gammaList.append(gamma)
        gammaTensor = torch.cat(gammaList)
        gammaTensor_detached = gammaTensor.detach()
        for batchdata in dataloader:
            index0 , indexN = batchdata
            negtDist = model.scoreNeg(index0, indexN)
            negtDistList.append(negtDist)
        negTensor = torch.cat(negtDistList)
        negTensor_detached = negTensor.detach()
        posloss = torch.where(gammaTensor<-1, gammaTensor, gammaTensor_detached).sum().neg()
        negloss = torch.where(negTensor>-0.2, negTensor, negTensor_detached).sum()
        loss = posloss + negloss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            EPOCHS_ITER.set_description("Epoch %d | postive loss : %.2f, negtive loss : %.2f, worst gamma: %.2f, worst gap: %.2f" \
                                % (epoch, gammaTensor_detached.mean().item(), negTensor_detached.mean().item(), 
                                        gammaTensor_detached.min().item(), negTensor_detached.max().item()))
        
    print('Info : finish model training')
    model.saveCheckpoint(PathArg["modelDirectory"] + ModelArg["name"] + '.ckpt')
    print('Info : save model sucessfully')
    myIndex.saveIndex(PathArg["indexDirectory"])
    print('Info : save data index sucessfully')

    evaluater = Evaluater(finaldata, myIndex, model)
    print(f"The f1score is {evaluater.evaluateF1score()}")
    print(f"The auc score is {evaluater.evaluateAUC()}")
    evaluater.findWorstGamma()
    #d = evaluater.checkgamma('Cat')
    #evaluater.findWorstNegt(negTensor_detached, indexN)
    #print(evaluater.checkgap('Cat', ['has_id Bird'], 'attr'))
    #print(evaluater.checkgap('Cat', ['has_id Bee'], 'attr'))
    a=0

if __name__ == "__main__":
    main()
    #displayArgs()
    
    
        
'''
if False:
        print('the validation begin')
        for paramter in validateArg.candidate: 
            print('set ' + validateArg.name + ' with ' + str(paramter) \
                  + '  ' + '-' * 50)
            setattr(validateArg.field, validateArg.name, paramter) # type: ignore
            main()
            print('\n\n\n')
    else:
        if False:
            lprofiler = LineProfiler(SN2E.calcLambda)
            lprofiler.run('main()')
            lprofiler.print_stats()
            lprofiler.dump_stats(DatapathArg.path_profiler)
        else:
            main()
'''
    


                

        
