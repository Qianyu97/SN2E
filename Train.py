#from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch
import re
import collections
import time
import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from config import PathArg, DataloaderArg, TrainArg
from config_model import SN2E_Arg
from finaldata import FinalData
from utils import initModule
from utils.evaluate import Evaluater
from datasets.attrDataset import attrDataset
from datasets.tripleDataset import tripleDataset
from models.SN2E import SN2E
from Tester import *

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
    def __init__(self, dataLoader:DataLoader, model:SN2E, evaluater:Evaluater, trainArg:TrainArg) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.model      = model
        self.trainArg   = trainArg
        self.optimizer, self.scheduler  = initModule.initOptimizer(self.model, trainArg)
    
    def train_one_batch(self, batchdata):
        loss, posloss, negloss, lambd_max, gap_min = self.model(batchdata)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.batchEndWork()
        return posloss, negloss, lambd_max, gap_min
    
    def run(self):
        print('Info -- : start model training')
        EPOCHS = self.trainArg.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS))
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            worstlambd, worstgap = 0, float("-inf")
            posloss_sum, negloss_sum = 0, 0
            self.model.epochStartWork()
            if epoch == 100:
                a = 0
            for now_depth, depth_range in self.model.depth_range_record.items():
                start_idx, end_idx = depth_range
                self.dataLoader.sampler.indices = list(range(start_idx, end_idx))
                for batchdata in self.dataLoader:
                    posloss, negloss, lambd_max, gap_min = self.train_one_batch(batchdata)
                    posloss_sum += posloss
                    negloss_sum += negloss
                    worstlambd  = min(lambd_max, worstlambd)
                    worstgap    = max(gap_min, worstgap)
            aveposloss = posloss_sum / self.model.defi_num
            avenegloss = negloss_sum / self.model.defi_num/self.trainArg.negtsample_num
            EPOCHS_ITER.set_description("Epoch %d | postive loss : %.2f, negtive loss : %.2f, worst lambd: %.2f, worst gap: %.2f" \
                                % (epoch, aveposloss, avenegloss, worstlambd, worstgap))
            a=0
            
                #if epoch % 100 == 0:
                #    self.model.generateWholeEmbedding()
                #    self.evaluater.calcF1score(chunknum=50)
                #self.scheduler.step() 
        print('Info : finish model training')
        #self.model.generateWholeEmbedding()
        #self.
        print('Info : save model sucessfully')
        #self.evaluater.findworstlambd()
        #
        a = 0

def displayArgs():
    print('\n\n\n\n')
    print(time.strftime("\n%Y-%m-%d %H:%M:%S", time.localtime()))
    showstring = str()
    args = [i for i in dir(displayArg) if not i.startswith('__')]
    args.sort()
    for argname in args:
        if not argname.startswith('__'):
            arg = getattr(displayArg, argname)
            showstring += (argname + ': ' + str(arg))
            showstring += '    '
    print(showstring)

def main():
    pathArg    = PathArg()
    dataloaderArg  = DataloaderArg()
    modelArg       = SN2E_Arg()
    trainArg       = TrainArg()

    finaldata   = FinalData(pathArg.dataDirectory, modelArg)
    dataloader  = DataLoader(
            dataset     = attrDataset(finaldata, trainArg.negtsample_num),              
            batch_size  = dataloaderArg.batchsize,
            num_workers = dataloaderArg.numworkers,
            drop_last   = dataloaderArg.droplast,
            sampler     = RangeSampler(),
            shuffle     = False
            )
    model     = initModule.initModel(modelArg, usegpu=trainArg.usegpu, gpunum=trainArg.gpunum)
    evaluater = None#Evaluater(finaldata, model)
    trainer = Trainer(dataloader, model, evaluater, trainArg=trainArg)
    trainer.run()
    trainer.model.saveCheckpoint(pathArg.model_dir + modelArg.name + '.ckpt')
    #model = prepare.prepareModel(ifLoadmodel=True)
    #finaldata.save(Arg.DatapathArg.path_indexdict, 'dictionary')
    
    # Test
    #drawer = GaussianDrawer(finaldata, model)
    #test = Tester(finaldata, model, drawer, evaluater)
    #test.run()
    

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
    


                

        
