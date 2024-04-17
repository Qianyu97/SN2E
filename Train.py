#from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import re
import collections
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from config import DatapathArg, DataloaderArg, TrainArg, ModelArg, validateArg, displayArg
from finaldata import FinalData, RawData
from mycode.utils import prepare
from mycode.utils.evaluate import Evaluater
from mycode.datasets.attrDataset import attrDataset
from mycode.datasets.tripleDataset import tripleDataset
from mycode.models.SN2E import SN2E
from Tester import *

class Trainer():
    def __init__(self, dataLoader:DataLoader, model:SN2E, evaluater:Evaluater) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.model      = model
        self.optimizer, self.scheduler  = prepare.prepareOptimizer(self.model)
    
    def train_one_batch(self, batchdata):
        loss, posloss, negloss, lambd_max, gap_min = self.model(batchdata)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.tailingWorks()
        return posloss, negloss, lambd_max, gap_min
    
    def run(self):
        print('Info -- : start model training')
        EPOCHS = TrainArg.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS), mininterval=30 ,miniters=5)
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            worstlambd, worstgap = 0, float("inf")
            posloss_sum, negloss_sum = 0, 0
            for batchdata in self.dataLoader:
                posloss, negloss, lambd_max, gap_min = self.train_one_batch(batchdata)
                posloss_sum += posloss
                negloss_sum += negloss
                worstlambd  = max(lambd_max, worstlambd)
                worstgap    = min(gap_min, worstgap)
            aveposloss = posloss_sum / ModelArg.model.num_defi
            avenegloss = negloss_sum / ModelArg.model.num_defi/DataloaderArg.negtsamplenum
            EPOCHS_ITER.set_description("Epoch %d | postive loss : %.2f, negtive loss : %.2f, worst lambd: %.2f, worst gap: %.2f" \
                            % (epoch, aveposloss, avenegloss, worstlambd, worstgap), refresh=False)
            if epoch % 100 == 0:
                self.model.generateWholeEmbedding()
                self.evaluater.calcF1score(chunknum=50)
            self.scheduler.step() 
        print('Info : finish model training')
        self.model.generateWholeEmbedding()
        self.model.saveCheckpoint(ModelArg.path_model)
        #self.evaluater.findworstlambd()
        print('Info : save model sucessfully')
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
    dataloader = DataLoader(
            dataset     = dataset,                
            batch_size  = DataloaderArg.batchsize,
            shuffle     = DataloaderArg.shuffle,
            num_workers = DataloaderArg.numworkers,
            drop_last   = DataloaderArg.droplast,
            #collate_fn  = my_collate
            )
    model      = prepare.prepareModel(finaldata.indexdata.homoDF, ifLoadModel=TrainArg.ifloadmodel)
    evaluater = Evaluater(finaldata, model)
    trainer = Trainer(dataloader, model, evaluater)
    trainer.run()
    finaldata.save(DatapathArg.path_indexdict, 'dictionary')
    
    # Test
    drawer = GaussianDrawer(finaldata, model)
    test = Tester(finaldata, model, drawer, evaluater)
    test.run()
    

if __name__ == "__main__":
    displayArgs()
    finaldata = FinalData(ifloadDictionary=TrainArg.ifloadmodel)
    dataset = attrDataset(finaldata.indexdata) 
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
    
        
    
    


                

        
