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
        loss, loss_pos, loss_neg, delta_max, gap_min = self.model(batchdata)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.batchEndWork()
        return loss_pos, loss_neg, delta_max, gap_min

    def run(self):
        pass

    def run_deprecated(self):
        print('Info -- : start model training')
        EPOCHS = self.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS))
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            worstdelta, worstgap = 0, float("-inf")
            loss_pos_sum, loss_neg_sum = 0, 0
            self.model.epochStartWork()
            for now_depth, depth_range in self.model.depthRangeRecord.items():
                start_idx, end_idx = depth_range
                self.dataLoader.sampler.indices = list(range(start_idx, end_idx))
                for batchdata in self.dataLoader:
                    loss_pos, loss_neg, delta_max, gap_min = self.train_one_batch(batchdata)
                    loss_pos_sum += loss_pos
                    loss_neg_sum += loss_neg
                    worstdelta  = min(delta_max, worstdelta)
                    worstgap    = max(gap_min, worstgap)
            aveloss_pos = loss_pos_sum / self.model.node_num
            aveloss_neg = loss_neg_sum / self.model.node_num/self.trainArg.negtsample_num
            EPOCHS_ITER.set_description("Epoch %d | postive loss : %.2f, negtive loss : %.2f, worst delta: %.2f, worst gap: %.2f" \
                                % (epoch, aveloss_pos, aveloss_neg, worstdelta, worstgap))
        print('Info : finish model training')
        a = 0
        

def displayArgs(DataloaderArg:dict, TrainArg:dict, ModelArg:dict, DataArg:dict):
    print('Current training arguments:')
    for key, value in DataArg.items():
        print(f'  {key} : {value}')
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
    import numpy as np
    doEvalHyperParams = True
    evalHP_name = 'logv_min'
    evalHP_range = list(range(-16,-3, 1))#np.linspace(-100, -500, 5).tolist()
    PR_AUC_inherited_record = []
    PR_AUC_node_record = []
    ModelArg = SN2EArg
    finaldata = FinalData(
        data_dir=PathArg['dataDirectory']
        )
    myIndex = Indexer_SN2E(
        nodeList=finaldata.nodeList,
        attrList=finaldata.attrList
        )
    DataArg = finaldata.returnDataParams()
    displayArgs(DataloaderArg, TrainArg, SN2EArg, DataArg)
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
    if not doEvalHyperParams:
        evalHP_range = [ModelArg[evalHP_name]]
    for hp in evalHP_range:
        ModelArg[evalHP_name] = hp
        #ModelArg['entail_obj'] = ModelArg['dim']/8
        model:SN2E = initModule.initModel(
            ModelArg, DataArg,
            usegpu=TrainArg["usegpu"], 
            gpunum=TrainArg["gpunum"]
            )
        optimizer, scheduler  = initModule.initOptimizer(model, **TrainArg)
        print('Info -- : start model training')
        EPOCHS_ITER = tqdm(range(TrainArg["epochs"]), miniters=100, mininterval=100)
        for epoch in EPOCHS_ITER:
            record_delta   = []
            record_entail = []
            record_neg = []
            model.epochStartWork()
            model.batchStartWork()
            for now_depth, depth_range in model.depthRangeRecord.items():
                range_st, range_ed = depth_range
                index0 = dataset.nodeList_idx[range_st:range_ed]
                indexA = dataset.attrArray[range_st:range_ed]
                indexF = dataset.upperArray[range_st:range_ed]
                delta  = model.scorePos(indexA, indexF)
                entail = model.scoreEntail(index0, indexF, type2='node')
                record_delta.append(delta)
                record_entail.append(entail)
            deltaTensor = torch.cat(record_delta)
            deltaTensor_detached = deltaTensor.detach()
            entailTensor = torch.cat(record_entail)
            entailTensor_detached = entailTensor.detach()
            
            for batchdata in dataloader:
                index0 , indexN = batchdata
                negtDist = model.scoreNeg(index0, indexN)
                record_neg.append(negtDist)
            negTensor = torch.cat(record_neg)
            negTensor_detached = negTensor.detach()

            loss_pos = torch.where(deltaTensor  < ModelArg["delta_obj"] , deltaTensor  , deltaTensor_detached   ).sum().neg()
            loss_ent = torch.where(entailTensor < ModelArg["entail_obj"], entailTensor , entailTensor_detached  ).sum().neg()
            loss_neg = torch.where(negTensor    > ModelArg["h_obj"]     , negTensor    , negTensor_detached     ).sum()
            loss = loss_pos + loss_neg #+ loss_ent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 1000 == 0:
                EPOCHS_ITER.set_description("Epoch %d | postive loss : %.2f, entail loss : %.2f, negtive loss : %.2f," \
                                    % (epoch, deltaTensor_detached.mean().item(), entailTensor_detached.mean().item(), negTensor_detached.mean().item()))
        evaluater = Evaluater(finaldata, myIndex, model)
        f1score_intrinsic, best_threshold_intrinsic, pr_auc_intrinsic, roc_auc_intrinsic = evaluater.evaluateF1score(threshold=np.linspace(-5, 5, 21), evalmode='intrinsic')
        f1score_inherited, best_threshold_inherited, pr_auc_inherited, roc_auc_inherited = evaluater.evaluateF1score(threshold=np.linspace(-5, 5, 21), evalmode='inherited')
        f1score_node, best_threshold_node, pr_auc_node, roc_auc_node = evaluater.evaluateF1score(threshold=np.linspace(-5, 5, 21), evalmode='node')
        print(f"Evaluating hyperparameter {evalHP_name} = {hp}")
        #print(f"The f1score for attributes (intrinsic) is {f1score_intrinsic:.3f}, its threshold is {best_threshold_intrinsic}")
        print(f"The PR-AUC for attributes (intrinsic) is {pr_auc_intrinsic:.3f}")#, the ROC-AUC is {roc_auc_intrinsic:.3f}")
        #print(f"The f1score for attributes (inherited) is {f1score_inherited:.3f}, its threshold is {best_threshold_inherited}")
        print(f"The PR-AUC for attributes (inherited) is {pr_auc_inherited:.3f}")#, the ROC-AUC is {roc_auc_inherited:.3f}")
        #print(f"The f1score for nodes is {f1score_node:.3f}, its threshold is {best_threshold_node}")
        print(f"The PR-AUC for nodes is {pr_auc_node:.3f}")#, the ROC-AUC is {roc_auc_node:.3f}")
        print("\n\n")
        PR_AUC_inherited_record.append(pr_auc_inherited)
        PR_AUC_node_record.append(pr_auc_node)
    a = np.array(PR_AUC_inherited_record)
    print('Info : finish model training')
    model.saveCheckpoint(PathArg["modelDirectory"] + ModelArg["name"] + '.ckpt')
    print('Info : save model sucessfully')
    myIndex.saveIndex(PathArg["indexDirectory"])
    print('Info : save data index sucessfully')
    print(f"Hyperparameter evaluation results for {evalHP_name}:")
    print([PR_AUC_inherited_record])
    print(PR_AUC_node_record)
    print(f"The PR-AUC for attributes (inherited) is {[i.item() for i in PR_AUC_inherited_record]}")
    print(f"The PR-AUC for nodes is {[i.item() for i in PR_AUC_node_record]}")
    #print(f"{PR_AUC_node_record:.2f}")
        #print(f"The auc score is {evaluater.evaluateAUC()}")
    #evaluater.findWorstGamma()
    #d = evaluater.checkdelta('Cat')
    #evaluater.findWorstNegt(negTensor_detached, indexN)
    #print(evaluater.checkgap('Cat', ['has_id Bird'], 'attr'))
    #print(evaluater.checkgap('Cat', ['has_id Bee'], 'attr'))
    a=0
    b
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
    


                

        
