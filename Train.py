
import os
import json
import torch
import codecs
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from mcode.utils import utils, prepare
from mcode.models import SN2E#TransE, TransH, TransA, TransD, KG2E
from mcode.utils.evaluater import Evaluater
from Tester import Tester
from tensorboardX import SummaryWriter



class Trainer():
    def __init__(self, configs:Config, dataLoader:DataLoader, evaluater:Evaluater, ifLoadModel = False) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.configs    = configs
        self.model      = prepare.prepareModel(configs, ifLoadModel)
        self.optimizer  = prepare.prepareOptimizer(configs, self.model)
    
    def train_one_batch(self, batchData, trainMode):
        self.optimizer.zero_grad()
        loss = self.model(batchData, trainMode)
        loss.backward()
        self.optimizer.step()
        self.model.tailingWorks()
        return loss.item()
    
    def run(self):
        EPOCHS = self.configs.model.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS))
        posMinLoss, negMinLoss = float("inf"), float("inf") 
        bestHR = float("inf")
        
        for epoch in EPOCHS_ITER:
            posSumLoss, negSumLoss = 0, 0
            posCheckLoss, negCheckLoss = 0, 0
            i, j = 0, 0
            for primBatchData in self.dataLoader:
                negLoss = self.train_one_batch(primBatchData, 'negMode')
                negCheckLoss += negLoss
                negSumLoss += negLoss
                j += 1
            if negMinLoss > negSumLoss / j /self.configs.model.Alpha:
                negMinLoss = negSumLoss / j /self.configs.model.Alpha
            self.dataLoader.dataset.transformLoader()

            for defiBatchData in self.dataLoader:
                posLoss = self.train_one_batch(defiBatchData, 'posMode')
                posCheckLoss += posLoss
                posSumLoss += posLoss
                i += 1 
            if posMinLoss > posSumLoss / i:
                posMinLoss = posSumLoss / i
            self.dataLoader.dataset.transformLoader()

            EPOCHS_ITER.set_description("Epoch %d | postive loss : %f, negtive loss : %f, min positive loss: %f, min negtive loss %f" \
                        % (epoch, posSumLoss/i, negSumLoss/j/self.configs.model.Alpha, posMinLoss, negMinLoss))
            '''
            if epoch % self.configs.evalepoch == 0:
                HR = self.evaluater.HREvaluate(self.model)
                #CR = self.evaluater.CREvaluate(self.model)
                EPOCHS_ITER.set_description("Epoch %d | Loss : %d, HR: %d, CR: %d " \
                        % (epoch, minLoss, HR))#, CR))
                if bestHR < HR:
                    bestHR = HR
                    self.model.saveCheckpoint(self.configs.modelPath)
            '''
        self.model.setDefiConceptEmbedding(self.dataLoader.dataset.defiNumConcepts, self.dataLoader.dataset.homoDF)
        self.model.saveCheckpoint(self.configs.modelPath)
        a = 0

if __name__ == "__main__":
    configs = Config()
    mdataset = prepare.prepareDataSet(configs)
    mdataloader = DataLoader(mdataset,
                             batch_size=configs.batchsize,
                             shuffle=configs.shuffle,
                             num_workers=configs.numworkers,
                             drop_last=configs.droplast)
    mevaluater = Evaluater(configs, mdataset)
    mtrainer = Trainer(configs, mdataloader, mevaluater)
    mtrainer.run()

    #mtester = Tester(configs, mdataset, mevaluater, model = mtrainer.model)
    #mtester.run()


                

        
