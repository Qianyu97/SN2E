#from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatapathArg, DataloaderArg, TrainArg, ModelArg, validateArg, displayArg
from finaldata import FinalData, RawData
from mycode.utils import prepare
from mycode.utils.evaluate import Evaluater
from mycode.datasets.attrDataset import attrDataset
from mycode.datasets.tripleDataset import tripleDataset
from mycode.models.SN2E import SN2E

class Trainer():
    def __init__(self, dataLoader:DataLoader, model:SN2E, evaluater:Evaluater) -> None:
        self.dataLoader = dataLoader
        self.evaluater  = evaluater
        self.model      = model
        self.optimizer, self.scheduler  = prepare.prepareOptimizer(self.model)
    
    def train_one_batch(self, batchdata):
        loss, posloss, negloss, maxlambd, mingap = self.model(batchdata)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.tailingWorks()
        return posloss, negloss, maxlambd, mingap
    
    def run(self):
        print('Info -- : start model training')
        EPOCHS = TrainArg.epochs
        EPOCHS_ITER = tqdm(range(EPOCHS), miniters=100)
        worstlambd, worstgap = float("inf"), float("inf") 
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            sumposloss, sumnegloss = 0, 0
            for batchdata in self.dataLoader:
                posloss, negloss, maxlambd, mingap = self.train_one_batch(batchdata)
                sumposloss  += posloss
                sumnegloss  += negloss
                worstlambd = max(maxlambd, worstlambd)
                worstgap   = min(mingap, worstgap)
            aveposloss = sumposloss / ModelArg.model.num_defi
            avenegloss = sumnegloss / ModelArg.model.num_full
            EPOCHS_ITER.set_description("Epoch %d | postive loss : %.2f, negtive loss : %.2f, worst lambd: %.2f, worst gap: %.2f" \
                        % (epoch, aveposloss, avenegloss, worstlambd, worstgap))
            self.scheduler.step()
        print('Info : finish model training')
        self.model.saveCheckpoint(ModelArg.path_model)
        print('Info : save model sucessfully')
        a = 0

def main():
    dataloader = DataLoader(
            dataset     = dataset,                
            batch_size  = DataloaderArg.batchsize,
            shuffle     = DataloaderArg.shuffle,
            num_workers = DataloaderArg.numworkers,
            drop_last   = DataloaderArg.droplast,
            )
    model      = prepare.prepareModel(finaldata.indexdata.homoDF)
    evaluater = Evaluater(finaldata, model)
    trainer = Trainer(dataloader, model, evaluater)
    trainer.run()
    finaldata.save(DatapathArg.path_indexdict, 'dictionary')

def displayArgs():
    showstring = str()
    args = [i for i in dir(displayArg) if not i.startswith('__')]
    args.sort()
    for argname in args:
        if not argname.startswith('__'):
            arg = getattr(displayArg, argname)
            showstring += (argname + ': ' + str(arg))
            showstring += '    '
    print(showstring)

if __name__ == "__main__":
    displayArgs()
    VALIDATE = True
    finaldata = FinalData(DatapathArg.path_rawdata)
    dataset = attrDataset(finaldata.indexdata) 
    if VALIDATE:
        print('the validation begin')
        for paramter in validateArg.candidate: 
            print('set ' + validateArg.name + ' with ' + str(paramter) \
                  + '  ----------------------------------------------------------------------')
            setattr(validateArg.field, validateArg.name, paramter) # type: ignore
            main()
    else:
        if TrainArg.timemeasure:
            lprofiler = LineProfiler(Trainer.run)
            lprofiler.run('main')
            lprofiler.print_stats()
            lprofiler.dump_stats(DatapathArg.path_profiler)
        else:
            main()
    
        
    
    


                

        
