#from tensorboardX import SummaryWriter
from line_profiler import LineProfiler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import re
import collections
from torch._six import string_classes

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
        EPOCHS_ITER = tqdm(range(EPOCHS), mininterval=60, miniters=10)
        
        bestHR = float("inf")
        for epoch in EPOCHS_ITER:
            worstlambd, worstgap = 0, float("inf") 
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
                            % (epoch, aveposloss, avenegloss, worstlambd, worstgap), refresh=None)
            self.scheduler.step()
        print('Info : finish model training')
        self.model.saveCheckpoint(ModelArg.path_model)
        print('Info : save model sucessfully')
        a = 0

def my_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return my_collate([torch.as_tensor(b) for b in batch if b.any()])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch) 
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        a = [my_collate(samples) for samples in transposed]
        return a

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def main():
    dataloader = DataLoader(
            dataset     = dataset,                
            batch_size  = DataloaderArg.batchsize,
            shuffle     = DataloaderArg.shuffle,
            num_workers = DataloaderArg.numworkers,
            drop_last   = DataloaderArg.droplast,
            #collate_fn  = my_collate
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
    VALIDATE = False
    finaldata = FinalData(DatapathArg.path_rawdata)
    dataset = attrDataset(finaldata.indexdata) 
    if VALIDATE:
        print('the validation begin')
        for paramter in validateArg.candidate: 
            print('set ' + validateArg.name + ' with ' + str(paramter) \
                  + '  ' + '-' * 50)
            setattr(validateArg.field, validateArg.name, paramter) # type: ignore
            main()
            print('\n\n\n')
    else:
        if displayArg.timemeasure:
            lprofiler = LineProfiler(SN2E.forward)
            lprofiler.run('main()')
            lprofiler.print_stats()
            lprofiler.dump_stats(DatapathArg.path_profiler)
        else:
            main()
    
        
    
    


                

        
