# -*- coding: utf-8 -*-

import argparse
import math
import os
import dill

from collections import OrderedDict

from tqdm import tqdm

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors

import torch
import torch.nn as nn
import torch.optim as optim

from options import train_opts
from options import model_opts
from model import Seq2seqAttn
from components import load_pretrained_embedding_from_file

from apex import amp


class Trainer(object):
    def __init__(
        self, model, criterion, optimizer, scheduler, clip):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.n_updates = 0
        self.accumulation_steps = 50

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, samples, tf_ratio, i):
        #self.model.zero_grad() #gradient accumulation step1
        #self.optimizer.zero_grad() #gradient accumulation step1
        bsz = samples.src.size(1)
        outs = self.model(samples.src, samples.tgt, tf_ratio)
        loss = self.criterion(outs.view(-1, outs.size(2)), samples.tgt.view(-1))

        
        if self.model.training:
            loss = loss / self.accumulation_steps #gradient accumulation step2
            #with amp.scale_loss(loss, self.optimizer) as scale_loss:
                #scale_loss.backward()
            loss.backward()
            
            if (i+1) % self.accumulation_steps == 0: 
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.n_updates += 1
        return loss


def save_model(save_vars, filename):
    model_path = os.path.join(args.savedir, filename)
    torch.save(save_vars, model_path)


def save_vocab(savedir, fields):
    name, field = fields
    save_path = os.path.join(savedir, f"{name}_vocab.txt")
    with open(save_path, 'w') as fout:
        for w in field.vocab.itos:
            fout.write(w + '\n')


def save_field(savedir, fields):
    name, field = fields
    save_path = os.path.join(savedir, f"{name}.field")
    with open(save_path, 'wb') as fout:
        dill.dump(field, fout)
    

def load_fasttext_embeddings(path):
    embeddings = torch.load(path)
    return embeddings


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    device = torch.device('cuda' if args.gpu  else 'cpu')

    # load data and construct vocabulary dictionary
    SRC = data.Field(lower=True)
    TGT = data.Field(lower=True, eos_token='<eos>', fix_length=50)
    fields = [('src', SRC), ('tgt', TGT)]

    print('load train data')
    train_data = data.TabularDataset(
        path=args.train,
        format='tsv',
        fields=fields,
    )
    
    print('load valid data')

    valid_data = data.TabularDataset(
        path=args.valid,
        format='tsv',
        fields=fields,
    )

    print('build vocabularies')
    SRC.build_vocab(train_data, min_freq=args.src_min_freq)
    TGT.build_vocab(train_data, min_freq=args.tgt_min_freq)

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # save field and vocabulary 
    for field in fields:
        save_field(args.savedir, field)
        save_vocab(args.savedir, field)

    if args.pretrained_embedding is not None:
        #pretrained = load_pretrained_embedding_from_file(args.pretrained_embedding, fields[0][1], 300)
        pretrained = load_fasttext_embeddings(args.pretrained_embedding)
    else:
        pretrained = None
        
    # set iterator
    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data), 
        batch_size=args.batch_size,
        sort_within_batch=True,
        sort_key= lambda x: len(x.src),
        repeat=False,
        device=device
    )
    
    
    
    print('load model')
    model = Seq2seqAttn(args, pretrained, fields, device).to(device)
    print(model)
    print('')

    criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi['<pad>'])
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    trainer = Trainer(model, criterion, optimizer, scheduler, args.clip)
   
    opt_level = 'O3'
    #model, optimizer = amp.initialize(model, 
                                    #optimizer,
                                    #keep_batchnorm_fp32=False,
                                    #opt_level=opt_level)
   
   
    epoch = 1
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    best_loss = math.inf

    while epoch < max_epoch and trainer.n_updates < max_update \
        and args.min_lr < trainer.get_lr():

        # training
        with tqdm(train_iter, dynamic_ncols=True) as pbar:
            train_loss = 0.0
            trainer.model.train()
            #trainer.model.zero_grad() #gradient accumulation step1
            ibz = 0
            for samples in pbar:
                bsz = samples.src.size(1)
                ibz += 1 
                loss = trainer.step(samples, args.tf_ratio, ibz)
                train_loss += loss.item()

                # setting of progressbar
                pbar.set_description(f"epoch {str(epoch).zfill(3)}")
                progress_state = OrderedDict(
                    loss=loss.item(),
                    ppl=math.exp(loss.item()),
                    bsz=len(samples),
                    lr=trainer.get_lr(), 
                    clip=args.clip, 
                    num_updates=trainer.n_updates)
                pbar.set_postfix(progress_state)
        train_loss /= len(train_iter)

        print(f"| epoch {str(epoch).zfill(3)} | train ", end="") 
        print(f"| loss {train_loss:.{4}} ", end="")
        print(f"| ppl {math.exp(train_loss):.{4}} ", end="")
        print(f"| lr {trainer.get_lr():.1e} ", end="")
        print(f"| clip {args.clip} ", end="")
        print(f"| num_updates {trainer.n_updates} |")
        
        # validation
        valid_loss = 0.0
        trainer.model.eval()
        for samples in valid_iter:
            bsz = samples.src.size(1)
            loss = trainer.step(samples, tf_ratio=0.0)
            valid_loss += loss.item()

        valid_loss /= len(valid_iter)

        print(f"| epoch {str(epoch).zfill(3)} | valid ", end="") 
        print(f"| loss {valid_loss:.{4}} ", end="")
        print(f"| ppl {math.exp(valid_loss):.{4}} ", end="")
        print(f"| lr {trainer.get_lr():.1e} ", end="")
        print(f"| clip {args.clip} ", end="")
        print(f"| num_updates {trainer.n_updates} |")

        # saving model
        save_vars = {"train_args": args, 
                     "state_dict": model.state_dict()}

        if valid_loss < best_loss:
            best_loss = valid_loss
            save_model(save_vars, 'checkpoint_best.pt')
        save_model(save_vars, "checkpoint_last.pt")

        # update
        trainer.scheduler.step(valid_loss)
        epoch += 1

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    train_opts(parser)
    model_opts(parser)
    args = parser.parse_args()
    main(args)

