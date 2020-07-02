import argparse
from options import train_opts
from options import model_opts
from collections import OrderedDict

from tqdm import tqdm

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors
import torch
from components import load_pretrained_embedding_from_file
def main(args):

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

    #if not os.path.exists(args.savedir):
        #os.mkdir(args.savedir)

    ## save field and vocabulary 
    #for field in fields:
        #save_field(args.savedir, field)
        #save_vocab(args.savedir, field)

    if args.pretrained_embedding is not None:
        pretrained = load_pretrained_embedding_from_file(args.pretrained_embedding, fields[0][1], 300)
    else:
        pretrained = None
        
    print('save embeddings')
    torch.save(pretrained, '/scratch_global/frejus/embeddings/fasttext.pt')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    train_opts(parser)
    model_opts(parser)
    args = parser.parse_args()
    main(args)
    
