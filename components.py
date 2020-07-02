import logging
import torch.nn as nn
import torch, fasttext
logger = logging.getLogger(__name__)



def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    #nn.init.constant_(m.weight[padding_idx], 0)
    return m

def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim, debug='True'):
    num_embeddings = len(dictionary.vocab.itos)
    #padding_idx = PAD #dictionary.pad()
    if debug:
        print(f"Embedding({num_embeddings}, {embed_dim})")
    #embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
    embed_tokens = Embedding(num_embeddings, embed_dim)
    embed_dict = get_fasttext_embedding(embed_path) #get_embedding(embed_path)
    #print_embed_overlap(embed_dict, dictionary)
    
    return load_embedding(embed_dict, dictionary, embed_tokens)

def load_embedding(embed_dict, vocab, embedding):
    oov = 0
    for idx in range(len(vocab.vocab.itos)):
        token = vocab.vocab.itos[idx]
        #print(f"{idx}/{len(vocab.vocab.itos)}")
        if token in embed_dict:
            embedding.weight.data[idx] = torch.from_numpy(embed_dict[token])
        else:
            oov+=1
            #print(oov, token)
    print('len of Out of vocabulary: {}'.format(oov))
    return embedding

def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    logger.info("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))

def get_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.
    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.
    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def get_fasttext_embedding(embed_path):
    
     #vecs =  fasttext.load_model(fname)
     
     return fasttext.load_model(embed_path)
