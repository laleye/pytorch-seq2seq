import time
import math
import argparse
import torch.nn as nn
from model import TLanguageModel
from data_loader import *
from engine import train, evaluate
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=".", help="location of the data corpus", type=str)
parser.add_argument("--ntokens", default=28785, help="display the number of tokens in the vocabulary", type=int)
parser.add_argument("--emsize", default=200, help="embedding dimension for tokens representation", type=int)
parser.add_argument("--nlayers", default=4, help="the number of encoder layer/bloc in the encoder", type=int)
parser.add_argument("--nhid", default=256, help="he dimension of the feedforward network model in the encoder", type=int)
parser.add_argument("--nhead", default=4, help="the number of heads in the multiheadattention models", type=int)
parser.add_argument("--dropout", default=0.2, help="the dropout percentage", type=float)
parser.add_argument("--epochs", default=30, help="the number of epochs of training", type=int)
parser.add_argument("--bsize", default=20, help="batch size", type=int)
parser.add_argument("--lr", default=3e-5, help="the learning rate for the model", type=float)

args = parser.parse_args()

ntokens = args.ntokens
emsize = args.emsize
nhid = args.nhid
nlayers = args.nlayers
nhead = args.nhead
dropout = args.dropout
epochs = args.epochs
lr = args.lr

# Instanciate the model
model = TLanguageModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
num_train_steps = int(len(train_data) * epochs)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # No warmup
                                            num_training_steps=num_train_steps)

best_val_loss = float("inf")
best_model = None

# Training           
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model, ntokens, optimizer, criterion, epoch, scheduler)
    val_loss = evaluate(model, val_data, ntokens, criterion)
    print('-' * 82)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}       |'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 82)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
