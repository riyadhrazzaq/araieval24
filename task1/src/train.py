import argparse

import torch
from transformers import BertTokenizerFast

import config as cfg
from datautil import *

args = argparse.ArgumentParser()
args.add_argument('training-file', type=str)
args.add_argument('validation-file', type=str)
args.add_argument("experiment-name", type=str)

args.add_argument('--batch_size', type=int, default=cfg.batch_size)
args.add_argument('--epochs', type=int, default=cfg.epochs)
args.add_argument('--lr', type=float, default=cfg.lr)

args = args.parse_args()

# build param dictionary from args
params = vars(args)


def main():
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    train_ds = DatasetFromJson(args.training_file)
    val_ds = DatasetFromJson(args.validation_file)
    train_dl = DataLoader(train_ds, batch_size=3, collate_fn=CollateFn(tokenizer, return_raw=False))
    val_dl = DataLoader(val_ds, batch_size=3, collate_fn=CollateFn(return_raw=True))




if __name__ == '__main__':
    main()
