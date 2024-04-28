import argparse

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

import config as cfg
from datautil import *
from modelutil import *
from trainutil import *

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# define arguments, override the defaults from config.py with arguments
args = argparse.ArgumentParser()
args.add_argument('training-file', type=str)
args.add_argument('validation-file', type=str)
args.add_argument("experiment-name", type=str)

args.add_argument('--batch-size', type=int, default=cfg.batch_size)
args.add_argument('--epochs', type=int, default=cfg.epochs)
args.add_argument('--lr', type=float, default=cfg.lr)
args.add_argument("--model-name", type=str, default=cfg.model_name)
args.add_argument("--max-step", type=int, default=-1)
args.add_argument("--max-length", type=int, default=cfg.max_length)
args.add_argument("--max-epoch", type=str, default=cfg.max_epoch)

args = args.parse_args()

# build param dictionary from args
params = vars(args)
params = {k.replace('-', '_'): v for k, v in params.items()}
logger.info(f"Params: {params}")


def main():
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    train_ds = DatasetFromJson(params['training_file'], tokenizer, cfg.max_length)
    val_ds = DatasetFromJson(params['validation_file'], tokenizer, cfg.max_length)
    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], collate_fn=CollateFn(tokenizer, return_raw=False))
    val_dl = DataLoader(val_ds, batch_size=params['batch_size'],
                        collate_fn=CollateFn(tokenizer=tokenizer, return_raw=True))

    model = model_init(params['model_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    checkpoint_dir = f"{cfg.checkpoint_dir}/{params['experiment_name']}"
    history_dir = f"{checkpoint_dir}/history"
    history = fit(model, optimizer, train_dl, val_dl, params, checkpoint_dir, max_step=args.max_step, epoch=0)
    save_history(history, history_dir, save_graph=True)


if __name__ == '__main__':
    main()
