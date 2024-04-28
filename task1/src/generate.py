import argparse

import jsonlines
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
args.add_argument('evaluation-file', type=str)
args.add_argument('checkpoint', type=str)

args.add_argument('--batch-size', type=int, default=cfg.batch_size)

args = args.parse_args()

# build param dictionary from args
params = vars(args)
params = {k.replace('-', '_'): v for k, v in params.items()}
logger.info(f"Params: {params}")


def generate(model, tokenizer, text, max_length):
    model.eval()

    with torch.no_grad():
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        # calculate loss on valid
        _, logits = step(model, encoding)
        logits = logits[0].transpose(0, 1)

        # returns a list of span
        hypothesis = parse_label_encoding(text, encoding, logits, LABELS)
    return hypothesis


def evaluate(filepath, model, tokenizer, max_length):
    """
    takes a filepath and saves output following the shared task's format and metrics if labels are available
    """
    infile = jsonlines.open(filepath)
    outfile = filepath.replace("jsonl", "hyp.jsonl")
    outfile = open(outfile, 'w', encoding="utf-8")

    for sample in tqdm(infile):
        hypothesis = generate(model, tokenizer, sample['text'], max_length)
        outfile.write(json.dumps({'id': sample['id'], 'labels': hypothesis}, ensure_ascii=False) + "\n")
    infile.close()
    outfile.close()

    print('🎉 output saved to', outfile)


def main():
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    model = model_init(params['model_name'])
    model, _, _, _ = load_checkpoint(model, params['checkpoint'])
    logger.info("🎉 Model loaded successfully!")
    logger.info("Generating predictions...")

    evaluate(params['evaluation_file'], model, tokenizer, cfg.max_length)


if __name__ == '__main__':
    main()
