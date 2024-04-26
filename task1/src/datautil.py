from collections import defaultdict

from ..scorer import task1 as task1scorer

from config import labels as LABELS
import torch
from statistics import mean

from tqdm import tqdm
from pathlib import Path
import jsonlines
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast, DataCollatorWithPadding
import json
from torch.utils.data import Dataset, DataLoader


def parse_text(text, span_objs, tokenizer: PreTrainedTokenizerFast, labels, max_length):
    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Must be a sub-class of PreTrainedTokenizerFast"

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    label_encoding = [
        [
            0,
        ]
        * len(encoding.tokens())
        for _ in labels
    ]

    for span in span_objs:
        label_start, label_end, label = span["start"], span["end"], span["technique"]
        # l is for sequence number
        for seq_num, (token_start, token_end) in enumerate(encoding.offset_mapping):
            if is_inside(token_start, token_end, label_start, label_end):
                label_encoding[labels.index(label)][seq_num] = 1

    return encoding, label_encoding


def is_inside(token_start, token_end, label_start, label_end):
    return token_start >= label_start and token_end <= label_end


def find_consecutive_trues(flags):
    """
    This function takes an array of boolean flags and returns a list of ranges
    of all consecutive true values.

    Args:
        flags: A list of boolean flags.

    Returns:
        A list of tuples, where each tuple represents a range of consecutive
        true values. The tuple contains the starting and ending indices (inclusive)
        of the range.
    """
    ranges = []
    start_idx = None
    for i, flag in enumerate(flags):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            ranges.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        ranges.append((start_idx, len(flags) - 1))
    return ranges


def parse_sample(sample, tokenizer, labels, max_length):
    encoding, label_encoding = parse_text(
        sample["text"], sample["labels"], tokenizer, labels, max_length
    )

    return {"encoding": encoding, 'id': sample['id'], **encoding, "labels": label_encoding}


class DatasetFromJson(Dataset):
    def __init__(self, data_path):
        """
        Args:
          data_path (str): Path to the JSONLines file containing map style data.
        """
        self.data_path = data_path
        self.encodings = []
        self.tensors = []
        self.raw = []
        self._load_data()

    def _load_data(self):
        """
        Loads map style data from the JSONLines file.
        """
        with open(self.data_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                if 'labels' not in data:
                    data['labels'] = []

                sample = parse_sample(data)
                self.encodings.append(sample["encoding"])

                del sample["encoding"], sample["id"]

                self.tensors.append(sample)
                self.raw.append(data)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.raw)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
          idx (int): Index of the sample to retrieve.

        Returns:
          dict: A dictionary containing the map style features.
        """
        return self.tensors[idx], self.encodings[idx], self.raw[idx]


from typing import List, Dict


class CollateFn:
    def __init__(self, tokenizer, return_raw=False):
        self.data_collator = DataCollatorWithPadding(tokenizer, padding=True)
        self.return_raw = return_raw

    def __call__(self, data: List):
        # 'encoding', 'input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'labels'
        encodings = []
        raws = []
        tensors = []

        for tensor, encoding, raw in data:
            encodings.append(encoding)
            raws.append(raw)
            tensors.append(tensor)

        batch = self.data_collator(tensors)

        return {
            'tensors': batch,
            'encodings': encodings,
            'raws': raws
        }


def format_for_output(objs):
    labels_per_par = defaultdict(list)
    for obj in objs:
        par_id = obj["id"]
        labels = obj["labels"]
        labels_per_par[par_id] = task1scorer.process_labels(labels)
    return labels_per_par


def parse_label_encoding(text, encoding, label_encoding, labels):
    label_encoding = label_encoding > 0
    span_objs = []
    word_ids = encoding.word_ids()
    for i, label in enumerate(labels):
        flags = label_encoding[i]
        span_ranges = find_consecutive_trues(flags)
        for start_idx, end_idx in span_ranges:
            start_word_id, end_word_id = word_ids[start_idx], word_ids[end_idx]

            if start_word_id is None or end_word_id is None:
                # skip padding
                continue

            (start_char_idx, _), (_, end_char_idx) = encoding.word_to_chars(
                start_word_id
            ), encoding.word_to_chars(end_word_id)

            span_text = ""
            if text is not None:
                span_text = text[start_char_idx: end_char_idx]

            obj = {
                "technique": label,
                "start": start_char_idx,
                "end": end_char_idx,
                "text": span_text,
            }

            span_objs.append(obj)
    return span_objs


def compute_metrics(batch, logits):
    gold = format_for_output(batch['raws'])
    hypotheses: List[Dict] = []
    logits = logits.transpose(1, 2)
    assert logits.size(1) == len(LABELS), "expects the label in dim=1"

    for i in range(logits.size(0)):
        hypothesis = parse_label_encoding(None, batch['encodings'][i], logits[i], LABELS)
        hypotheses.append({
            "id": batch['raws'][i]['id'],
            "labels": hypothesis
        })

    hypotheses_formatted = format_for_output(hypotheses)

    res_for_screen, f1, f1_per_label = task1scorer.FLC_score_to_string(gold, hypotheses_formatted, per_label=True)

    metrics = {
        'f1': f1,
        **f1_per_label
    }

    return metrics
