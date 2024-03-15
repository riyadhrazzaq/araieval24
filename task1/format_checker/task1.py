import argparse
import re
import logging
import json
from collections import defaultdict


"""
This script checks whether the results format for task-1 is correct. 
It also provides some warnings about possible errors.

The correct format of the task-1 results file is the following:
{"id": "", "labels": []}

where id is the ID of the data 
and label contains the predicted technique, text, start and end information of the given text.
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

propaganda_techniques = ['Appeal_to_Values', 'Loaded_Language', 'Consequential_Oversimplification',
                         'Causal_Oversimplification', 'Questioning_the_Reputation', 'Straw_Man', 'Repetition',
                         'Guilt_by_Association', 'Appeal_to_Hypocrisy', 'Conversation_Killer',
                         'False_Dilemma-No_Choice', 'Whataboutism', 'Slogans',
                         'Obfuscation-Vagueness-Confusion',
                         'Name_Calling-Labeling', 'Flag_Waving', 'Doubt',
                         'Appeal_to_Fear-Prejudice', 'Exaggeration-Minimisation', 'Red_Herring',
                         'Appeal_to_Popularity', 'Appeal_to_Authority', 'Appeal_to_Time']

def check_format(file_path, gold_data):
    logging.info("Checking format of prediction file...")

    lines = open(file_path, encoding='utf-8').read().strip().split("\n")
    for i, line in enumerate(lines):
        js_obj = json.loads(line)
        if 'id' not in js_obj:
            logging.error("id not present in line {}".format(i))
            return False
        if 'labels' not in js_obj:
            logging.error("labels not present in line {}".format(i))
            return False
        if js_obj['id'] not in gold_data:
            logging.error("Unknown id in line {}".format(i))
            return False
        if len(js_obj['labels']) != 0:
            for label_dict in js_obj['labels']:
                if label_dict['technique'] not in propaganda_techniques:
                    logging.error("Unknown technique '{}' in line {}".format(label_dict['technique'], i))
                    return False
    if len(lines) != len(gold_data):
        logging.error("Mismatch in total number of predicted samples")
        return False
    logging.info("File passed format checker!")
    return True

def load_json_as_list(fname):
    labels_per_par = defaultdict(list)

    with open(fname, 'r', encoding="utf-8") as inf:
        for i,line in enumerate(inf):
            jobj = json.loads(line)
            par_id = jobj['id']

            labels = jobj['labels']

            per_par_labels = []

            for label in labels:
                start = label['start']
                end = label['end']

                per_par_labels.append((label['technique'], [start, end]))

            per_par_labels = sorted(per_par_labels, key=lambda span: span[1][0])

            labels_per_par[par_id] = per_par_labels

    return labels_per_par


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", "-g", required=True, type=str, help="The absolute path to gold file")
    parser.add_argument("--pred-files-path", "-p", required=True, type=str, nargs='+',
                        help="The absolute paths to the files you want to check.")
    args = parser.parse_args()
    gold_data = load_json_as_list(args.gold_file)

    for pred_file_path in args.pred_files_path:
        logging.info(f"Task 1: Checking file: {pred_file_path}")
        check_format(pred_file_path, gold_data)

