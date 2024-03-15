import jsonlines
import random
import argparse
import warnings
import sys
import logging
from collections import defaultdict
import json

sys.path.append('.')
from scorer.task1 import FLC_score_to_string

random.seed(0)  # to make runs deterministic

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

propaganda_techniques = ['Appeal_to_Values', 'Loaded_Language', 'Consequential_Oversimplification',
                         'Causal_Oversimplification', 'Questioning_the_Reputation', 'Straw_Man', 'Repetition',
                         'Guilt_by_Association', 'Appeal_to_Hypocrisy', 'Conversation_Killer',
                         'False_Dilemma-No_Choice', 'Whataboutism', 'Slogans',
                         'Obfuscation-Vagueness-Confusion',
                         'Name_Calling-Labeling', 'Flag_Waving', 'Doubt',
                         'Appeal_to_Fear-Prejudice', 'Exaggeration-Minimisation', 'Red_Herring',
                         'Appeal_to_Popularity', 'Appeal_to_Authority', 'Appeal_to_Time']

def random_baseline(dev_file, out_fname):
    gold_labels = {}
    pred_labels = {}

    out_f = open(out_fname, 'w', encoding="utf-8")

    with jsonlines.open(dev_file) as gold_f:
        for obj in gold_f:
            doc_id = str(obj["id"])
            per_par_labels = []

            for label in obj["labels"]:
                start = label['start']
                end = label['end']
                per_par_labels.append((label['technique'], [start, end]))

            per_par_labels = sorted(per_par_labels, key=lambda span: span[1][0])
            gold_labels[doc_id] = per_par_labels

            techniques_list = []


            # Most docs in dev and train set has max of 4 labels/document
            rand_no_labels = random.randint(1, 4)

            while len(techniques_list) < rand_no_labels:
                random_technique = propaganda_techniques[random.randint(0, len(propaganda_techniques) - 1)]
                if random_technique not in techniques_list:
                    start = random.randint(0, len(obj['text']))
                    end = random.randint(0, len(obj['text']))
                    pos = [start, end]
                    pos.sort()
                    techniques_list.append((random_technique, tuple(pos)))

            techniques_list = set(techniques_list)

            pred_labels[doc_id] = techniques_list

            wlabels = []
            for item in techniques_list:
                wlabels.append({'technique': item[0], 'start': item[1][0], 'end': item[1][1], "text": obj['text'][item[1][0]:item[1][1]]})

            out_f.write(json.dumps({'id': doc_id, 'labels': wlabels}, ensure_ascii=False) + "\n")

        res_for_screen = FLC_score_to_string(gold_labels, pred_labels, False)

        out_f.close()
        print(res_for_screen)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file_path", "-g", required=True,
                        help="The absolute path to the dev file.", type=str)
    parser.add_argument("--output_file_path", "-o", required=True,
                        help="Path to output file to save run.", type=str)
    args = parser.parse_args()

    random_baseline(args.dev_file_path, args.output_file_path)
