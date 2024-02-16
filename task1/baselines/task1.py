import jsonlines
import random
import argparse
import warnings
import sys

sys.path.append('.')
from scorer.task1 import evaluate

random.seed(42)  # to make runs deterministic


def random_baseline_1B(persuasion_techniques_file, dev_file, out_fname, subtask):
    gold_labels = {}
    pred_labels = {}

    with open(persuasion_techniques_file, "r") as f:
        techniques = [line.strip() for line in f.readlines()]

    out_f = open(out_fname, 'w', encoding="utf-8")
    out_f.write("id\tlabels\n")

    with jsonlines.open(dev_file) as gold_f:
        for obj in gold_f:
            doc_id = str(obj["id"])
            gold_labels[doc_id] = obj["labels"]

            techniques_list = []

            # Most docs in dev ad train set has max of 5 labels/document
            rand_no_labels = random.randint(1, 5)

            while len(techniques_list) < rand_no_labels:
                random_technique = techniques[random.randint(0, len(techniques) - 1)]
                if random_technique not in techniques_list:
                    techniques_list.append(random_technique)

            techniques_list = set(techniques_list)

            if "no_technique" in techniques_list and len(techniques_list) > 1:
                techniques_list.remove("no_technique")

            pred_labels[doc_id] = techniques_list

            out_f.write(doc_id + "\t" + ",".join(techniques_list) + "\n")

        micro_f1, macro_f1 = evaluate(pred_labels, gold_labels, subtask, techniques)

        out_f.close()
        print("micro-F1={:.4f}\tmacro-F1={:.4f}".format(micro_f1, macro_f1))


def random_baseline_1A(dev_file, out_fname, subtask):
    gold_labels = {}
    pred_labels = {}

    out_f = open(out_fname, 'w', encoding="utf-8")
    out_f.write("id\tlabel\n")

    with jsonlines.open(dev_file) as gold_f:
        for obj in gold_f:
            doc_id = str(obj["id"])
            gold_labels[doc_id] = obj["label"]

            labels_list = ["true", "false"]

            preds = random.choice(labels_list)
            pred_labels[doc_id] = preds


            out_f.write(doc_id + "\t" + preds + "\n")

        print(gold_labels)
        micro_f1, macro_f1 = evaluate(pred_labels, gold_labels, subtask)

        out_f.close()
        print("micro-F1={:.4f}\tmacro-F1={:.4f}".format(micro_f1, macro_f1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file_path", "-g", required=True,
                        help="The absolute path to the dev file.", type=str)
    parser.add_argument("--techniques_file_path", "-c", required=False,
                        help="In case of subtask 1B, the absolute path to the file containing all "
                             "possible persuasion techniques", type=str)
    parser.add_argument("--output_file_path", "-o", required=True,
                        help="Path to output file to save run.", type=str)
    parser.add_argument("--subtask", "-s", required=True, choices=['1A', '1B'],
                        help="The subtask for which we are running baseline format: 1A, 1B", type=str)
    args = parser.parse_args()

    subtask = args.subtask

    if subtask == "1A":
        random_baseline_1A(args.dev_file_path, args.output_file_path, subtask)

    if subtask == "1B":
        random_baseline_1B(args.techniques_file_path, args.dev_file_path, args.output_file_path, subtask)
