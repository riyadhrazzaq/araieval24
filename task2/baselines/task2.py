import jsonlines
import random
import argparse
import warnings
import sys

sys.path.append('.')
from scorer.task2 import evaluate

random.seed(42)  # to make runs deterministic


def random_baseline(dev_file, out_fname, subtask):
    gold_labels = {}
    pred_labels = {}

    out_f = open(out_fname, 'w', encoding="utf-8")
    out_f.write("id\tlabel\n")

    with jsonlines.open(dev_file) as gold_f:
        for obj in gold_f:
            doc_id = str(obj["id"])
            gold_labels[doc_id] = obj["label"]

            if subtask == "2A": labels_list = ["disinfo", "no-disinfo"]
            else: labels_list = ["OFF", "Rumor", "SPAM", "HS"]

            preds = random.choice(labels_list)
            pred_labels[doc_id] = preds

            out_f.write(doc_id + "\t" + preds + "\n")

        micro_f1, macro_f1 = evaluate(pred_labels, gold_labels)

        out_f.close()

        print("micro-F1={:.4f}\tmacro-F1={:.4f}".format(micro_f1, macro_f1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file_path", "-g", required=True,
                        help="The absolute path to the dev file.", type=str)
    parser.add_argument("--output_file_path", "-o", required=True,
                        help="Path to output file to save run.", type=str)
    parser.add_argument("--subtask", "-s", required=True, choices=['2A', '2B'],
                        help="The subtask for which we are running baseline for: 2A, 2B", type=str)
    args = parser.parse_args()

    random_baseline(args.dev_file_path, args.output_file_path, args.subtask)
