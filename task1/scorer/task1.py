import json
from collections import defaultdict
import sys
import logging
import os
import argparse

import sys

sys.path.append(".")

from format_checker.task1 import check_format


logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

propaganda_techniques = [
    "Appeal_to_Values",
    "Loaded_Language",
    "Consequential_Oversimplification",
    "Causal_Oversimplification",
    "Questioning_the_Reputation",
    "Straw_Man",
    "Repetition",
    "Guilt_by_Association",
    "Appeal_to_Hypocrisy",
    "Conversation_Killer",
    "False_Dilemma-No_Choice",
    "Whataboutism",
    "Slogans",
    "Obfuscation-Vagueness-Confusion",
    "Name_Calling-Labeling",
    "Flag_Waving",
    "Doubt",
    "Appeal_to_Fear-Prejudice",
    "Exaggeration-Minimisation",
    "Red_Herring",
    "Appeal_to_Popularity",
    "Appeal_to_Authority",
    "Appeal_to_Time",
]


def process_labels(labels):
    per_par_labels = []

    for label in labels:
        start = label["start"]
        end = label["end"]

        per_par_labels.append((label["technique"], [start, end]))

    per_par_labels = sort_spans(per_par_labels)
    return per_par_labels


def load_json_as_list(fname):
    labels_per_par = defaultdict(list)

    with open(fname, "r", encoding="utf-8") as inf:
        for i, line in enumerate(inf):
            jobj = json.loads(line)
            par_id = jobj["id"]

            labels = jobj["labels"]

            labels_per_par[par_id] = process_labels(labels)

    return labels_per_par


def compute_technique_frequency(annotations, technique_name):
    all_annotations = []
    for example_id, annot in annotations.items():
        for x in annot:
            all_annotations.append(x[0])

    techn_freq = sum([1 for a in all_annotations if a == technique_name])

    return techn_freq


def compute_span_score(gold_annots, pred_annots):
    # count total no of annotations
    prec_denominator = sum([len(pred_annots[x]) for x in pred_annots])
    rec_denominator = sum([len(gold_annots[x]) for x in gold_annots])

    technique_Spr_prec = {
        propaganda_technique: 0 for propaganda_technique in propaganda_techniques
    }
    technique_Spr_rec = {
        propaganda_technique: 0 for propaganda_technique in propaganda_techniques
    }
    cumulative_Spr_prec, cumulative_Spr_rec = (0, 0)
    f1_articles = []

    for example_id, pred_annot_obj in pred_annots.items():
        gold_annot_obj = gold_annots[example_id]
        # print("%s\t%d\t%d"parse_label_encoding % (example_id, len(gold_annot_obj), len(pred_annot_obj)))

        document_cumulative_Spr_prec, document_cumulative_Spr_rec = (0, 0)
        for j, pred_ann in enumerate(pred_annot_obj):
            s = ""
            ann_length = pred_ann[1][1] - pred_ann[1][0]

            for i, gold_ann in enumerate(gold_annot_obj):
                if pred_ann[0] == gold_ann[0]:
                    # print(pred_ann, gold_ann)

                    # s += "\tmatch %s %s-%s - %s %s-%s"%(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2])
                    intersection = span_intersection(gold_ann[1], pred_ann[1])
                    # print(intersection)
                    # print(intersection)
                    s_ann_length = gold_ann[1][1] - gold_ann[1][0]
                    Spr_prec = intersection / ann_length
                    document_cumulative_Spr_prec += Spr_prec
                    cumulative_Spr_prec += Spr_prec
                    s += (
                        "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|p| = %d/%d = %f (cumulative S(p,r)=%f)\n"
                        % (
                            pred_ann[0],
                            pred_ann[1][0],
                            pred_ann[1][1],
                            gold_ann[0],
                            gold_ann[1][0],
                            gold_ann[1][1],
                            intersection,
                            ann_length,
                            Spr_prec,
                            cumulative_Spr_prec,
                        )
                    )
                    technique_Spr_prec[gold_ann[0]] += Spr_prec

                    Spr_rec = intersection / s_ann_length
                    document_cumulative_Spr_rec += Spr_rec
                    cumulative_Spr_rec += Spr_rec
                    s += (
                        "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|r| = %d/%d = %f (cumulative S(p,r)=%f)\n"
                        % (
                            pred_ann[0],
                            pred_ann[1][0],
                            pred_ann[1][1],
                            gold_ann[0],
                            gold_ann[1][0],
                            gold_ann[1][1],
                            intersection,
                            s_ann_length,
                            Spr_rec,
                            cumulative_Spr_rec,
                        )
                    )
                    technique_Spr_rec[gold_ann[0]] += Spr_rec

        p_article, r_article, f1_article = compute_prec_rec_f1(
            document_cumulative_Spr_prec,
            len(pred_annot_obj),
            document_cumulative_Spr_rec,
            len(gold_annot_obj),
        )
        f1_articles.append(f1_article)

    p, r, f1 = compute_prec_rec_f1(
        cumulative_Spr_prec, prec_denominator, cumulative_Spr_rec, rec_denominator
    )

    f1_per_technique = []

    for technique_name in technique_Spr_prec.keys():
        prec_tech, rec_tech, f1_tech = compute_prec_rec_f1(
            technique_Spr_prec[technique_name],
            compute_technique_frequency(pred_annots, technique_name),
            technique_Spr_prec[technique_name],
            compute_technique_frequency(gold_annots, technique_name),
        )
        f1_per_technique.append(f1_tech)

    return p, r, f1, f1_per_technique


# if per_label is true, the scorer returns F1 score per technique
def FLC_score_to_string(gold_annotations, user_annotations, per_label):
    precision, recall, f1, f1_per_class = compute_span_score(
        gold_annotations, user_annotations
    )

    if per_label:
        res_for_screen = f"\nF1=%f\nPrecision=%f\nRecall=%f\n%s\n" % (
            f1,
            precision,
            recall,
            "\n".join(
                [
                    "F1_" + pr + "=" + str(f)
                    for pr, f in zip(propaganda_techniques, f1_per_class)
                ]
            ),
        )
    else:
        average = sum(f1_per_class) / len(f1_per_class)
        res_for_screen = f"Micro-F1\tMacro-F1\tPrecision\tRecall\n%f\t%f\t%f\t%f" % (
            f1,
            average,
            precision,
            recall,
        )

    res_for_script = "%f\t%f\t%f\t" % (f1, precision, recall)
    res_for_script += "\t".join([str(x) for x in f1_per_class])

    return res_for_screen, f1, dict(zip(propaganda_techniques, f1_per_class))


def sort_spans(spans):
    """
    sort the list of annotations with respect to the starting offset
    """
    spans = sorted(spans, key=lambda span: span[1][0])

    return spans


def compute_prec_rec_f1(
    prec_numerator, prec_denominator, rec_numerator, rec_denominator
):
    p, r, f1 = (0, 0, 0)
    if prec_denominator > 0:
        p = prec_numerator / prec_denominator
    if rec_denominator > 0:
        r = rec_numerator / rec_denominator
    if prec_denominator == 0 and rec_denominator == 0:
        f1 = 1.0
    if p > 0 and r > 0:
        f1 = 2 * (p * r / (p + r))

    return p, r, f1


def span_intersection(gold_span, pred_span):
    x = range(gold_span[0], gold_span[1])
    y = range(pred_span[0], pred_span[1])
    inter = set(x).intersection(y)
    return len(inter)


def validate_files(pred_file):
    base = os.path.basename(pred_file)
    file_basename = os.path.splitext(base)[0]
    subtask = file_basename.split("_")[0]
    print(base, file_basename, subtask)

    logging.info("Validating if passed files exist...")

    if not os.path.exists(pred_file):
        logging.error("File doesn't exist: {}".format(pred_file))
        return False

    # Check if the filename matches what is required by the task
    subtasks = ["task1", "task2"]

    if not any(file_basename.startswith(st_name) for st_name in subtasks):
        logging.error(
            "The submission file must start by task name! possible prefixes: "
            + str(subtasks)
        )
        return False

    return subtask


def main(gold_file, pred_file, output_dir=""):
    subtask = validate_files(pred_file)
    print("subtask", subtask)
    if subtask:
        if subtask == "task1":
            gold_annotations = load_json_as_list(gold_file)
            format_pass = check_format(pred_file, gold_annotations)
            if format_pass:
                user_annotations = load_json_as_list(pred_file)

                res_for_output = FLC_score_to_string(
                    gold_annotations, user_annotations, False
                )
                # with open(os.path.join(output_dir, 'scores.txt'), "w") as output_file:
                #     output_file.write("results: " + str(res_for_output) + "\n")
                print(res_for_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold-file",
        "-g",
        required=True,
        type=str,
        default="araieval24_task1_train.jsonl",
        help="Training file name",
    )
    parser.add_argument(
        "--predicted-file",
        "-p",
        required=True,
        type=str,
        default="predicted.jsonl",
        help="Test file name",
    )
    args = parser.parse_args()

    print("Evaluating %s" % args.predicted_file)

    main(args.gold_file, args.predicted_file)
