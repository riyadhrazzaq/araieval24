# Task 2: Disinformation Detection

The aim of this task is to identify disinformative content from tweets. This year we offer two subtasks. Please see the [Task Description](#task-description) below.

Please follow the website https://araieval.gitlab.io/ for any latest information.


__Table of contents:__
- [List of Versions](#list-of-versions)
- [Contents of the Directory](#contents-of-the-directory)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [Scorer and Official Evaluation Metrics](#scorer-and-official-evaluation-metrics)
- [Baselines](#baselines)
- [Format checker](#format-checker)
- [Submission](#submission)
- [Licensing](#licensing)
- [Credits](#Credits)

## List of Versions
* __[13/08/2023]__  Test released for both subtasks
* __[10/07/2023]__  Training and dev data released for both subtask



## Contents of the Directory
* Main folder: [data](./data)<br/>
  This directory contains files for the subtasks.
* Main folder: [baselines](./baselines)<br/>
	Contains scripts provided for baseline models of the tasks.
* Main folder: [format_checker](./format_checker)<br/>
	Contains scripts provided to check format of the submission file.
* Main folder: [scorer](./scorer)<br/>
	Contains scripts provided to score output of the model when provided with label (i.e., dev).

* [README.md](./README.md) <br/>
	This file!

## Task Description

**Subtask 2A:** Given a tweet, categorize whether it is disinformative. This is a binary classification task.

**Subtask 2B:** Given a tweet, detect the fine-grained disinformation class, if any. This is a multiclass classification task. The fine-grained labels include _hate-speech_, _offensive_, _rumor_, and _spam_

## Dataset

### Input data format
The format is identical for both subtasks. Each file uses the JSONL format. An object within the JSON adheres to the following structure:
```
{
  id -> identifier of the example,
  text -> text,
  label -> the label annotated for the text,

}
```
##### Example
```
    {
        "id": "2061",
        "text": #شركات_الاتصالات #شركات_الأدوية #شركات_المعقمات المستفيد الاكبر من #كورونا... اين انتم؟ اين مبادراتكم؟,
        "label": "disinfo"
    }
```


## Scorer and Official Evaluation Metrics

### Scorers

The scorer for the subtasks is located in the [scorer](scorer) module of the project. The scorer will report official evaluation metric and other metrics of a prediction file. The scorer invokes the format checker for the task to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all tweets from the gold one.


You can install all prerequisites through,
```
pip install -r requirements.txt
```
Launch the scorer for the subtask as follows:
```
python scorer/task2.py --gold-file-path=<path_gold_file> --pred-file-path=<predictions_file> --subtask=<name_of_the_subtask>
```

`--subtask` expects options for subtasks (2A or 2B) to indicate the subtask for which to score the predictions file.

##### Example

```
python scorer/task2.py --pred_files_path task2B_dev_output.txt --gold_file_path data/ArAiEval23_disinfo_subtask2B_dev --subtask 2B
```

### Official Evaluation Metrics
The **official evaluation metric** for the task is **micro-F1**. However, the scorer also reports macro-F1.


## Baselines
The same script can be used for both subtasks.

##### Random baseline
```
python baselines/task2.py --dev_file_path data/ArAiEval23_disinfo_subtask2B_dev.jsonl --output_file_path task2B.txt --subtask 2B
 ```
If you submit the predictions of the baseline on the development set for 2A to the shared task website, you would get a micro-F1 score of 0.5173.

If you submit the predictions of the baseline on the development set for 2B to the shared task website, you would get a micro-F1 score of 0.2191.


## Format checker

The format checkers for both subtsaks are located in the [format_checker](format_checker) module of the project. The format checker verifies that your generated results file complies with the expected format.

Before running the format checker please install all prerequisites through,
```
pip install -r requirements.txt
```

To launch it, please run the following command:

```
python format_checker/task2.py -p paths_to_your_results_files -s 2A/2B
```

##### Example
```
python format_checker/task2.py -p ./task2A.txt -s 2A
```
**paths_to_your_results_files**: can be one path or space separated list of paths


**Note that the checker cannot verify whether the prediction file you submit contains all lines, because it does not have access to the corresponding gold file.**


## Submission

### Guidelines

The process consists of two phases:

1. **System Development Phase:** This phase involves working on the *dev set*.
2. **Final Evaluation Phase:** This phase involves working on the *test set*, which will be released during the ***evaluation cycle***.
For each phase, please adhere to the following guidelines:

- Each team should create and maintain a single account for submissions. Please ensure all runs are submitted through this account. Submissions from multiple accounts by the same team could result in your system being not ranked in the overview paper.
- The most recent file submitted to the leaderboard will be considered your final submission.
- The output file must be named task2[A/B].tsv, where A/B is your specific subtask (for example, task2B_team_name.tsv). Failure to follow this naming convention will result in an error on the leaderboard.  Subtasks include 2A and 2B.
- You are required to compress the .tsv file into a .zip file (for example, zip task2B.zip task2B.tsv) and submit it via the Codalab page.
- Please include your team name and a description of your method with each submission.
- You are permitted to submit a maximum of 200 submissions per day for each subtask.

### Submission Site
**System Development Phase:** Please submit your results on the respective subtask tab: https://codalab.lisn.upsaclay.fr/competitions/14563

**Final Evaluation Phase:** Please submit your results on the respective subtask tab: https://codalab.lisn.upsaclay.fr/competitions/15099

## Licensing
The dataset is free for general research use.


## Credits
Please find it on the task website: https://araieval.gitlab.io/
