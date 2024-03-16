# Task 1: Unimodal (Text) Propagandistic Technique Detection


The aim of this task is to identify the propagandistic content from multigenre (tweet and news paragraphs of the news articles) text. Please see the [Task Description](#task-description) below.

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
* __[14/03/2024]__  Training and dev data released for both subtasks


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

Given a multigenre text snippet (a news paragraph or a tweet), the task is to detect the propaganda techniques used in the text together with the exact span(s) in which each propaganda technique appears. This a sequence tagging task.

## Dataset
All datasets are JSONL files where each line has one JSON object. The text encoding is UTF-8.
The input and the result files have the same format for all the subtasks.

### Input data format

Each object has the following format:
```
{
  id -> identifier of the example,
  text -> text,
  label -> list of json object contains techniques and other information,
  type -> type of text: tweet or news article
}
```
##### Example
```
    {
        "id": "7365",
        "text": "قائد الجيش "خطّ أحمر" وواشنطن "راجعة"... LINK",
        "labels": [
			{
			  "start": 0,
			  "end": 50,
			  "technique": "Appeal_to_Fear-Prejudice",
			  "text": "تحذيرات من حرب جديدة في حال فشل الانتخابات القادمة"
			},
			{
			  "start": 11,
			  "end": 14,
			  "technique": "Loaded_Language",
			  "text": "حرب"
			}
		],
	  "type": "tweet"
	}
```

### Output Data Format

Each object has the following format:
```
{
  id -> identifier of the example,
  label -> list of json object contains techniques and other information
}
```

##### Example

```
    {
        "id": "7365",
        "labels": [
			{
			  "start": 0,
			  "end": 50,
			  "technique": "Appeal_to_Fear-Prejudice",
			  "text": "تحذيرات من حرب جديدة في حال فشل الانتخابات القادمة"
			},
			{
			  "start": 11,
			  "end": 14,
			  "technique": "Loaded_Language",
			  "text": "حرب"
			}
		]
	}
```

## Scorer and Official Evaluation Metrics

### Scorers

The scorer for the subtasks is located in the [scorer](scorer) module of the project. The scorer will report official evaluation metric and other metrics of a prediction file. The scorer invokes the format checker for the task to verify the output is properly shaped.
<!-- It also handles checking if the provided predictions file contains all tweets from the gold one. -->


You can install all prerequisites through,
```
pip install -r requirements.txt
```
Launch the scorer for the subtask as follows:
```
python scorer/task1.py --gold-file=<path_gold_file> --predicted-file=<predictions_file>
```


##### Example

```
python scorer/task1.py --predicted_file task1_dev_output.jsonl --gold_file data/task1_dev.jsonl
```

### Official Evaluation Metrics
The **official evaluation metric** for the task is **modified micro-F1**. However, the scorer also reports macro-F1, Precision, and Recall.


## Baselines
The same script can be used for both subtasks.

##### Random baseline
```
python baselines/task1.py --dev_file_path data/araieval24_task1_dev.jsonl --output_file_path task1_random_baseline.jsonl
 ```

If you submit the predictions of the baseline on the development set to the shared task website, you would get a modfied micro-F1 score of 0.0260.


## Format checker

The format checker for both subtsaks is located in the [format_checker](format_checker) module of the project. The format checker verifies that your generated results file complies with the expected format.

Before running the format checker please install all prerequisites through,
```
pip install -r requirements.txt
```

To launch it, please run the following command:

```
python format_checker/task1.py -p paths_to_your_results_files -g paths_to_your_gold_file
```

##### Example:
```
python format_checker/task1.py -p task1_random_baseline.jsonl -g data/araieval24_task1_dev.jsonl
```

**paths_to_your_results_files**: can be one path or space seperated list of paths


## Submission

### Guidelines
The process consists of two phases:

1. **System Development Phase:** This phase involves working on the dev set.
2. **Final Evaluation Phase (will start on 27 April 2023):** This phase involves working on the test set, which will be released during the evaluation cycle.

For each phase, please adhere to the following guidelines:
- Each team should create and maintain a single account for submissions. Please ensure all runs are submitted through this account. Submissions from multiple accounts by the same team could result in your system being not ranked in the overview paper.
- The most recent file submitted to the leaderboard will be considered your final submission.
- The output file must be named task1_any_suffix.jsonl, where [1] (for example, task1_team_name.jsonl). Failure to follow this naming convention will result in an error on the leaderboard.
- You are required to compress the .jsonl file into a .zip file (for example, zip task1.zip task1.jsonl) and submit it via the Codalab page.
- Please include your team name and a description of your method with each submission.
- You are permitted to submit a maximum of 200 submissions per day for each subtask.

### Submission Site
Please submit your results on the respective subtask tab: https://codalab.lisn.upsaclay.fr/competitions/18111

## Licensing
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. This allows for sharing and adapting the work, provided that attribution is given, the use is non-commercial, and any derivative works are shared under the same terms. For more information, please visit https://creativecommons.org/licenses/by-nc-sa/4.0/.

## Credits
Please find it on the task website: https://araieval.gitlab.io/
