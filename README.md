# [ArAIEval](http://araieval.gitlab.io/) at [ArabicNLP-2024](https://arabicnlp2024.sigarab.org/)

In this edition we offered two tasks (i) persuasion techniques, and (ii) disinformation detection. Each task offers two subtasks with different variations. This repository contains the _dataset_, _format checker, scorer and baselines_ for each task of the [ArAIEval](https://araieval.gitlab.io/).



- [Task 1: Persuasion Technique Detection](task1)

  Given a multigenre text snippet (a news paragraph or a tweet), the task is to detect the propaganda techniques used in the text together with the exact span(s) in which each propaganda technique appears. This a sequence tagging task.


- [Task 2: Multimodal Propagandistic Memes Classification](task2)

  All tasks are formulated as binary classification tasks.
    - **Subtask 2A:** Unimodal Text
    - **Subtask 2B:** Unimodal Image
    - **Subtask 2C:** Multimodal

<!-- # Leaderboard

## Task 1
Kindly find the leaderboard [link](http://shorturl.at/nuCOS).

## Task 2
Kindly find the leaderboard [link](http://shorturl.at/dzX28). -->


# Licensing

Please check the task-specific directory for the licensing of the respective dataset.

# Credits

**Lab Organizers:** Please find on the task website: https://araieval.gitlab.io/

# Contact
Slack Channel: [join](https://join.slack.com/t/araieval/shared_invite/zt-20rzypxs7-LuHUsw8ltj7ylae9c4I7XQ) <br>
Email: [araieval@googlegroups.com](araieval@googlegroups.com)

# Citation

Please find relevant papers below:

<a id="1">[1]</a>
Maram Hasanain, and Fatema Ahmed and Firoj Alam. "Can GPT-4 Identify Propaganda? Annotation and Detection of Propaganda Spans in News Articles." arXiv preprint arXiv:2402.17478 (2024).
<br>
<a id="2">[2]</a>
Maram Hasanain and Fatema Ahmed and Firoj Alam. "Large Language Models for Propaganda Span Annotation." arXiv preprint arXiv:2311.09812 (2023).
<br>
<a id="3">[3]</a>
Maram Hasanain, Firoj Alam, Hamdy Mubarak, Samir Abdaljalil, Wajdi Zaghouani, Preslav Nakov, Giovanni Da San Martino, and Abed Alhakim Freihat. 2023. "ArAIEval Shared Task: Persuasion Techniques and Disinformation Detection in Arabic Text." In Proceedings of the First Arabic Natural Language Processing Conference (ArabicNLP 2023), December. Singapore: Association for Computational Linguistics.
<br>
<a id="4">[4]</a>
Firoj Alam, Hamdy Mubarak, Wajdi Zaghouani, Giovanni Da San Martino, and Preslav Nakov. 2022. "Overview of the WANLP 2022 Shared Task on Propaganda Detection in Arabic.", In Proceedings of the Seventh Arabic Natural Language Processing Workshop (WANLP), 108–118, December. Abu Dhabi, United Arab Emirates (Hybrid): Association for Computational Linguistics. [https://aclanthology.org/2022.wanlp-1.11](https://aclanthology.org/2022.wanlp-1.11).
<br>

```
@article{hasanain2023large,
  title={Large Language Models for Propaganda Span Annotation},
  author={Hasanain, Maram and Ahmed, Fatema and Alam, Firoj},
  journal={arXiv preprint arXiv:2311.09812},
  year={2023}
}
@inproceedings{hasanain2024can,
  title={Can GPT-4 Identify Propaganda? Annotation and Detection of Propaganda Spans in News Articles},
  author={Hasanain, Maram and Ahmed, Fatema and Alam, Firoj},
  booktitle={Proceedings of the Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year={2024}
}
@inproceedings{araieval:arabicnlp2023-overview,
    title = "ArAIEval Shared Task: Persuasion Techniques and Disinformation Detection in Arabic Text",
    author = "Hasanain, Maram and Alam, Firoj and Mubarak, Hamdy, and Abdaljalil, Samir  and Zaghouani, Wajdi and Nakov, Preslav  and Da San Martino, Giovanni and Freihat, Abed Alhakim",
    booktitle = "Proceedings of the First Arabic Natural Language Processing Conference (ArabicNLP 2023)",
    month = Dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}

@inproceedings{alam-etal-2022-overview,
    title = "Overview of the {WANLP} 2022 Shared Task on Propaganda Detection in {A}rabic",
    author = "Alam, Firoj  and
      Mubarak, Hamdy  and
      Zaghouani, Wajdi  and
      Da San Martino, Giovanni  and
      Nakov, Preslav",
    booktitle = "Proceedings of the The Seventh Arabic Natural Language Processing Workshop (WANLP)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wanlp-1.11",
    pages = "108--118",
}

```

## Additional Resources (Tools/Source code)
We listed the following tools/source code, which might be helpful to run the experiments.
* https://huggingface.co/docs/transformers/en/tasks/image_classification
* https://llava-vl.github.io/
* https://github.com/dsaidgovsg/multimodal-learning-hands-on-tutorial
* https://fasttext.cc/docs/en/supervised-tutorial.html
* https://github.com/facebookresearch/mmf

## Recommended reading
The following papers might be useful. We have not provided exhaustive list. But these could be a good start.<br>
[Download bibliography](bibtex/bibliography.bib)

**Persuasion Techniques Detection**
* Maram Hasanain, Ahmed El-Shangiti, Rabindra Nath Nandi, Preslav Nakov, and Firoj Alam. 2023. **QCRI at SemEval-2023 Task 3: News Genre, Framing and Persuasion Techniques Detection Using Multilingual Models.** In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 1237–1244, Toronto, Canada. Association for Computational Linguistics.
* Jakub Piskorski, Nicolas Stefanovitch, Giovanni Da San Martino, and Preslav Nakov. 2023. **SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup**. In Proceedings of the The 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2343–2361, Toronto, Canada. Association for Computational Linguistics.
* Alam, Firoj, Hamdy Mubarak, Wajdi Zaghouani, Giovanni Da San Martino, and Preslav Nakov. **"Overview of the WANLP 2022 Shared Task on Propaganda Detection in Arabic."** In Proceedings of the The Seventh Arabic Natural Language Processing Workshop (WANLP), pp. 108-118. Association for Computational Linguistics, 2022.
* Dimitrov, Dimitar, Bishr Bin Ali, Shaden Shaar, Firoj Alam, Fabrizio Silvestri, Hamed Firooz, Preslav Nakov, and Giovanni Da San Martino **"SemEval-2021 Task 6: Detection of Persuasion Techniques in Texts and Images."** Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021). 2021.
* Da San Martino, Giovanni, Shaden Shaar, Yifan Zhang, Seunghak Yu, Alberto Barrón-Cedeno, and Preslav Nakov. **"Prta: A system to support the analysis of propaganda techniques in the news."** In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 287-293. 2020.
* Dimitrov, Dimitar, Bishr Bin Ali, Shaden Shaar, Firoj Alam, Fabrizio Silvestri, Hamed Firooz, Preslav Nakov, and Giovanni Da San Martino. **"Detecting Propaganda Techniques in Memes."** In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 6603-6617. 2021
* Sharma, Shivam, Firoj Alam, Md Shad Akhtar, Dimitar Dimitrov, Giovanni Da San Martino, Hamed Firooz, Alon Halevy, Fabrizio Silvestri, Preslav Nakov, and Tanmoy Chakraborty. **"Detecting and Understanding Harmful Memes: A Survey."** In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence,{IJCAI} 2022, Vienna, Austria, 23-29 July 2022, pp. 5597-5606. 2022.
* Alam, Firoj, Stefano Cresci, Tanmoy Chakraborty, Fabrizio Silvestri, Dimiter Dimitrov, Giovanni Da San Martino, Shaden Shaar, Hamed Firooz, and Preslav Nakov. **"A Survey on Multimodal Disinformation Detection."** In Proceedings of the 29th International Conference on Computational Linguistics, pp. 6625-6643. 2022.
