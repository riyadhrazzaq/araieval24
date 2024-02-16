# [ArAIEval](http://araieval.gitlab.io/) at [ArabicNLP](http://arabicnlp2023.sigarab.org/) 2023

In this edition we offered two tasks (i) persuasion techniques, and (ii) disinformation detection. Each task offers two subtasks with different variations. This repository contains the _dataset_, _format checker, scorer and baselines_ for each task of the [ArabicNLP2023-ArAIEval](https://araieval.gitlab.io/).



- [Task 1: Persuasion Technique Detection](task1)
  - **Subtask 1A:** Given a multigenre (tweet and news paragraph) text snippet, identify whether it contains content with persuasion technique. This is a binary classification task.
  - **Subtask 1B:** Given a multigenre (tweet and news paragraph) text snippet, identify the propaganda techniques used in it. This is a multilabel classification task.


- [Task 2: Disinformation Detection](task2)
    - **Subtask 2A:** Disinfo vs Not-disinfo
    - **Subtask 2B:** Multiclass classification task

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
Maram Hasanain, Firoj Alam, Hamdy Mubarak, Samir Abdaljalil, Wajdi Zaghouani, Preslav Nakov, Giovanni Da San Martino, and Abed Alhakim Freihat. 2023. "ArAIEval Shared Task: Persuasion Techniques and Disinformation Detection in Arabic Text." In Proceedings of the First Arabic Natural Language Processing Conference (ArabicNLP 2023), December. Singapore: Association for Computational Linguistics.
<br>
<a id="2">[2]</a>
Firoj Alam, Hamdy Mubarak, Wajdi Zaghouani, Giovanni Da San Martino, and Preslav Nakov. 2022. "Overview of the WANLP 2022 Shared Task on Propaganda Detection in Arabic.", In Proceedings of the Seventh Arabic Natural Language Processing Workshop (WANLP), 108–118, December. Abu Dhabi, United Arab Emirates (Hybrid): Association for Computational Linguistics. [https://aclanthology.org/2022.wanlp-1.11](https://aclanthology.org/2022.wanlp-1.11).
<br>
<a id="3">[3]</a>
Mubarak, Hamdy, Samir Abdaljalil, Azza Nassar, and Firoj Alam. 2023. “Detecting and Identifying the Reasons for Deleted Tweets Before They Are Posted.” Frontiers in Artificial Intelligence 6. https://doi.org/10.3389/frai.2023.1219767.

```
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

@article{10.3389/frai.2023.1219767,
  author    = {Hamdy Mubarak and Samir Abdaljalil and Azza Nassar and Firoj Alam},
  title     = {Detecting and identifying the reasons for deleted tweets before they are posted},
  journal   = {Frontiers in Artificial Intelligence},
  volume    = {6},
  year      = {2023},
  url       = {https://www.frontiersin.org/articles/10.3389/frai.2023.1219767},
  doi       = {10.3389/frai.2023.1219767},
  issn      = {2624-8212},  
}

```

## Additional Resources (Tools/Source code)
We listed the following tools/source code, which might be helpful to run the experiments.
* https://fasttext.cc/docs/en/supervised-tutorial.html
* https://huggingface.co/docs/transformers/training
* https://github.com/Tiiiger/bert_score
* https://github.com/clef2018-factchecking/clef2018-factchecking
* https://github.com/utahnlp/x-fact
* https://github.com/firojalam/COVID-19-disinformation/tree/master/bin
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

**Fact-checking**
* Nakov, Preslav, Giovanni Da San Martino, Tamer Elsayed, Alberto Barrón-Cedeño, Rubén Míguez, Shaden Shaar, Firoj Alam et al. **"Overview of the CLEF–2021 CheckThat! Lab on Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News."** In International Conference of the Cross-Language Evaluation Forum for European Languages, pp. 264-291. Springer, Cham, 2021.
* Shahi, Gautam Kishore, Julia Maria Struß, and Thomas Mandl. **"Overview of the CLEF-2021 CheckThat! lab task 3 on fake news detection."** Working Notes of CLEF (2021).
* Barrón-Cedeño, Alberto, Tamer Elsayed, Preslav Nakov, Giovanni Da San Martino, Maram Hasanain, Reem Suwaileh, Fatima Haouari et al. **"Overview of CheckThat! 2020: Automatic identification and verification of claims in social media."** In International Conference of the Cross-Language Evaluation Forum for European Languages, pp. 215-236. Springer, Cham, 2020.
* Shaar, Shaden, Alex Nikolov, Nikolay Babulkov, Firoj Alam, Alberto Barrón-Cedeno, Tamer Elsayed, Maram Hasanain et al. **"Overview of CheckThat! 2020 English: Automatic identification and verification of claims in social media."** In International Conference of the Cross-Language Evaluation Forum for European Languages. 2020.
* Hasanain, Maram, Fatima Haouari, Reem Suwaileh, Zien Sheikh Ali, Bayan Hamdan, Tamer Elsayed, Alberto Barrón-Cedeno, Giovanni Da San Martino, and Preslav Nakov. **"Overview of CheckThat! 2020 Arabic: Automatic identification and verification of claims in social media."** In International Conference of the Cross-Language Evaluation Forum for European Languages. 2020.
* Elsayed, Tamer, Preslav Nakov, Alberto Barrón-Cedeno, Maram Hasanain, Reem Suwaileh, Giovanni Da San Martino, and Pepa Atanasova. **"Overview of the CLEF-2019 CheckThat! Lab: automatic identification and verification of claims."** In International Conference of the Cross-Language Evaluation Forum for European Languages, pp. 301-321. Springer, Cham, 2019.
* Elsayed, Tamer, Preslav Nakov, Alberto Barrón-Cedeno, Maram Hasanain, Reem Suwaileh, Giovanni Da San Martino, and Pepa Atanasova. **"CheckThat! at CLEF 2019: Automatic identification and verification of claims."** In European Conference on Information Retrieval, pp. 309-315. Springer, Cham, 2019.
* Nakov, Preslav, Alberto Barrón-Cedeno, Tamer Elsayed, Reem Suwaileh, Lluís Màrquez, Wajdi Zaghouani, Pepa Atanasova, Spas Kyuchukov, and Giovanni Da San Martino. **"Overview of the CLEF-2018 CheckThat! Lab on automatic identification and verification of political claims."** In International conference of the cross-language evaluation forum for european languages, pp. 372-387. Springer, Cham, 2018.
* Barrón-Cedeno, Alberto, Tamer Elsayed, Reem Suwaileh, Lluís Màrquez, Pepa Atanasova, Wajdi Zaghouani, Spas Kyuchukov, Giovanni Da San Martino, and Preslav Nakov. **"Overview of the CLEF-2018 CheckThat! Lab on Automatic Identification and Verification of Political Claims. Task 2: Factuality."**   CLEF (Working Notes) 2125 (2018).


**COVID-19 Infodemic**
* Alam, Firoj, Shaden Shaar, Fahim Dalvi, Hassan Sajjad, Alex Nikolov, Hamdy Mubarak, Giovanni Da San Martino et al. **["Fighting the COVID-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society."](https://aclanthology.org/2021.findings-emnlp.56.pdf)** In Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 611-649. 2021.
* Shaar, Shaden, Firoj Alam, Giovanni Da San Martino, Alex Nikolov, Wajdi Zaghouani, Preslav Nakov, and Anna Feldman. **"Findings of the NLP4IF-2021 Shared Tasks on Fighting the COVID-19 Infodemic and Censorship Detection."** In Proceedings of the Fourth Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda, pp. 82-92. 2021.
* Nakov, Preslav, Firoj Alam, Shaden Shaar, Giovanni Da San Martino, and Yifan Zhang. **"A Second Pandemic? Analysis of Fake News about COVID-19 Vaccines in Qatar."** In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pp. 1010-1021. 2021.
* Nakov, Preslav, Firoj Alam, Shaden Shaar, Giovanni Da San Martino, and Yifan Zhang. **"COVID-19 in Bulgarian social media: Factuality, harmfulness, propaganda, and framing."** In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pp. 997-1009. 2021.
