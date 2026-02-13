---
title: "Few Shot NER Annotated Paper"
date: 2021-08-09T00:34:30-04:00
categories: ["Annotated Paper"]
tags: ["NLP"]
draft: false
ShowToc: true
---

## Few-Shot Named Entity Recognition: A Comprehensive Study ##

A lesser-known albeit important paper in my opinion. This paper highlights a key problem in the industry that does not always appear in research making it all the more impressive.
In this paper, the authors talk about the problem of less data for NER in industry and experimentally try the effects of three key approaches on few-shot NER:

- Meta-Learning: Construct prototypes for different entities
- Supervised pre-training on huge noisy data
- Self Training


Please feel free to read along with the paper with my notes and highlights.

| Color | Meaning |
| :---: | :--- | 
| Green | Topics about the current paper |
| Yellow | Topics about other relevant references |
| Blue | Implementation details/ maths/experiments |
| Red | Text including my thoughts, questions, and understandings | 



<embed src="/pdfs/Few_shot_NER.pdf" width="1000px" height="2100px" />

Follow me on Github and star this repo for regular updates. [GitHub](https://github.com/au1206/paper_annotations)

**PS:** For now, the PDF Above does not render properly on mobile devices, so please download the pdf from the above button or get it from my [Github](https://github.com/au1206/paper_annotations)


<br>
## Citation
```
@misc{huang2020fewshot,
      title={Few-Shot Named Entity Recognition: A Comprehensive Study}, 
      author={Jiaxin Huang and Chunyuan Li and Krishan Subudhi and Damien Jose and Shobana Balakrishnan and Weizhu Chen and Baolin Peng and Jianfeng Gao and Jiawei Han},
      year={2020},
      eprint={2012.14978},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```