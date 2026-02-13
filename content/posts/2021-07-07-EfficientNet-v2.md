---
title: "EfficientNet-V2 Annotated Paper"
date: 2021-07-07T00:34:30-04:00
categories: ["Annotated Paper"]
tags: ["CV"]
draft: false
ShowToc: true
---

## EfficientNetV2: Smaller Models and Faster Training ##

This very recent paper (1-month-old at the time of writing this) introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency. Based on top of the EfficientNet this paper pushes the boundary of model scaling and architecture search by further optimizing the network by using training aware Neural Architecture Search (NAS) and scaling. It jointly optimizes training speed and parameter efficiency to create the lightest best-performing models.



Please feel free to read along with the paper with my notes and highlights.

| Color | Meaning |
| :---: | :--- | 
| Green | Topics about the current paper |
| Yellow | Topics about other relevant references |
| Blue | Implementation details/ maths |
| Red | Text including my thoughts, questions, and understandings | 

<embed src="/pdfs/EfficientNet-v2.pdf" width="1000px" height="2100px" />

<br>

```
@misc{tan2021efficientnetv2,
      title={EfficientNetV2: Smaller Models and Faster Training}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2021},
      eprint={2104.00298},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```