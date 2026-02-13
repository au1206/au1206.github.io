---
title: "EfficientNet Annotated Paper"
date: 2021-07-07T00:34:30-04:00
categories: ["Annotated Paper"]
tags: ["CV"]
draft: false
ShowToc: true
---

## EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks ##

This paper created a huge mark in the field of model scaling and parametric optimization in terms of model architecture.
It brings us a new scaling method called compund scaling, to scale the convolution network in all the three dimensions - Width, Depth and, resolution/channels. Along with this novel way of scaling it also brings us a new family of architecture created using Neural Architecture Search called the EfficentNet Family.

Please feel free to read along with the paper with my notes and highlights.

| Color | Meaning |
| :---: | :--- | 
| Green | Topics about the current paper |
| Yellow | Topics about other relevant references |
| Blue | Implementation details/ maths |
| Red | Text including my thoughts, questions, and understandings | 

<embed src="/pdfs/EfficientNet.pdf" width="1000px" height="2100px" />

<br>

```
@misc{tan2020efficientnet,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```