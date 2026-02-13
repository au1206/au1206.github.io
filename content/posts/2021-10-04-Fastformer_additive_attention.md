---
title: "Fastformer Annotated Paper"
date: 2021-10-04T00:00:30-04:00
categories: ["Annotated Paper"]
tags: ["NLP"]
draft: false
ShowToc: true
---

## Fastformer: Additive Attention Can Be All You Need ##

 Of late this paper is all the rage with its claims to introduce an attention mechanism that has a linear time complexity with respect to the sequence length. Why is this such a big deal you ask? Well, If you are familiar with transformers, one of the biggest downsides is the quadratic complexity which creates a huge bottleneck for longer sequences. So if additive attention works out, we will no longer have a strict cap of 512 tokens as introduced in the original and subsequent transformer-based architectures. This paper compares itself with other well-known efficient transformer techniques and conducts experiments on five well-known datasets.




Please feel free to read along with the paper with my notes and highlights.

| Color | Meaning |
| :---: | :--- | 
| Green | Topics about the current paper |
| Yellow | Topics about other relevant references |
| Blue | Implementation details/ maths/experiments |
| Red | Text including my thoughts, questions, and understandings | 



<embed src="/pdfs/Fastformer.pdf" width="1000px" height="2100px" />



Follow me on Github and star this repo for regular updates. [GitHub](https://github.com/au1206/paper_annotations)

Also, Follow me on [Twitter](https://twitter.com/akshayuppal12). 


**PS:** For now, the PDF Above does not render properly on mobile devices, so please download the pdf from the above button or get it from my [Github](https://github.com/au1206/paper_annotations)

<br>
## Citation
```bibtex
@misc{wu2021fastformer,
      title={Fastformer: Additive Attention Can Be All You Need}, 
      author={Chuhan Wu and Fangzhao Wu and Tao Qi and Yongfeng Huang and Xing Xie},
      year={2021},
      eprint={2108.09084},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```