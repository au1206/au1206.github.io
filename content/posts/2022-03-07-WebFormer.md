---
title: "WebFormer Annotated Paper"
date: 2022-03-07T00:00:30-04:00
categories: ["Annotated Paper"]
tags: ["NLP"]
draft: false
ShowToc: true
---

## WebFormer: The Web-page Transformer for Structure Information Extraction ##

 Understanding tokens from unstructured web pages is challenging in practice due to a variety of web layout patterns, this is where WebFormer comes into play. In this paper, the authors propose a novel architecture, WebFormer, a Web-page transFormer model for structure information extraction from web documents. This paper also introduces rich attention patterns between HTML tokens and text tokens, which leverages the web layout for effective attention weight computation. This can prove to be a big leap in web page understanding as it provides great incremental results and a way forward for the domain.



Please feel free to read along with the paper with my notes and highlights.

| Color | Meaning |
| :---: | :--- | 
| Green | Topics about the current paper |
| Yellow | Topics about other relevant references |
| Blue | Implementation details/ maths/experiments |
| Red | Text including my thoughts, questions, and understandings | 



<embed src="/pdfs/Webformer.pdf" width="1000px" height="2100px" />



Follow me on Github and star this repo for regular updates. [GitHub](https://github.com/au1206/paper_annotations)

Also, Follow me on [Twitter](https://twitter.com/akshayuppal12). 


**PS:** For now, the PDF Above does not render properly on mobile devices, so please download the pdf from the above button or get it from my [Github](https://github.com/au1206/paper_annotations)

<br>
## Citation
```
@misc{wu2021fastformer,
      title={WebFormer: The Web-page Transformer for Structure Information Extraction}, 
      author={Qifan Wang and Yi Fang and Anirudh Ravula and Fuli Feng and Xiaojun Quan and Dongfang Liu},
      year={2022},
      eprint={2202.00217},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```