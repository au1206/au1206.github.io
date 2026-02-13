# Akshay Uppal's ML Blog

Personal blog focused on Machine Learning, featuring paper annotations, tutorials, and technical articles.

## Built With

- **Static Site Generator**: Hugo (v0.155+)
- **Theme**: PaperMod (dark theme)
- **Hosting**: GitHub Pages
- **Content**: 14+ ML research paper annotations and tutorials

## Local Development

### Prerequisites

- Hugo Extended v0.155+ ([installation guide](https://gohugo.io/installation/))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/akshayuppal/akshayuppal.github.io.git
cd akshayuppal.github.io
```

2. Initialize submodules (PaperMod theme):
```bash
git submodule update --init --recursive
```

3. Start the development server:
```bash
hugo server
```

4. Visit http://localhost:1313

### Adding a New Post

1. Create a new markdown file in `content/posts/`:
```bash
hugo new content/posts/YYYY-MM-DD-post-title.md
```

2. Add frontmatter:
```yaml
---
title: "Your Post Title"
date: 2026-02-14T10:00:00-04:00
categories: ["Annotated Paper"]  # or ["Tutorials"]
tags: ["NLP", "Transformers"]
draft: false
ShowToc: true
---
```

3. Write your content in markdown

4. For PDFs, place them in `static/pdfs/` and embed with:
```markdown
<embed src="/pdfs/your-paper.pdf" width="1000px" height="2100px" />
```

## Deployment

Pushes to the `main` branch automatically trigger GitHub Actions to build and deploy the site.

## Content

### Annotated Papers
- DiT (Document Image Transformer)
- LayoutLM, LayoutLMv2
- BERT, RoBERTa
- Attention is All You Need
- EfficientNet v1 & v2
- And more...

### Tutorials
- Fine-Tuning BERT for Text Classification

## License

Content is © 2021-2026 Akshay Uppal. All rights reserved.

## Contact

- Email: akshayuppal12@gmail.com
- LinkedIn: [uppalakshay](https://www.linkedin.com/in/uppalakshay/)
- Twitter: [@akshayuppal12](https://twitter.com/akshayuppal12)
- GitHub: [@akshayuppal](https://github.com/akshayuppal)
