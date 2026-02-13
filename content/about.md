---
title: "About Me"
url: "/about/"
ShowToc: false
ShowBreadCrumbs: false
---

<style>
.hero-section {
  text-align: center;
  padding: 2rem 0;
}

.hero-section h2 {
  font-size: 2.5rem;
  margin: 1.5rem 0 0.5rem 0;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-tagline {
  font-size: 1.2rem;
  color: #aaa;
  margin: 0.5rem auto 1.5rem;
  font-weight: 500;
}

.hero-subtitle {
  font-size: 1.1rem;
  color: #888;
  line-height: 1.8;
  max-width: 900px;
  margin: 0 auto 1.5rem;
}

.consulting-badge {
  display: inline-block;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 0.5rem 1.5rem;
  border-radius: 25px;
  text-decoration: none;
  font-weight: 500;
  margin: 1rem 0;
  transition: transform 0.2s;
}

.consulting-badge:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

.section-divider {
  margin: 3rem 0;
  border: none;
  border-top: 2px solid #333;
  opacity: 0.3;
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  text-align: center;
  margin: 2rem 0 1.5rem 0;
}

.category-title {
  font-size: 1rem;
  font-weight: 600;
  color: #888;
  text-align: center;
  margin: 1.5rem 0 0.8rem 0;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.badge-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin-bottom: 1rem;
  padding: 0 2rem;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}

.badge-container a {
  display: inline-block;
  transition: transform 0.2s;
}

.badge-container a:hover {
  transform: translateY(-3px);
}

.expertise-section {
  margin: 3rem auto;
  max-width: 1600px;
  padding: 0 2rem;
}

.expertise-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  margin: 0 auto;
}

@media (max-width: 968px) {
  .expertise-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 640px) {
  .expertise-grid {
    grid-template-columns: 1fr;
  }
}

.expertise-card {
  background: rgba(102, 126, 234, 0.05);
  border: 1px solid rgba(102, 126, 234, 0.2);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  transition: transform 0.3s, box-shadow 0.3s, background 0.3s;
}

.expertise-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
  background: rgba(102, 126, 234, 0.1);
}

.expertise-icon {
  font-size: 2rem;
  margin-bottom: 0.6rem;
}

.expertise-card h4 {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0.3rem 0;
  color: #667eea;
}

.expertise-card p {
  font-size: 0.9rem;
  line-height: 1.5;
  color: #888;
  margin: 0.4rem 0 0 0;
}

.stats-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 20px;
  margin: 2rem auto;
  padding: 0 2rem;
  max-width: 1200px;
}

.tech-stack-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, max-content));
  gap: 10px;
  justify-content: center;
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem 2rem;
}

.tech-stack-grid a {
  display: inline-block;
  transition: transform 0.2s, box-shadow 0.2s;
}

.tech-stack-grid a:hover {
  transform: translateY(-3px) scale(1.05);
  filter: brightness(1.1);
}

.pikachu-gif {
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
}

@media (max-width: 768px) {
  .pikachu-gif {
    position: relative !important;
    right: auto !important;
    top: auto !important;
    width: 150px !important;
    display: block;
    margin: 0 auto 1rem;
  }
}
</style>

<div class="hero-section">

![Header image](/images/Header.png)

<div style="position: relative;">
<img src="https://media.giphy.com/media/AFdcYElkoNAUE/giphy.gif" alt="Pikachu" style="position: absolute; right: 0; top: -50px; width: 200px; z-index: 10;" class="pikachu-gif"/>

## Akshay Uppal

<p class="hero-tagline">
Machine Learning Engineer • Researcher • Technical Writer
</p>

<p class="hero-subtitle">
Building intelligent systems at the intersection of Document AI, NLP, and Generative AI. Specializing in conversational interfaces and LLM-powered applications. I demystify cutting-edge ML research through detailed paper annotations and practical tutorials, making advanced AI concepts accessible to engineers and researchers worldwide.
</p>

<a href="https://cleverx.com/@Akshay-U" class="consulting-badge" target="_blank">
💼 Available for Consulting
</a>

</div>

</div>

<hr class="section-divider">

<div class="expertise-section">
<h3 class="section-title" style="margin-bottom: 2rem;">🎯 Areas of Expertise</h3>

<div class="expertise-grid">
<div class="expertise-card">
<div class="expertise-icon">📄</div>
<h4>Document AI</h4>
<p>Intelligent document processing, layout analysis, and information extraction from structured and unstructured documents.</p>
</div>

<div class="expertise-card">
<div class="expertise-icon">🧠</div>
<h4>Natural Language Processing</h4>
<p>Advanced NLP techniques, transformers, language models, and text understanding systems for real-world applications.</p>
</div>

<div class="expertise-card">
<div class="expertise-icon">💬</div>
<h4>Conversational AI</h4>
<p>Building intelligent chatbots, dialogue systems, and conversational interfaces using LLMs and advanced NLP techniques.</p>
</div>

<div class="expertise-card">
<div class="expertise-icon">✨</div>
<h4>Generative AI</h4>
<p>Working with large language models, prompt engineering, RAG systems, and creating AI-powered content generation solutions.</p>
</div>

<div class="expertise-card">
<div class="expertise-icon">⚡</div>
<h4>ML Engineering</h4>
<p>Production ML systems, model deployment, API development, and scalable ML infrastructure using FastAPI and modern tools.</p>
</div>

<div class="expertise-card">
<div class="expertise-icon">📊</div>
<h4>LLM Evaluation</h4>
<p>Comprehensive evaluation frameworks for large language models, including performance metrics, bias detection, and quality assessment.</p>
</div>
</div>
</div>

<hr class="section-divider">

<h3 class="section-title">🛠️ Tech Stack</h3>

<div class="tech-stack-grid">
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
<a href="https://www.tensorflow.org"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/></a>
<a href="https://huggingface.co"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/></a>
<a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/></a>
<a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
<a href="https://flask.palletsprojects.com"><img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/></a>
<a href="https://git-scm.com"><img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git"/></a>
<a href="https://postman.com"><img src="https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=postman&logoColor=white" alt="Postman"/></a>
</div>

<hr class="section-divider">

<h3 class="section-title">🌐 Connect With Me</h3>

<div class="badge-container">
<a href="mailto:akshayuppal12@gmail.com"><img src="https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=Gmail&logoColor=white" alt="Email"/></a>
<a href="https://www.linkedin.com/in/uppalakshay/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=Linkedin&logoColor=white" alt="LinkedIn"/></a>
<a href="https://twitter.com/akshayuppal12"><img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=Twitter&logoColor=white" alt="Twitter"/></a>
<a href="https://github.com/akshayuppal"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub"/></a>
<a href="https://kaggle.com/au1206"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle"/></a>
</div>

<hr class="section-divider">

<h3 class="section-title">📊 GitHub Stats</h3>

<div class="stats-container">
<img src="https://github-readme-stats.vercel.app/api/top-langs?username=akshayuppal&show_icons=true&theme=tokyonight&layout=compact&hide_border=true" alt="Top Languages" width="400"/>
<img src="https://github-readme-stats.vercel.app/api?username=akshayuppal&show_icons=true&theme=tokyonight&hide_border=true" alt="GitHub Stats" width="400"/>
</div>
