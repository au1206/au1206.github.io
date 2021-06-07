---
title: "Annotated Papers"
permalink: /annotated-papers/
classes: wide
author_profile: true
# categories:
#  - Annotated Paper
---

Here is the List for all Annotated papers
<ul>
{% for tag in site.tags %}
  <p>{{ tag[0] }}</p>
  <!-- {% if tag[0] == "Annotated_Papers" %} -->
    <h3>{{ tag[0] }}</h3>
        {% for post in tag[1] %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endfor %}
    <!-- {% endif %} -->
{% endfor %}
</ul>

<!-- <ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      {{ post.excerpt }}
    </li>
  {% endfor %}
</ul> -->