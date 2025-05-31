# InterScope

This project is made for MontyHacks 2025 by **Samir Rangwalla, Anshuman Roy, and Shubham Roy-Choudhury**

## Table of Contents

- [InterScope](#interscope)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Our Purpose](#our-purpose)
  - [Key Features](#key-features)
  - [Functionality](#functionality)
  - [Why Our Model](#why-our-model)
  - [What we learned](#what-we-learned)
  - [Extensibility](#extensibility)
  - [Reasons Behind This Idea](#reasons-behind-this-idea)
  - [Sources](#sources)

## Introduction:
InterScope is an application that allows users to discover intriguing connections between two “unrelated” topics of the user’s choosing. Its primary utilization is for research and technical purposes, hypotheses, and methodologies for scientific experiments, allowing both professionals and students alike to delve further into niche research fields. InterScope is striving to develop a valuable exploration tool that will facilitate scientific breakthroughs and have a significant impact on researchers.

## Our Purpose:
According to data reported in 2021 from the National Center for Science and Engineering Statistics, approximately 47.2% of dissertation fields intersected 2 or more fields to lead to scientific breakthroughs. InterScope’s primary purpose is to spearhead these scientific breakthroughs by generating research topics merging 2 independent fields, such as “Bioinformatics” and “Artificial Intelligence”. We are reforming the way scientific research is approached and encouraging impact-based development globally. By combining two research interactions that have little to no prior published work, we are organizing a collective search for knowledge.

## Key Features:
- A home page featuring an “About” and “Get Started” section
- A topic selection page, prompting the user to enter two different topics, as well as enter a timeframe
- An output page showcasing a list of research keywords, hypothesis, methodologies, and other experimental procedures for research, and a regenerate button for further ideas.
- User Profile Page, featuring their username, password, and starred research procedures

## Functionality:
Topic Selection:
Users input one or two research categories, such as “bioinformatics”, “quantum computing”, and specify a date range.

Data Retrieval:
The system fetches data from each domain using public archive APIs, ensuring novelty in the ideas generated.

Keyword Extraction:
Utilizing techniques like TF-IDF, the system extracts the top 100 keywords from the collected topics for each category.

Gap Detection:
A co-occurrence matrix is constructed to identify keyword pairs that individually appear frequently but rarely together. These pairs are scored using the formula:
GapScore(A,B) = freq(A) × freq(B) – α × coOccur(A,B) where α is a small weighting factor is used to rank the pairs, and the top 10 are selected

Experiment Generation:
For the top-scoring keyword pairs, OpenAI’s o4-mini generates multiple possible experiments the researcher could perform, and even outlines hypotheses and procedures for each.

The results are then presented, including hypotheses, methodologies, and experimental procedures.

# Why Our Model:

Our model is a highly specialized research model, utilizing current and up-to-date information from reputable scientific studies and journals. In contrast, ChatGPT and other models are restricted in the number of sources they can access, which renders them less credible. 

## What we Learned:
During our development of InterScope, we learned many interesting things. Firstly, we were previously unaware of the importance of the intersection of research fields in leading scientific breakthroughs, at a rate of 47.2%. These included vaccinations, waste management, and more. In addition, through this project, we pushed ourselves to train and deploy an extremely ambitious model, one of our best works to date, although we had previously collaborated.

## Extensibility:
In the future, we aim to enhance InterScope’s user experience by enabling more dynamic interactions, allowing users to ask follow-up questions and receive detailed, tailored research proposals. We also plan to expand the platform’s guidance tools to better support users in pursuing their ideas beyond the initial suggestions. Additionally, we are committed to making InterScope more accessible, especially within research communities at the forefront of cross-disciplinary scientific innovation and discovery.

## Reasons Behind This Idea
We selected this concept as it is closely related to the theme of "Exploration," as we are fostering research and the desire to identify the correlation between two topics that the user inputs. In those areas, InterScope leads research by giving users hypotheses and methods for research proposals that they can use to run their own unique experiments.
## Sources:
National Center for Science and Engineering

InterScope - Explore "Interconnected" Ideas!
