# Data Management & Collection Lab - On Target 

## Not to Lie, but to Emphasize

### Jornet Jeremy - Sasson Eden - Bijaoui Tom

Technion - Israel Institute of Technology - Faculty of Data Science and Decisions

![](https://upload.wikimedia.org/wikipedia/commons/b/b7/Technion_logo.svg)


## Project's Files

* Model Interpretability
    * Model Interpretability Notebook
* On Target tool
    * On Target Notebook
    * Example generated instructions
* Scraping
    * Scraping "Comparably"
    * Scraping companies' websites
    * An example of scraped informations from "Comparably" in CSV format


## Overview
First impressions and emphasizing the right qualities are crucial in the business world, especially when applying for a job. That's why we developed On Target, our revolutionary big data tool. This system generates tailored guidelines that enable candidates to highlight the skills and values most valued by the specific company they are targeting using Statistical and Machine Learning methods.

## Implementing Methods
* Data Preprocessing
* Features Engineering
* Pre-trained models from Hugging Face
* Statistical tests to significance inference
* Machine Learning model to Model Interpretability
* NLP keywords extraction techniques
* LLM to generate instructions

## On Target Tool
You can find here the main notebook with all our work. 

Here, we were able to learn the key values for each company, the inherent values of each profile, and understand their level of importance for each feature. Finally, the system generates the instructions the candidates has to follow to enhance his profil towards a specific company that he targeted.

### Running the Code

#### Commented Parts
- Some sections are commented out for **testing purposes**, mainly to sample a small portion of datasets for **faster results**.
- Other commented parts are **time-consuming** and **not critical** for our inferences.
- **PART 5 – VERSION 2 (Word Embeddings Extension)** is not necessary as it did not yield satisfying results.

#### Databricks & DBFS Usage
- We **write engineered datasets** in **DBFS (Databricks File System)** to avoid re-running code each time.
- These datasets **cannot be uploaded here**.
- **DBFS write and load cells can be ignored**, as they are inaccessible without our environment.

#### Hugging Face API Requirements
- The code uses **Hugging Face API** to deploy **pretrained models**.
- You must **log in with your own valid personal token**.
- Any token appearing in the notebook has already been **disabled**.

#### Support & Questions
- If you encounter **any issues running the notebook**, feel free to **contact us**.
- We’ll be **glad to help**! 🎯


## Model Interpretability
The model interpretability notebook gives information about the importance of the features in the recruitment process by training for each company a Random Forest model for a binary classification task.

### Running the code
Before you begin, make sure the following prerequisites are met:

* _Databricks Account_: A Databricks account is required to run this project.
* _Databricks Cluster_: A cluster must be configured and started before running the code.

Then, start the cluster and run the code. You may need access to datasets that we could not upload here. 

## Scraping 
This file is about the scraping methods we used to scrape the relevant data from the companies' websites and from Comparably. For the scraping mission, we used the BrightData application that allows us to do high-scaling scraping without being blocked. 

### Running the code
Before you begin, make sure you have a BrightData account. Copy paste the following lines of codes by replacing "username" and "password" by your own username password.

``` 
AUTH = 'username:password'
SBR_WEBDRIVER = f'https://{AUTH}@brd.superproxy.io:9515'
```

## Links

Inception alert 🚨 : You may check our Linkedin post about [On Target](https://www.linkedin.com/posts/tom-bijaoui-2799402ab_machinelearning-bigdata-nlp-activity-7293316200053248000-um9R?utm_source=share&utm_medium=member_ios&rcm=ACoAAEq2IX0Bx9yjkh8KcKEaqRrj5e5HWYojE1c) based on Linkedin Big Data!
