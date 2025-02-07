<h1 align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Technion_logo.svg" alt="Technion Logo" height="100">
  <br>
  On Target: Emphasize Your Best Qualities
</h1>

<p align="center">
  <em>"Not to Lie, but to Emphasize"</em>
</p>

<p align="center">
  <strong>Technion - Israel Institute of Technology</strong> <br>
  Faculty of Data Science and Decisions
</p>

---

<details open>
<summary><strong>Table of Contents</strong> ⚙️</summary>

1. [Project Overview](#project-overview)  
2. [Implementing Methods](#implementing--methods)  
3. [Project's Files](#projects-files)  
4. [On Target Tool](#on-target-tool)  
5. [Model Interpretability](#model-interpretability)  
6. [Scraping](#scraping)  
7. [Links](#links)  

</details>

---

## Project Overview
Emphasizing the right core values are crucial in the business world, especially when applying for a job. It can be challenging to determine which values to emphasize, how to subtly convey them, establish a clear order of priorities, and recognize a company’s recruitment patterns. That's why we developed On Target, a  big data tool that aims to give it a solution. This system generates tailored guidelines that enable candidates to highlight the skills and values most valued by the specific company they are targeting using Statistical and Machine Learning methods.

## Implementing Methods

* Data Preprocessing
* Feature Engineering
* Pre-trained Models from Hugging Face
* Statistical Tests for Significance Inference
* Machine Learning Model for Model Interpretability
* NLP Keyword Extraction Techniques
* LLM to Generate Instructions

---

## Project's Files
- **Model Interpretability**
  - `Model_Interpretability_Notebook`: Analyzes feature importance via Random Forest Model Interpretability Techniques.

- **On Target tool**
  - `On_Target_Notebook`: Core notebook generating guidelines.
  - `example_generated_instructions`: output example showing how the tool provides instructions.

- **Scraping**
  - `Scraping_Comparably`: Code to scrape data from [Comparably](https://www.comparably.com).
  - `Scraping_Company_Websites`: Methods for company website scraping.
  - `Sample_CSV_Scraped_Data`: Example CSV containing raw scraped data.

---

## On Target Tool
Here, we were able to learn the key values for each company, the inherent values of each profile, and understand their leel of importance for each feature. Finally, the system generates the instructions the candidates has to follow to enhance his profil towards a specific company that he targeted.

Example Instruction:
> “Highlight your team collaboration skills more prominently; emphasize your adaptability and willingness to learn.”

---

## Running the On Target Notebook

- **Commented Parts**  
  - Some sections are commented out for *testing purposes*, mainly to sample a small portion of datasets for **faster results**.  
  - Other commented parts are *time-consuming* and *not critical* for our inferences.  
  - **PART 5 – VERSION 2 (Word Embeddings Extension)** is not necessary as it did not yield satisfying results.

- **Databricks & DBFS Usage**  
  - We *write engineered datasets* in **DBFS (Databricks File System)** to avoid re-running code each time.  
  - These datasets **cannot be uploaded here**.  
  - **DBFS write and load cells can be ignored**, as they are inaccessible without our environment.

- **Hugging Face API Requirements**  
  - The code uses *Hugging Face API* to deploy **pretrained models**.  
  - You must *log in with your own valid personal token*.  
  - Any token appearing in the notebook has already been *disabled*.

- **Support & Questions**  
  - If you encounter *any issues running the notebook*, feel free to **contact us**.  
  - We’ll be *glad to help*! 🎯

---

## Model Interpretability
The **Model Interpretability Notebook**:
- Trains **Random Forest** models for **binary classification** per company.
- Measures **Feature Importance** and displays how each feature influences hiring decisions.

Make sure the following prerequisites are met:

•⁠  ⁠Databricks Account: A Databricks account is required to run this project.
•⁠  ⁠Databricks Cluster: A cluster must be configured and started before running the code.

Then, start the cluster and run the code.

You may need to access patterns_df to run the notebook. We could not load it here.

---

## Scraping
We used **BrightData** for large-scale scraping of:
- **Company Websites**: 'about us' section or similar.
- **Comparably**: references of companies, revealing informations including company's culture.

This file is about the scraping methods we used to scrape the relevant data from the companies' websites and from Comparably. For the scraping mission, we used the BrightData application that allows us to do high-scaling scraping without being blocked. 


### Running the Scraping Code
Before you begin, make sure you have a BrightData account. Copy paste the following lines of codes by replacing "username" and "password" by your own username password.

1. **BrightData Account**  
   Replace `"username"` and `"password"`:
   ```python
   AUTH = 'username:password'
   SBR_WEBDRIVER = f'https://{AUTH}@brd.superproxy.io:9515'


---

## Links

Inception alert 🚨 : You may check our Linkedin post about [On Target](https://www.linkedin.com/posts/tom-bijaoui-2799402ab_machinelearning-bigdata-nlp-activity-7293316200053248000-um9R?utm_source=share&utm_medium=member_ios&rcm=ACoAAEq2IX0Bx9yjkh8KcKEaqRrj5e5HWYojE1c) based on Linkedin Big Data!
