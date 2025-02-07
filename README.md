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
2. [Aims & Targets](#aims--targets)  
3. [Project's Files](#projects-files)  
4. [On Target Tool](#on-target-tool)  
5. [Model Interpretability](#model-interpretability)  
6. [Scraping](#scraping)  
7. [Running the Code](#running-the-code)  
   - [Setting Up](#setting-up)  
   - [Commented Parts](#commented-parts)  
   - [Databricks & DBFS Usage](#databricks--dbfs-usage)  
   - [Hugging Face API Requirements](#hugging-face-api-requirements)  
8. [Links](#links)  
9. [Contact & Support](#contact--support)

</details>

---

## Project Overview
In today’s competitive job market, first impressions matter—especially when applying for a job. **On Target** is a data-driven tool designed to help candidates highlight the skills and values most cherished by specific companies. Combining **Statistical** and **Machine Learning** methods, **On Target** personalizes guidelines that can truly make a difference.

**Key Highlights**:
- **NLP** and **Machine Learning** to identify essential traits.
- **Company-specific** feature engineering for tailored profiles.
- **LLMs** (Large Language Models) to generate actionable instructions.

---

## Aims & Targets
- **Accurate Profile Enhancement**: Guide candidates on which strengths to emphasize.
- **Data-Driven Insights**: Provide statistically sound and interpretable feature importance.
- **Scalable Big Data Approach**: Implement robust scraping and large-scale data processing.
- **Customizability**: Adapt instructions for different company cultures and values.
- **Practical Usability**: Provide straightforward notebooks runnable on platforms like Databricks, with Hugging Face integration.

---

## Project's Files
- **Model Interpretability**
  - `Model_Interpretability_Notebook`: Analyzes feature importance via Random Forest models.

- **On Target tool**
  - `On_Target_Notebook`: Core notebook generating guidelines.
  - `Example_generated_instructions`: Illustrative output showing how the tool provides recommendations.

- **Scraping**
  - `Scraping_Comparably`: Code to scrape data from [Comparably](https://www.comparably.com).
  - `Scraping_Company_Websites`: Methods for company website scraping.
  - `Sample_CSV_Scraped_Data`: Example CSV containing raw scraped data.

---

## On Target Tool
**On Target** automatically learns the key values sought by each company and aligns them with candidate profiles, assigning importance to each feature. It then generates **step-by-step instructions** for enhancing a candidate’s profile.

Example Instruction:
> “Highlight your team collaboration skills more prominently; emphasize your adaptability and willingness to learn.”

---

## Model Interpretability
The **Model Interpretability Notebook**:
- Trains **Random Forest** models for **binary classification** per company.
- Measures **Feature Importance** and displays how each feature influences hiring decisions.
- Includes **Statistical tests** to verify the significance of these features.

These insights explain *why* certain changes to a candidate’s profile could help.

---

## Scraping
We used **BrightData** for large-scale scraping of:
- **Company Websites**: Mission statements, culture pages, job postings.
- **Comparably**: Employee reviews and ratings, revealing key recurring values.

### Running the Scraping Code
1. **BrightData Account**  
   Replace `"username"` and `"password"`:
   ```python
   AUTH = 'username:password'
   SBR_WEBDRIVER = f'https://{AUTH}@brd.superproxy.io:9515'
