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
    - [Commented Parts](#commented-parts)
    - [Databricks & DBFS Usage](#databricks--dbfs-usage)
    - [Hugging-Face API Requirements](#hugging-face-api-requirements)
8. [Links](#links)
9. [Contact & Support](#contact--support)

</details>

---

## Project Overview
> In today’s competitive job market, first impressions matter—especially when applying for a job. **On Target** is a data-driven tool designed to help candidates highlight the skills and values most cherished by specific companies. Combining **Statistical** and **Machine Learning methods**, **On Target** personalizes guidelines that can truly make a difference.

**Key Highlights**:
- Uses **NLP** and **Machine Learning** to identify essential traits.
- Integrates **company-specific** features for tailored profiles.
- Leverages **LLMs** (Large Language Models) to generate instructions.

---

## Aims & Targets
- **Accurate Profile Enhancement**: Guide candidates on which strengths to emphasize.
- **Data-Driven Insights**: Provide statistically sound and interpretable feature importance.
- **Scalable Big Data Approach**: Implement robust scraping and large-scale data processing.
- **Customizability**: Adapt instructions for different companies’ cultures and values.
- **Practical Usability**: Make the tool easy to run on platforms like Databricks, integrating smoothly with Hugging Face APIs.

---

## Project's Files
Below is a quick overview of the main files and folders:

- **Model Interpretability**
  - **Model_Interpretability_Notebook**: Analyzes feature importance via Random Forest models.
  
- **On Target tool**
  - **On_Target_Notebook**: Core notebook generating tailored guidelines.
  - **Example generated instructions**: Illustrative output showcasing how the tool provides recommendations.

- **Scraping**
  - **Scraping_Comparably**: Code to scrape data from [Comparably](https://www.comparably.com).
  - **Scraping_Company_Websites**: Scraping methods for company websites.
  - **Sample_CSV_Scraped_Data**: Example CSV with raw data from scraping.

---

## On Target Tool
**On Target** automatically learns key values for each company and matches them with candidate profiles, allowing you to see how important each feature is for hiring decisions. Finally, it generates **step-by-step guidelines** for profile enhancement.

*Example Output*:
> “Highlight your team collaboration skills more prominently; emphasize your adaptability and willingness to learn.”

---

## Model Interpretability
Our **Model Interpretability Notebook** provides:
- **Binary classification** approach using **Random Forest** models.
- Detailed **Feature Importance** metrics.
- **Statistical tests** to confirm the significance of each feature.
  
This helps you understand *why* the system recommends certain profile changes.

---

## Scraping
We used **BrightData** for large-scale scraping, focusing on:
- **Company Websites**: Gathering mission statements, culture pages, and job descriptions.
- **Comparably**: Mining employee reviews and ratings to identify recurring keywords and values.

### Running the Scraping Code
1. **BrightData Account Required**  
   Replace `"username"` and `"password"` in the following snippet with your own:
   ```python
   AUTH = 'username:password'
   SBR_WEBDRIVER = f'https://{AUTH}@brd.superproxy.io:9515'
