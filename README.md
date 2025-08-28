# Team1\_IT5006\_Healthcare\_Analytics\_AY2526

> Name of the GitHub repository is as above.

---

## Objective

Analyze the **Diabetes 130-US Hospitals (1999-2008)** dataset to predict hospital readmissions and understand key factors influencing patient outcomes.

### Dataset

The [Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset[^1][^2] is maintained by the [UC Irvine (UCI) Machine Learning Repository](https://archive.ics.uci.edu/). It is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. Which allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

[^1]: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.

[^2]: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records", BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

#### Provenance

It might be helpful to retrace the steps of Strack _et al._ (2014)[^2], since that is where the dataset came from, but not too much as there might be overlaps with Milestone 1:

- Introduction
  - There are protocols to manage hyperglycemia in ICU patients
  - Similar strategies for non-ICU patients are being ramped up
  - Lack of data for a baseline
  - Analyze a large clinical database
    - It is challenging
  - Hypothesis: measurement of HbA1c is associated with a reduction in readmission rates
- Materials and Methods
  - Data Assembly
    - Health Facts database (Cerner Corporation, Kansas City, MO)
    - blah blah ...
    - ... 10 years (1999-2008) ...
    - ... 74,036,643 unique encounters (visits) ...
    - ... 17,880,231 unique patients ...
    - ... data from out-of-network providers is not captured
  - Extraction of the Initial Dataset from the Database
    - Inclusion criteria:
      - It is an inpatient encounter (a hospital admission)
      - It is a diabetic encounter, that is, one during which any kind of diabetes was entered into the system as a diagnosis
      - The length of stay was at least 1 day and at most 14 days
      - Laboratory tests were performed during the encounter
      - Medications were administered during the encounter
    - Filtered down to 101,766 encounters
    - Feature selection performed by experts
    - **This is where the raw dataset came from**
    - 30 days threshold for readmission, based on criteria used by funding agencies
    - Focus:
      - Readmission
      - HbA1c test
      - Change in diabetes management/diabetic medications
    - 4 groups:
      - No HbA1c test performed
      - HbA1c test performed and in normal range
      - HbA1c > 8% with no change in diabetic medications
      - HbA1c > 8% with changes in diabetic medications
  - Preliminary Analysis and the Final Dataset
    - High percentage missing values (weight, payer code, medical specialty (rescued))
    - Multiple inpatient visits for some patients (not statistically independent)
      - Only use first encounter for each patient
    - Remove all encounters that resulted in either discharge to a hospice or patient death
    - Filtered down to 69,984 encounters in final dataset
    - **We don't have to follow their steps**
    - Not mentioned in text: Age grouped into 3 categories, see Figure 2
  - Statistical Methods
    - Unit of analysis = encounter (limited to one per patient)
    - Multivariable logistic regression
    - blah blah ...
- Results and Discussion
  - As a whole (not adjusting for covariates)
    - Measurement of HbA1c was infrequent
    - Not done: 42.5% medication change
    - Done: 55% medication change (P < 0.001); readmission rate 8.7% vs. 9.4% (P = 0.007)
    - Done and > 8%: 65% medication change
  - Adjusting for covariates
    - Readmission vs. HbA1c measurement significantly depends on the primary diagnosis
  - ...
- Conclusions
  - the decision to obtain a measurement of HbA1c for patients with diabetes mellitus is a useful predictor of readmission rates


---

## Milestones

### Phase/Milestone 1: Foundation - Literature Review & Exploratory Data Analysis

- Deliverables:
  - Literature Review Report (2 pages)
  - Exploratory Data Analysis Report (2-3 pages)
  - Interactive Dashboard
    - Built using Streamlit, Tableau Public, or Power BI
    - Submit as live link (include link in submitted report)
- Submission Format:
  - Combined PDF report (Literature Review + EDA) with dashboard link included
  - GitHub repository with all raw code/notebooks
  - File name: `Team1_Milestone1_IT5006_AY2526.pdf`


#### TODO

- Read Strack _et al._ (2014)[^2], to avoid pitfalls
  - Have a different research question
  - Different unit of analysis
  - Different data processing procedures
  - Different analysis procedures
- Choose outcome variable:
  - Choices:
    - Continuous: Length of stay
    - Categorical:
      - Binomial: Readmission, early readmission (will need to binarize `readmission`)
      - Multinomial: Readmission (No, Not early, Early)
  - "Safe" choice: Early readmission
- Explore the dataset
  - Choose data types
    - Will ordinal variables be useful?
      - If not treat all categorical variables as nominal
    - Thinking of using CatBoost, might as well treat all as nominal
    - Engineer method to enforce data types (_e.g._ wrapper function)
  - Repeated encounters for some patients will contribute duplicate data, depending on unit of analysis
- Dashboard ideas
  - Scatterplots might look rather lame for the raw dataset
  - Counts + bar chart/histogram
  - Group by
    - Grouped bar charts
    - Box plots for continuous variables
  - Cross-tabulations
  - Heatmap for measures of association (correlations, Cramer's V)
  - Clustering?


#### Features overview

##### Categories

- Identifiers
  - Patients are not unique in this dataset (duplicate `patient_nbr`)
    - Strategy to aggregate (first, average, ...)
- Demographics
  - `weight` has too many missing values
- Admission Details
  - Can be used to filter out encounters
    - Expired or situation(s) where there is no chance of readmission (if readmission is the target)
- Healthcare Provider
  - `payer_code` is probably irrelevant
- Clinical Metrics
- Diagnoses
  - There are too many categories!
  - Potentially a lot of processing to do
  - Should we distinguish primary vs. secondary?
- Laboratory Results
- Medications
  - Some have little signal, but how to measure "variance" for nominal variables?
- Treatment Changes
- Target Variables


---


### Phase/Milestone 2: Analytics Implementation - Model Building & Evaluation

- File name: `Team1_Milestone2_IT5006_AY2526.pdf`


### Phase/Milestone 3: Integration & Communication - Final Report & Presentation

- File names:
  - `Team1_Milestone3_IT5006_AY2526.pdf`
  - `Team1_Milestone3_IT5006_AY2526.zip`


