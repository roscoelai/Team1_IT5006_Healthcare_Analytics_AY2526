# Team1\_IT5006\_Healthcare\_Analytics\_AY2526

## Naming conventions

### GitHub repository
- `Team1_IT5006_Healthcare_Analytics_AY2526`

### Files
- `Team1_Milestone1_IT5006_AY2526.pdf`
- `Team1_Milestone2_IT5006_AY2526.pdf`
- `Team1_Milestone3_IT5006_AY2526.pdf`
- `Team1_Milestone3_IT5006_AY2526.zip`


---

## Objective

Analyze the **Diabetes 130-US Hospitals (1999-2008)** dataset to predict hospital readmissions and understand key factors influencing patient outcomes.


### Dataset

The [Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset[^1][^2] is maintained by the [UC Irvine (UCI) Machine Learning Repository](https://archive.ics.uci.edu/). It is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. Which allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

[^1]: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.

[^2]: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records", BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.


#### Information

- 101,766 records of hospitalized patients diagnosed with diabetes
- No recommended data splits
- Inclusion criteria:
  - It is an inpatient encounter (a hospital admission)
  - It is a diabetic encounter, that is, one during which any kind of diabetes was entered into the system as a diagnosis
  - The length of stay was at least 1 day and at most 14 days
  - Laboratory tests were performed during the encounter
  - Medications were administered during the encounter


#### Features overview

##### Categories

(Not sure where these came from, but they are in the project brief)

- Identifiers
  - IDs are usually not of interest, but the patients might not be unique in this dataset
- Demographics
  - `race` and `weight` have missing values, `weight` might have to be dropped
- Admission Details
- Healthcare Provider
  - `payer_code` and `medical_specialty` have missing values
- Clinical Metrics
- Diagnoses
  - `diag_1`, `diag_2`, and `diag_3` have missing values
- Laboratory Results
- Medications
- Treatment Changes
- Target Variables

While only `readmitted` is categorized under "Target Variables", the project brief also suggests `time_in_hospital` as an outcome variable for regression.

##### Data types

- All data types are either integer or categorical
- `encounter_id`, `patient_nbr`, `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` are integers that should not be treated as numbers.
- Ordinal: `age`, `max_glu_serum`, `A1Cresult`, Medications, and possibly `readmitted` (but might need to binarize, unless doing multinomial classification?)


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

#### TODO
- [ ] What's the goal/story?
  - Save hospital costs
  - Improve patient outcomes
  - Recommend SOP to do certain lab tests or procedures?
- [ ] Handle data types
  - Decide what data type each variable should be (integer, ordinal, or nominal)
  - For ordinal variables, decide on the order of categories
    - Moving forward think about how ordinal variables will be handled (vs. just treating them as nominal)
  - Explicitly set data types when reading raw data, do not let the CSV reader decide!
  - Create a wrapper function to read the raw data
  - Optional: Save in a format with schema preservation
- [ ] Decide what to display in the interactive dashboard
  - [ ] Check if `patient_nbr` is unique, if not, find out distribution of number of hospital admissions per patient
  - [ ] Summary statistics (without too many decimal places)?
  - [ ] Bar charts for categorical variables
    - [x] Interactive visualizations might not respect ordered categories without some workarounds 
  - [ ] Histograms/Box plots/KDE plots for integers
  - [ ] Pair-wise plots?
  - [ ] Group by: `readmitted`, `max_glu_serum`, `A1Cresult`, ...
  - [ ] Clustering? PCA? CATPCA? MCA?


### Phase/Milestone 2: Analytics Implementation - Model Building & Evaluation

### Phase/Milestone 3: Integration & Communication - Final Report & Presentation



