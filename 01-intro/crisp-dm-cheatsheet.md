# CRISP-DM Cheatsheet

This README provides a quick reference for the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, a widely-used methodology in data science and analytics. It breaks down each phase of the CRISP-DM process and offers key questions, rubrics, and tips to guide you through data science projects.

---

## 1. Business Understanding
**Goal**: Define the project’s objectives from a business perspective.

- **Key Questions**:
  - What is the problem we’re solving?
  - Who are the stakeholders, and what are their needs?
  - What success metrics define project success (accuracy, revenue impact)?
  - What are the constraints (time, resources, data availability)?

- **Rubrics**:
  - Clear, actionable project objectives.
  - Well-defined problem and scope.
  - Stakeholder interviews for better context.
  - Success criteria tied to business outcomes.

---

## 2. Data Understanding
**Goal**: Gather, explore, and assess the quality and structure of the data.

- **Key Questions**:
  - What data do I have access to?
  - What is the data source and its reliability?
  - How is the data structured (types, features, relationships)?
  - Are there missing values or anomalies in the data?

- **Rubrics**:
  - Exploratory Data Analysis (EDA) (summary stats, histograms, correlations).
  - Identifying outliers, missing values, and other data quality issues.
  - Visualizations to understand distributions and relationships (scatter plots, box plots).
  - Data profiling to determine the data’s fitness for modeling.

---

## 3. Data Preparation
**Goal**: Clean, transform, and structure the data for modeling.

- **Key Questions**:
  - How do I handle missing values (impute, remove, or leave as is)?
  - Should I engineer new features (e.g., age from birthdate, activity scores)?
  - Do I need to normalize or scale numerical features?
  - Are categorical variables in the right format (one-hot encoding, label encoding)?

- **Rubrics**:
  - Handle missing data (imputation, removal).
  - Feature engineering (creating meaningful features).
  - Data transformation (logarithmic transformation, standardization).
  - Data splitting (training, validation, and test sets).
  - Handle imbalanced data if needed (SMOTE, class weighting).

---

## 4. Modeling
**Goal**: Build predictive models that solve the problem.

- **Key Questions**:
  - What models (algorithms) should I try? (e.g., regression, classification, clustering)
  - What’s the best performance metric to use? (accuracy, precision, recall, RMSE)
  - How will I prevent overfitting or underfitting?
  - How do I tune the model (e.g., grid search, random search)?

- **Rubrics**:
  - Choose appropriate algorithms (e.g., Random Forest for classification, Linear Regression for continuous data).
  - Hyperparameter tuning (e.g., grid search, random search).
  - Model evaluation using cross-validation.
  - Track performance metrics (e.g., ROC-AUC, F1 score, confusion matrix).

---

## 5. Evaluation
**Goal**: Ensure the model is ready for deployment by validating its business impact.

- **Key Questions**:
  - Does the model meet the business success criteria?
  - Is the model explainable and interpretable?
  - What are the strengths and weaknesses of the model?
  - Are there any potential risks or ethical concerns?

- **Rubrics**:
  - Compare model performance against baseline models.
  - Analyze residuals and feature importance.
  - Assess model stability (using various test sets or cross-validation).
  - Check model for biases (e.g., fairness, transparency).
  - Evaluate model against business metrics (e.g., accuracy, revenue impact).

---

## 6. Deployment
**Goal**: Put the model into production and monitor its performance.

- **Key Questions**:
  - How will the model be deployed (real-time, batch processing)?
  - How will the model be monitored post-deployment?
  - Are there any plans for model retraining as new data comes in?
  - How will results be communicated to stakeholders?

- **Rubrics**:
  - Deploy the model to production (e.g., API, batch system).
  - Set up model monitoring (performance tracking, drift detection).
  - Retraining schedule (how often and based on what triggers).
  - Clear communication to stakeholders (dashboards, reports).
  - Document deployment process for reproducibility.

---

## Quick Tips:
- **Iterate**: CRISP-DM is iterative—don’t hesitate to loop back to a previous phase when you learn new things.
- **Data Quality > Model Complexity**: Prioritize good quality, well-prepared data over complex models.
- **Stakeholder Collaboration**: Keep in close contact with stakeholders throughout the process for feedback and alignment.

---

## Visual Aid for Phases:
Here’s a quick representation of CRISP-DM:

1. **Business Understanding**  
2. **Data Understanding**  
3. **Data Preparation**  
4. **Modeling**  
5. **Evaluation**  
6. **Deployment**  
(↺ Looping back as needed)

