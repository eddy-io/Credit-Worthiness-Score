# Credit-Worthiness-Score
Credit building is really important, but the decision-making process can seem very complex and at times, random. This project is not able to predict a specific card's approval, it aims to model a general approval likelihood and extracts factors that matter most. 

The Question

When someone applies for a credit card, the card issuer reports a hard inquiry to evaluate their credit-worthiness. A hard inquiry then appears on that person’s credit score over a certain period of time. The question here is: Can you estimate the likelihood of getting approved for a credit card before applying, and can you understand which factors matter most? 

Why this Matters 

Credit card applications typically trigger hard inquiries, which can temporarily ding your credit score. For someone building credit from scratch (like a young adult in college), applying blind and getting denied is a lose-lose: a rejection, and a small credit hit that can appear on the next application they attempt. This project aims to turn the guessing game into an informed decision.

	Note: This model is trained on a public/anonymized dataset, not real issuer decisioning data. It’s an educational tool for understanding what generally drives approval. Not a guarantee of any specific card’s outcome. Not financial advice.

Key Findings 
- Credit Score (r = 0.41) and years employed (r = 0.32) showed the clearest positive correlations with approval.
- No prior default, income, and credit score emerged as the top three features driving individual predictions, based on SHAP analysis of the Random Forest model.
- Age was a weaker predictor than expected. Going in, age seemed to matter a lot, but the data does not prove this, ranking age among the weaker correlations. While some of that effect may be absorbed by different indicators, such as years_employed, as they are related.
- Zip code was the only feature with a negative correlation, but it was very weak (r = -0.10), the weakest relationship of any feature measured, not a meaningful driver of approval on its own. 
Dataset
- Source: UCI Credit Approval Dataset (Kaggle)
- Size: 690 rows, 16 columns
- Key features: credit score, years employed, income, debt, prior default status, employment status, age, education level, zip code
- Target variable: Approved(1) / Denied (0)
- Important note: per the dataset's own documentation, all attribute values have been anonymized to protect applicant confidentiality. Numeric features (e.g. income, credit_score) should be read as relative indicators, not real-world units like actual dollars or FICO scores. 
Approach
1. Data Cleaning - loaded the dataset (originally shipped without headers), assigned descriptive column names, standardized missing-value placeholders, imputed missing numeric values with the median and missing categorical values with the mode, and confirmed no columns needed to be dropped.
2. EDA - examined correlations between individual features and approval outcome, compared numeric feature distributions across approved/denied applicants, checked for approval rate by category for categorical features, and screened for multicollinearity (none found above r = 0.6).
3. Modeling - Baseline: Logistic Regression with class_weight='balanced' to address class imbalance. Comparison: Random Forest, same class-weighting approach. 
4. Explainability - used SHAP values on the Random Forest model to see how each feature pushed individual predictions towards approval or denial, not just its overall ranking.
5. Evaluation - accuracy, precision, recall and ROC-AUC, weighting precision and ROC-AUC most heavily given the project's goal: minimizing false "you'll get approved" signals that could lead to an avoidable denial and hard inquiry.

Results

Random Forest was selected as the stronger model - it outperformed Logistic Regression on accuracy, precision, and ROC-AUC, while Logistic Regression held a slight edge on recall. Given this project's goal of helping applicants avoid a denial (a costlier mistake than missing out on an application they'd have been approved for), precision and ROC-AUC were weighted more heavily in this decision. 

What this means for new credit builders
For someone building credit from scratch, this model suggests prioritizing a clean payment history (avoiding any default) and demonstrating steady income before applying. Credit score itself compounds over time, so starting with a secured card or becoming an authorized user (if available) is advisable, letting positive history accumulate before applying for an unsecured card. These factors appear to matter more than any single detail on the application itself. 

Repo Structure
├── data/
│   ├── raw/              # original dataset
│   └── processed/        # cleaned data used for analysis
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── reports/
│   ├── figures/           # saved charts used in this README
│   └── best_model.pkl     # trained Random Forest model
├── requirements.txt
└── README.md

Running 
git clone [https://github.com/eddy-io/Credit-Worthiness-Score.git]
cd [Credit-Worthiness-Score]
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_data_cleaning.ipynb

Limitation and Next Steps
Anonymized data: per the source dataset's documentation, attribute values are disguised to protect applicant confidentiality. This means the model reflects relative feature importance rather than real-world credit scoring standards, (such as FICO or dollar amounts).

Inconsistent categories: education level pushed towards approval in some cases and towards denials in others with the SHAP analysis. Without proper data labels for what each education category represents, it isn't possible to safely conclude that higher education drives approval -- this is flagged rather than overstated. Being that there are no indicators, there is no remedially effective steps to fix this status. 

Zip Code as weak, uncertain signal: treated as a numeric feature for correlation purposes, but zip codes aren't truly ordinal -- very weak negative correlation observed (r= -0.10) may reflect regional patterns specific to this dataset rather than a generalizable predictor. 

The  next steps: are to test on a second, larger dataset (Kaggle/Google) to check whether these findings generalize; explore a card-recommendation angle as a follow up project. 

Tech Stack
Python, pandas, numpy, scikit-learn, SHAP, matplotlib, seaborn, Jupyter

Built By Eduardo
