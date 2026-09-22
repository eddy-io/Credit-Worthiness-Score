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
