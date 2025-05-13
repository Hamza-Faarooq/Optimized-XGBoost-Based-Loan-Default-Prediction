# Optimized-XGBoost-Based-Loan-Default-Prediction

# Credit Card Behaviour Score

A machine learning project developed for **Convolve 3.0: A Pan IIT AI/ML Hackathon** by Team "pip install DebugThugs" (Muhammad Hamza, Ayush Goel, Arnav Singh).

## Project Overview

This project aims to build a robust **Behaviour Score** model for Bank A to predict the probability of credit card defaults. By leveraging a large, diverse dataset of credit card accounts, the model enables the bank to proactively identify at-risk customers, optimize credit decisions, and enhance portfolio profitability.

## Motivation

Credit card defaults impose significant financial and reputational risks on banks. Traditional credit scoring models often fail to capture the complexities of modern customer behavior. Our goal was to develop a predictive, data-driven solution that:
- Accurately estimates the likelihood of default for each customer
- Enables targeted interventions and risk mitigation
- Improves customer relationships and regulatory compliance
- Optimizes the bank’s portfolio performance[1].

## Dataset

- **Source:** Historical data from Bank A
- **Size:** 96,806 credit card accounts (training), 41,792 accounts (validation)
- **Features:** On-us attributes (credit limits, history), transactional data (spending, payments), bureau tradeline characteristics, bureau enquiry metrics
- **Target Variable:** `bad_flag` (0 = non-default, 1 = default)
- **Preprocessing:**
  - Removal of features with excessive missing data
  - Imputation of missing values with mode
  - Feature scaling (MinMaxScaler)
  - Addressed class imbalance via oversampling and undersampling[1].

## Solution Approach

1. **Data Preprocessing**
   - Cleaned, transformed, and engineered features to ensure data quality and extract behavioral insights (e.g., credit utilization, payment history).
2. **Model Selection & Training**
   - Evaluated multiple algorithms: Logistic Regression, SVM, Decision Trees, Random Forests, Gradient Boosting
   - Hyperparameter tuning and cross-validation for optimal performance
3. **Model Evaluation**
   - Used metrics: AUC-ROC, F1-score, accuracy, precision, recall
   - Feature importance analysis for interpretability
4. **Deployment**
   - Generated default probabilities for each account in the validation set
   - Outputs designed for integration into risk management workflows[1].

## Technologies Used

- **Programming Language:** Python
- **Libraries:** XGBoost, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn[1].

## Key Results

| Metric         | Training | Testing  |
|----------------|----------|----------|
| AUC-ROC        | 0.8980   | 0.810    |
| F1-Score       | 0.9023   | 0.241    |
| Accuracy       | 0.9324   | 0.843    |
| MAE            | 0.222    | 0.234    |

**Insights:**
- High credit utilization and history of late payments are strong predictors of default.
- Recent credit bureau inquiries signal increased risk[1].

## Challenges

- **Data Quality:** Missing values, outliers, and inconsistencies required extensive cleaning.
- **Class Imbalance:** Addressed through sampling techniques to ensure fair model training.
- **Model Interpretability:** Balanced accuracy with explainability for regulatory compliance[1].

## Future Scope

- Integrate external data sources (macroeconomic, alternative data)
- Explore advanced ML (deep learning, ensemble methods)
- Implement explainable AI (e.g., SHAP, LIME)
- Real-time risk scoring and continuous model monitoring[1].

## How to Run

1. **Clone the Repository**
  
2. **Install Dependencies**

3. **Prepare Data**
- Place the dataset in the `data/` directory as specified in the code.

4. **Train the Model**

5. **Evaluate & Predict**

## Conclusion

This Behaviour Score model empowers banks to proactively manage credit risk, minimize losses, and enhance customer satisfaction. The framework is robust, interpretable, and adaptable to evolving financial landscapes[1].

---

*For questions or contributions, please open an issue or submit a pull request.*

