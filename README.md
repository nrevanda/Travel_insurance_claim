# Predicting Travel Insurance Claims: A Machine Learning Approach with Cost Analysis
[Streamlit](https://travelinsuranceclaim.streamlit.app/)

## Business Understanding

This section outlines the business context, the problems the company faces, and the goals of this machine learning project.

### 1. Background
A travel insurance company provides coverage for travelers both domestically and internationally. The premium for each policy is determined by factors like trip duration, destination, and desired coverage. Currently, the company assesses customer risk using general rules, which can be inaccurate in predicting the actual likelihood of a claim.

### 2. Problem Statement
The company wants to more accurately identify policyholders who are likely to file a claim. The current rule-based system leads to two primary issues:

* **Financial Loss**: Setting premiums too low for high-risk customers can lead to significant financial losses when they file a claim.
* **Lost Business Opportunities**: Setting premiums too high for low-risk customers may discourage them from purchasing a policy, resulting in lost revenue.

This project aims to build a predictive model to help the company set fairer premiums, maintain financial stability, and ensure all legitimate claims can be paid smoothly. **The model is intended for risk assessment, not for rejecting customers.**

### 3. Project Goals
The main objectives of this project are:
1.  **Build a machine learning model** to predict the probability that a policyholder will file a travel insurance claim.
2.  **Identify key factors (features)** that strongly influence claim submissions to understand customer risk patterns better.
3.  **Support business decisions**, such as more accurate premium pricing, targeted marketing strategies, and overall risk management.
4.  **Increase operational efficiency** by minimizing unexpected claims and maximizing profitability through risk-based customer segmentation.

### 4. Business Impact of Predictions
To align the model's performance with business goals, it's crucial to understand the cost of correct and incorrect predictions. The target variable is `Claim`, where **1 = Claim Filed** and **0 = No Claim**.

| | **Actual: Claim** | **Actual: No Claim** |
| :--- | :--- | :--- |
| **Predicted: Claim** | **True Positive (TP)**<br>✔ Correctly identifies a high-risk customer.<br>✔ Premium is adjusted, mitigating potential loss.<br>✔ **Ideal outcome for high-risk cases.** | **False Positive (FP)**<br>✘ Incorrectly flags a low-risk customer.<br>✘ The company must allocate **$65,000 USD** in standby funds.<br>✘ **Opportunity Cost:** **$3,250 USD/year** per policy (assuming a 5% annual return if invested in bond or time deposit). |
| **Predicted: No Claim** | **False Negative (FN)**<br>✘ Fails to identify a high-risk customer.<br>✘ **This is the highest-risk scenario.**<br>✘ The company faces an unexpected loss of up to **$65,000 USD** per claim.<br>✘ **An FN is ~20 times more costly than an FP.** | **True Negative (TN)**<br>✔ Correctly identifies a low-risk customer.<br>✔ Standard premium applies.<br>✔ **Ideal outcome for low-risk cases.** |

### 5. Key Implication: Prioritizing Recall
Based on the cost analysis:
* A **False Negative (FN)** is the most damaging error, leading to direct and substantial financial loss.
* A **False Positive (FP)** results in an opportunity cost but is significantly less severe than an FN.

Therefore, the primary goal of the model is to **minimize False Negatives**. This means the most critical evaluation metric for this project is **Recall**, as it measures the model's ability to correctly identify all actual positive (claim) cases.

## Dataset

This project utilizes a historical dataset of travel insurance policies sourced from the **Kaggle** platform.

* **Dataset Link:** [Travel Insurance Claim Prediction](https://www.kaggle.com/datasets/mhdzahier/travel-insurance)
* **Description:** The dataset consists of **44,328 rows** and **11 columns**, where each row represents a unique insurance policy.

### Data Dictionary

| Attribute | Description |
| :--- | :--- |
| **Agency** | Name of the insurance agency. |
| **Agency Type** | Type of travel insurance agency (e.g., Airlines, Travel Agency). |
| **Distribution Channel** | The distribution channel of the agency (e.g., Online, Offline). |
| **Product Name** | The name of the travel insurance product. |
| **Gender** | The gender of the policyholder. |
| **Duration** | The duration of the trip in days. |
| **Destination** | The destination of the trip. |
| **Net Sales** | The amount of sales for the insurance policy. |
| **Commission (in value)** | The commission received by the agency. |
| **Age** | The age of the policyholder. |
| **Claim** (Target) | The claim status (`1` = Claim Filed, `0` = No Claim). |

## Technology Stack

This project leverages a range of Python libraries for data analysis, machine learning, and visualization.

* **Data Analysis & Manipulation:**
    * `numpy`: For fundamental numerical operations.
    * `pandas`: For data handling, cleaning, and manipulation.

* **Data Visualization:**
    * `matplotlib` & `seaborn`: For creating static, animated, and interactive visualizations.
    * `missingno`: For visualizing missing data patterns.

* **Machine Learning & Modeling:**
    * `scikit-learn`: The core library for preprocessing, building baseline models, and evaluation.
    * `category-encoders`: For advanced categorical feature encoding techniques.
    * `imbalanced-learn`: For handling imbalanced datasets (e.g., using SMOTE).
    * `xgboost` & `lightgbm`: For implementing powerful gradient boosting models.

* **Model Optimization & Interpretability:**
    * `hyperopt`: For hyperparameter tuning and optimization.
    * `shap`: For explaining the output of machine learning models and feature importance.
 
## Project Workflow

This project followed a systematic machine learning workflow:

1.  **Data Cleaning:** Handled missing values, duplicates, and inconsistent data types to ensure data quality. Special attention was given to the `Age`, `Duration`, `Gender` and `Net_sales` columns.
2.  **Exploratory Data Analysis (EDA):** Performed in-depth analysis to understand data distributions, identify outliers, visualize relationships between features and the `Claim` target, and generate initial hypotheses.
3.  **Feature Engineering & Preprocessing:** Prepared the data for modeling. Key steps included:
    * Encoding categorical features (e.g., `Destination`, `Agency Type`) using `One-Hot Encoder` and `BinaryEncoder`.
    * Scaling numerical features (e.g., `Age`, `Net Sales`) using `RobustScaler`.
    * Handling the imbalanced dataset using `RandomOverSampler`, `RandomUnderSampler` and `SMOTE` from the `imbalanced-learn` library.
4.  **Model Training:** Trained and compared several classification algorithms. The models tested include `Logistic Regression`, `Random Forest`, `XGBoost`, `LightGBM`, and many more. However, the models then filtered based on their performance to get the best model for this case
5.  **Hyperparameter Tuning:** Optimized the best-performing model using `Hyperopt` and `GridSearchCV` to find the ideal combination of parameters that maximizes the **Recall** score.
6.  **Model Evaluation:** Assessed the final model on a held-out test set. The evaluation focused on the **Recall** metric due to the high business cost of False Negatives. A confusion matrix was used and to analyze the cost model.
7.  **Model Interpretation:** Analyzed the beta coefficients of the linear model and utilized **SHAP** values for the final model to explain predictions and identify the most influential features driving the likelihood of a claim.

## Model Evaluation & Results

Several classification models were trained and evaluated. While the primary technical metric was **Recall** (to minimize high-cost False Negatives), the final model selection was based on the **Total Business Cost**, which combines the financial impact of both False Positives and False Negatives.

The cost is calculated as:
* **False Negative (FN) Cost** = `Number of FNs` × `$65,000`
* **False Positive (FP) Cost** = `Number of FPs` × `$3,250`
* **Total Cost** = `FN Cost` + `FP Cost`

### Model Performance Comparison
The table below summarizes the performance of the final, tuned models on the test set.

| Model | Recall (Test) | False Negatives (FN) | False Positives (FP) | Total Business Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.81 | 24 | 1710 | **$7,117,500** |
| **LightGBM** | 0.79 | 27 | 1891 | $7,900,750 |
| **XGBoost** | 0.89 | 14 | 4209 | $14,589,250 |

### Confusion Matrices
The images below visualize the trade-offs each model makes after hyperparameter tuning.

**Logistic Regression**
<img width="1281" height="538" alt="image" src="https://github.com/user-attachments/assets/d3cf897f-0547-4999-9ed9-101f1e571bf7" />


**LightGBM**
<img width="1281" height="538" alt="image" src="https://github.com/user-attachments/assets/148d4f7b-2cac-4673-bd2e-e1e1f0873c33" />



**XGBoost**
<img width="1281" height="538" alt="image" src="https://github.com/user-attachments/assets/21b5b717-a024-4557-a7fe-4f11818639f3" />



### Final Model Selection: Logistic Regression

Despite XGBoost achieving the highest Recall (0.89), **Logistic Regression was selected as the final production model.**

The justification is as follows:
1.  **Superior Cost-Efficiency:** With a total business cost of **$7,117,500**, Logistic Regression is significantly more economical than both LightGBM ($7,900,750) and XGBoost ($14,589,250). It provides the best financial outcome for the company.

2.  **Balanced Trade-Off:** While advanced models like XGBoost excel at minimizing False Negatives, they overcompensate by drastically increasing False Positives. This leads to an explosion in opportunity costs, making them financially impractical for this highly imbalanced dataset (98.4% no-claim vs. 1.6% claim). Logistic Regression achieves a much more balanced and acceptable trade-off.

3.  **Simplicity and Interpretability:** In a highly regulated domain like insurance, model transparency is critical. The simplicity and inherent interpretability of Logistic Regression (e.g., via its coefficients) are significant advantages for explaining decisions to business stakeholders and meeting regulatory requirements.

---

## Conclusion & Recommendation

## Conclusion

This project successfully developed a machine learning model to predict the likelihood of travel insurance claims. After a comparative analysis of Logistic Regression, LightGBM, and XGBoost, **Logistic Regression** was selected as the final model. It provides the optimal balance between strong predictive performance (**achieving a Recall of 0.81 on test data**) and superior **financial cost-efficiency**.

While advanced ensemble models achieved higher recall, they did so by producing an excessive number of False Positives, which would lead to impractical increases in operational costs from allocating standby funds. The cost analysis confirmed that a balanced trade-off is critical, and Logistic Regression proved most effective in achieving this.


## Recommendations

Based on the model's findings, the following recommendations are proposed to drive business value and guide future development:

### For Business Strategy
1.  **Implement Risk-Based Premium Pricing**
    Integrate the model's risk score into the pricing workflow to assign fairer and more accurate premiums:
    * **High-Risk Customers:** Apply an adjusted, higher premium to cover the anticipated risk and maintain profitability.
    * **Low-Risk Customers:** Offer more competitive pricing to attract a larger customer base and increase market share.

2.  **Enhance Customer Segmentation**
    Use the model's output to segment policyholders by their predicted claim probability. This enables more targeted strategies for growth and risk management.

3.  **Optimize Operational Risk Management**
    Align the company's financial reserves with the model's predictions. By proactively identifying high-risk policies, the company can allocate standby funds more accurately and reduce financial shocks from unexpected claims.

---
### For the Model Lifecycle
1.  **Continuous Monitoring and Maintenance**
    Continuously track the model's performance in production, focusing on **Recall**, False Negative counts, and the overall business cost metric. The model should be periodically retrained with new data to adapt to evolving customer behaviors and market trends.

2.  **Future Enhancements**
    * Explore advanced techniques such as **cost-sensitive learning** to optimize the model directly for financial outcomes.
    * Enrich the dataset with new features (e.g., trip purpose, travel frequency) to potentially improve predictive power.
    * Experiment with hybrid models (e.g., stacking Logistic Regression with ensemble methods) to further refine risk calibration.

---
### For the Claim Validation Process
Leverage the model's risk score to create a tiered claim validation process, improving both efficiency and accuracy.

1.  **Prioritize High-Risk Claims for Review**
    Claims filed by policyholders who were initially flagged as high-risk by the model should be automatically routed for a more thorough review process. This review could include:
    * Verifying documentation (tickets, police/hospital reports).
    * Cross-checking information with relevant third parties.
    * Reviewing the customer's claim history for suspicious patterns.

2.  **Streamline Low-Risk Claims**
    Claims from customers predicted as low-risk can be processed through a faster, more automated validation track, improving customer satisfaction and operational efficiency.

> **Note:** Any enhanced validation process must be conducted with high regard for customer privacy and regulatory compliance.

## 👨‍💻 Author

* **Naufal Fajar Revanda**
    * E-mail: nrevanda@gmail.com
    * LinkedIn: [Naufal ](https://www.linkedin.com/in/naufalrevanda/)
