<p align="center">
  <img src="https://img.shields.io/badge/Student%20Admission%20Prediction-ML%20Project-blueviolet?style=for-the-badge&logo=python&logoColor=white" />
</p>

<p align="center">
  <b>Data Analysis • Machine Learning • PCA • Model Comparison</b>
</p>

🎓 ML-Based Admission Prediction System
---
A Complete Data Cleaning, EDA, Feature Engineering & Machine Learning Project

This project analyzes a large student admissions dataset containing 53,644 rows and 26 features, performs full data cleaning + preprocessing + EDA, and builds multiple machine learning models to predict whether a student will be admitted based on their academic profile.

The project is implemented in Python using Jupyter Notebook.

---
📌 Project Overview

The goal of this project is to:

Understand student academic and profile patterns

Clean a highly unstructured dataset with many missing values

Encode categorical features

Build machine learning models to predict admission chances

Evaluate and compare multiple ML algorithms

Improve performance using PCA and hyperparameter tuning

---

🗂️ Dataset Details

Rows: 53,644

Columns: 26

Target variable: admit (1 = admitted, 0 = rejected)

The dataset contains:

Academic scores (CGPA, GRE, GMAT)

Experience (internship, work experience, research experience)

Program details (major, specialization, department)

TOEFL score, essay score

Publications and profile details

---
🛠️ Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Label Encoding

PCA

ML Models:

Logistic Regression

Decision Tree

Random Forest

SVM

RandomizedSearchCV

---
🔧 Data Cleaning
✔ Handling Missing Values

Missing values were handled based on feature type:

Numerical features → filled using mean

Categorical features → filled using mode

Mixed features → cleaned individually

Example:
```
order.major.fillna(order['major'].mode()[0], inplace=True)
order.specialization.fillna(order["specialization"].mode()[0], inplace=True)
order.toeflScore.fillna(order["toeflScore"].mean(), inplace=True)
order.greV.fillna(order["greV"].mean(), inplace=True)
order.ugCollege.fillna(order["ugCollege"].mode()[0], inplace=True)
```
---
🔡 Feature Encoding

Categorical columns were encoded using LabelEncoder:
```
le = LabelEncoder()
order[order.select_dtypes(include='object').columns] = \
      order[order.select_dtypes(include='object').columns].apply(le.fit_transform)
```
---
📊 Exploratory Data Analysis (EDA)

Performed EDA to understand:

Distribution of academic scores

Relationship between GRE/CGPA and admission rate

Correlations between numerical features

Patterns in specialization, department, and program choices

Outlier detection

---
🧠 Machine Learning Models Used
1️⃣ Logistic Regression

Baseline classification model

Useful for linear patterns

2️⃣ Decision Tree Classifier

Captures non-linear relationships

Easy to interpret

3️⃣ Random Forest Classifier

High accuracy

Handles mixed feature types

Reduces overfitting

4️⃣ Support Vector Machine (SVM)

Effective in high-dimensional data

Strong classification performance

---
🧪 Model Evaluation

Evaluation metrics include:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC-AUC

Example:
```
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)

```
---
🧬 PCA (Principal Component Analysis)

PCA was used to:

Reduce dimensionality

Improve computational efficiency

Capture maximum variance

Visualize feature contribution

---
⚙ Hyperparameter Tuning

Used RandomizedSearchCV for Random Forest optimization:
```
RF_random = RandomizedSearchCV(
    estimator=RF_model_random,
    param_distributions=search_RF,
    random_state=42,
    cv=5,
    n_iter=10
)
```
📈 Key Insights

CGPA, GRE, and TOEFL strongly influence admission outcomes

Publications and research experience improve acceptance chances

Random Forest and SVM outperform baseline models

PCA highlighted the most important feature combinations

Cleaned dataset significantly improved prediction accuracy

---
📂 Project Structure
```
├── order.ipynb
├── original_data.csv
├── README.md
└── model_outputs/
```
---
## 📊 Model Comparison (Summary)

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | ~73%     | ~73%      | ~73%   | ~73%     |
| Decision Tree       | ~69%     | ~69%      | ~69%   | ~69%     |
| Random Forest       | ~74%     | ~74%      | ~74%   | ~74%     |
| KNN                 | ~54%     | ~55%      | ~72%   | ~62%     |

**🏆 Best Model → Random Forest Classifier**
---

🚀 Conclusion

This project showcases a full ML pipeline for building an admission prediction system:

Data Cleaning

Preprocessing & Encoding

Exploratory Data Analysis

PCA & Feature Engineering

Multiple ML Models

Hyperparameter Tuning

Evaluation & Comparison

- EDA revealed key trends in academic scores, experience, and test results affecting admissions.  
- Logistic Regression, Decision Tree, Random Forest, and KNN models were trained and optimized using Grid Search.  
- Random Forest achieved the best accuracy and overall performance.  
- PCA helped simplify feature space while maintaining high variance.  
- The project successfully predicts student admission outcomes and provides useful insights.

The final system can be used for profile assessment, analytics dashboards, and predictive modeling for academic institutions.

---
👤 Author

Aniket Jain
📧 jainaniket935@gmail.com

🔗 GitHub: https://github.com/iamAniketjain

🔗 LinkedIn: https://www.linkedin.com/in/aniket-jain-08a47423a/
