# 🎓 Student Admission Prediction using Machine Learning

This project analyzes a student admission dataset and builds multiple machine-learning models to predict whether a student will be admitted based on academic scores, test results, and profile attributes.  
The project includes Exploratory Data Analysis (EDA), data preprocessing, PCA, and model comparison (Logistic Regression, Decision Tree, Random Forest, and KNN).

---

## 📁 Dataset Information

- **Dataset:** original_data.csv  
- **Rows:** 53,644  
- **Columns:** 26  
- Each row represents a student's academic and profile details, along with the final admission decision (`admit`).

### 📌 Key Features:
- Academic scores (CGPA, CGPA scale, College)
- Standardized test scores (GRE, TOEFL, GMAT)
- Experience (Research, Industry, Internship)
- Publications (Conference & Journal)
- Program, Department, Specialization
- Final Target Variable → `admit` (1 = Admitted, 0 = Not Admitted)

---

## 🎯 AIM
The aim of this project is to perform Exploratory Data Analysis (EDA), preprocess the dataset, apply PCA, and build machine learning models to predict student admission outcomes based on academic and profile features.

---

## 🧪 Exploratory Data Analysis (EDA)

- Analyzed missing values and data types  
- Distribution plots for numerical features  
- Count plots for categorical columns  
- Relationship between features and admission rate  
- Correlation heatmap  
- Outlier & pattern detection  

---

## 🛠️ Data Preprocessing

- Handling missing values  
- Encoding categorical variables  
- Scaling numerical features  
- Splitting data into training and testing sets  
- PCA applied for dimensionality reduction  

---

## 🤖 Machine Learning Models Used

### **1️⃣ Logistic Regression**
- Baseline model  
- GridSearchCV used for hyperparameter tuning  
- Stable and consistent performance  

### **2️⃣ Decision Tree Classifier**
- Easy to interpret  
- Grid search performed  
- Prone to overfitting  

### **3️⃣ Random Forest Classifier**
- Best performing model in this project  
- Handles mixed data very well  
- Grid search for best parameters  

### **4️⃣ K-Nearest Neighbors (KNN)**
- Distance-based model  
- Sensitive to scaling  
- Lowest performance among all models  

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

## 📉 PCA (Principal Component Analysis)

- PCA applied to understand variance explained  
- Helps reduce dimensionality  
- First few components captured most of the total variance  

---

## ✅ Conclusion

- EDA revealed key trends in academic scores, experience, and test results affecting admissions.  
- Logistic Regression, Decision Tree, Random Forest, and KNN models were trained and optimized using Grid Search.  
- Random Forest achieved the best accuracy and overall performance.  
- PCA helped simplify feature space while maintaining high variance.  
- The project successfully predicts student admission outcomes and provides useful insights.

---

## 📌 Tech Stack

- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## 📂 Project Structure

├── original_data.csv
├── order.ipynb
├── README.md
└── models/

## 🙌 Author
**Aniket Jain**  
Feel free to contribute or raise issues!
