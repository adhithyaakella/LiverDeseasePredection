# Liver Disease Prediction

## 🚀 Overview
This project aims to predict whether a patient has liver disease or not based on various medical features, including age, bilirubin levels, liver enzyme levels, and more. It uses machine learning models such as **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Random Forest**, **Decision Tree**, and **XGBoost** to predict the presence of liver disease. The project includes data preprocessing, feature engineering, class balancing using **SMOTE**, and model evaluation techniques to enhance predictive performance.

## 📂 Files
- **Liver Disease Prediction using Classification Models.py** – Main Python script that handles data preprocessing, feature engineering, and model training.
- **Liver Patient Dataset.csv** – The dataset containing medical information about patients with features like liver function test results, age, gender, and liver disease status.
- **LiverDiseasePrediction_Report.pdf** – Detailed report that summarizes the findings, visualizations, and model evaluation metrics.
  

## 📊 Key Insights
✔ **Bilirubin levels**, **liver enzymes**, and **age** are significant factors in liver disease diagnosis.  
✔ **SMOTE** (Synthetic Minority Over-sampling Technique) was used to balance the dataset due to class imbalance (healthy vs. liver disease).  
✔ **Random Forest** was the best-performing model after hyperparameter tuning.  
✔ **Dimensionality reduction using PCA** improved model efficiency by reducing multicollinearity.  
✔ **Confusion Matrix**, **ROC Curve**, and **Classification Report** were used to evaluate the model’s performance.

## 🔧 Techniques Used:
1. **Data Preprocessing**:
   - **Handling Missing Values**: Categorical columns were imputed with the **mode** and numerical columns with the **median** to handle missing data.
   - **Outlier Detection**: Z-scores were used to detect and remove outliers to improve the model’s robustness.
   - **Feature Encoding**: One-hot encoding was applied to the categorical column (gender of the patient).
   
2. **Class Balancing**:
   - **SMOTE** was applied to the training data to balance the class distribution of the dataset (healthy vs. liver disease).

3. **Feature Engineering**:
   - **Correlation Analysis**: A heatmap was generated to identify highly correlated features, and highly correlated features were dropped to avoid multicollinearity.
   - **Dimensionality Reduction**: **PCA (Principal Component Analysis)** was applied to reduce feature dimensions and improve model efficiency.

4. **Model Training**:
   - **Logistic Regression**, **KNN**, **Random Forest**, **Decision Tree**, and **XGBoost** models were trained and evaluated on the dataset.
   - **Hyperparameter Tuning**: Grid Search was used for optimizing model hyperparameters.
   - **Cross-Validation**: Stratified K-fold cross-validation was employed to assess the model’s performance.

5. **Model Evaluation**:
   - **Metrics Used**: Accuracy, Precision, Recall, F1 Score, ROC Curve, and Confusion Matrix were used to evaluate the models.
   - **Best Performing Model**: **Random Forest** outperformed other models based on the evaluation metrics.

## 🔗 Demo
👉 **[Link to project notebook](https://colab.research.google.com/drive/1JQOkMWECdjUZIB4hzi1BttZhsNYPoAZx)** (Google Colab notebook)

## 📌 Tools Used
- **Python** (for data analysis and machine learning)
- **Pandas** (data manipulation)
- **Scikit-learn** (machine learning models and evaluation)
- **Seaborn** and **Matplotlib** (for data visualization)
- **SMOTE** (for balancing the dataset)
- **Plotly** (for advanced plotting)
- **Google Colab** (for running the code)
- **Kaggle** (data source)

## 💻 Usage
- The **Liver Disease Prediction** model uses **Logistic Regression**, **KNN**, **Random Forest**, **Decision Tree**, and **XGBoost** for classification.
- **SMOTE** is applied to balance the dataset before training the models.
- The **confusion matrix**, **classification report**, and **ROC curve** are used to evaluate model performance.

💡 **Feel free to fork this repo & explore further insights!** 🚀
