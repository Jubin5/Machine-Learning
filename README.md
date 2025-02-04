# Breast Cancer Prediction using Machine Learning

## 📌 Overview
Breast cancer is one of the most common types of cancer worldwide. Early detection can significantly improve treatment outcomes. This project uses machine learning techniques to
classify breast cancer tumors as **Malignant** or **Benign** based on the **Breast Cancer Wisconsin dataset** from sklearn.datasets.

## 📂 Dataset
- **Source**: sklearn.datasets.load_breast_cancer()
- **Features**: 30 numerical features describing tumor characteristics (e.g., radius, texture, smoothness, compactness, etc.)
- **Target**: Binary classification (0 = Malignant, 1 = Benign)
- **Dataset Size**: 569 samples

## 🚀 Technologies Used
- Python 🐍
- Scikit-Learn
- Pandas & NumPy
- Matplotlib & Seaborn (for visualization)

## 🔧 Installation & Setup
1. Clone the repository:
  bash
   git clone https://github.com/jubin5/breast-cancer-ml.git
   cd breast-cancer-ml
  
2. Install required libraries:
3. 

## 📊 Exploratory Data Analysis (EDA)
- Checked for missing values and handled them.
- Visualized feature distributions.
- Examined correlation between features.

## 🏗️ Model Training & Evaluation
- **Algorithms Used**: Logistic Regression, Random Forest, SVM, and K-Nearest Neighbors (KNN)
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 🔥 Best Performing Model
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|------------|--------|----------|
| Random Forest | 98.2% | 97.5% | 98.9% | 98.2% |

## 📌 Key Findings
- Random Forest outperformed other models with **98.2% accuracy**.
- Feature importance analysis revealed that **mean radius** and **worst area** were key predictors.

## 📎 Results & Visualizations
- Confusion matrix to evaluate model performance.
- ROC-AUC curve to compare models.
- Feature importance bar chart.

## 🎯 Future Improvements
- Fine-tune hyperparameters using GridSearchCV.
- Try deep learning models like ANN.
- Deploy the model using Flask or FastAPI.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests if you have any improvements or suggestions.

## 🔗 Connect with Me
- 🔗 www.linkedin.com/in/jubinkbabu
- 📧 Email:jubinkbabu5@gmail.com

⭐ **If you found this useful, please star the repository!** ⭐
