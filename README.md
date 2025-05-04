# Loan Default Prediction

## Description
This project aims to predict whether a loan applicant will default on a loan based on their financial and personal information. The dataset is preprocessed to handle categorical features, scale numerical features, and address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique). The project evaluates the performance of several machine learning models including:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- Naive Bayes
- Decision Tree

Model performance is measured using classification reports, confusion matrices, ROC AUC scores, and visualized through ROC curves for model comparison.

---

## Technologies
- **Python 3.x**
- **pandas** - For data manipulation
- **numpy** - For numerical operations
- **scikit-learn** - For machine learning models and evaluation metrics
- **imbalanced-learn** - For handling class imbalance with SMOTE
- **matplotlib** & **seaborn** - For data visualization and plotting
- **xgboost** - For gradient boosting

---

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone <repository_url>
cd <repository_folder>

2. Create a Virtual Environment (Optional but Recommended)
It's recommended to create a virtual environment to manage dependencies:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
Create a requirements.txt file with the following contents:

txt
Copy
Edit
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
xgboost
Then, install the dependencies using the following command:

bash
Copy
Edit
pip install -r requirements.txt
4. Dataset
Place the Loan_default.csv dataset in the project folder (or update the data_path variable in the script with the correct path to your dataset).

Running the Project
Ensure the virtual environment is activated (if you're using one).

Run the Python script:

bash
Copy
Edit
python loan_default_predictor.py
This script will:

Preprocess the dataset

Train models on the processed data

Evaluate each model's performance

Plot ROC curves for comparison

License
This project is licensed under the MIT License - see the LICENSE file for details.

Authors
Shubhra Shrivastava

vbnet
Copy
Edit

Make sure to replace `<repository_url>` with the actual URL of your repository. Once you paste this on your GitHub repository, it will serve as a complete guide for setting up and running the project. Let me know if you'd like further changes!







