### Customer Churn Prediction

### Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company (Churn) or continue using its services.

This project uses customer information such as age, gender, tenure, usage frequency, support calls, payment delays, subscription type, contract length, and total spending to predict customer churn.

### Objectives

* Analyze customer churn data.

* Perform data cleaning and preprocessing.

* Explore customer behaviour and churn patterns.

* Train and evaluate multiple Machine Learning models.

* Compare model performance.

* Select the best-performing model.

* Develop an interactive web application using Streamlit.

### Dataset

The project uses two CSV files:

* `customer_churn_dataset-training-master.csv`

* `customer_churn_dataset-testing-master.csv`

### Dataset Features

| Feature           | Description                       |
| ----------------- | --------------------------------- |
| CustomerID        | Unique customer ID                |
| Age               | Customer's age                    |
| Gender            | Customer's gender                 |
| Tenure            | Duration of customer relationship |
| Usage Frequency   | Frequency of service usage        |
| Support Calls     | Number of support calls           |
| Payment Delay     | Payment delay                     |
| Subscription Type | Subscription category             |
| Contract Length   | Contract duration                 |
| Total Spend       | Total customer spending           |
| Last Interaction  | Last customer interaction         |
| Churn             | Target variable (0 or 1)          |

### Technologies Used

* Python

* Pandas

* NumPy

* Matplotlib

* Seaborn

* Scikit-learn

* LightGBM

* Streamlit

* Jupyter Notebook

### Machine Learning Algorithms

The following classification algorithms are compared in this project:

* Logistic Regression

* Gaussian Naive Bayes

* Bernoulli Naive Bayes

* Decision Tree Classifier

* Random Forest Classifier

* Extra Trees Classifier

* Gradient Boosting Classifier

* Bagging Classifier

* AdaBoost Classifier

* K-Nearest Neighbors

* Multi-Layer Perceptron

* XGBoost

* LightGBM

### Best Model

The best-performing model recorded in this project is LightGBM Classifier.

### Model Performance

| Evaluation Metric | Score  |
| ----------------- | ------ |
| Accuracy          | 93.63% |
| Precision         | 89.83% |
| Recall            | 99.84% |
| F1 Score          | 94.57% |

### Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn.ipynb
├── app.py
├── customer_churn_LGBMC.pkl
│
├── customer_churn_dataset-training-master.csv
├── customer_churn_dataset-testing-master.csv
│
├── requirements.txt
└── README.md
```

### Installation and Setup

### 1\. Clone the repository

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
```

### 2\. Navigate to the project directory

```bash
cd Customer-Churn-Prediction
```

### 3\. Install required libraries

```bash
pip install -r requirements.txt
```

### 4\. Run the Streamlit application

```bash
streamlit run app.py
```

### Requirements

Create a `requirements.txt` file with the following libraries:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
```

### Web Application

The project includes an interactive Streamlit application.

Users can enter customer details and receive a prediction about whether the customer is likely to churn or stay.

The application displays:

* Customer churn prediction.

* Customer retention prediction.

* Prediction probability, when supported by the model.

### Model Saving

The trained LightGBM model is saved using Pickle.

```python
import pickle

with open("customer_churn_LGBMC.pkl", "wb") as file:
    pickle.dump(model, file)
```

### Future Improvements

* Improve model performance through hyperparameter tuning.

* Use separate training and testing datasets for reliable evaluation.

* Deploy the application online.

* Add more interactive visualizations.

* Develop strategies to help businesses reduce customer churn.

### Author

Aasif Khan

### License

This project is created for educational and learning purposes.
