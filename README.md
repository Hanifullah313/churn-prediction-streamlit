# 📊 Customer Churn Prediction App

A web-based machine learning application that predicts **customer churn** — i.e., whether a customer is likely to leave a service — using a **Logistic Regression** model built within a **scikit-learn pipeline**. The app is deployed with a clean and interactive **Streamlit** interface.

---

## 🧠 Overview

The app takes customer details such as **tenure, contract type, and monthly charges** as input and provides a **real-time prediction** (Stay or Churn) along with a **probability score**.  
It is designed for telecom or subscription-based businesses seeking quick churn risk insights.

---

## 📸 Application Demo

> *(Recommended: Add a screenshot or GIF of your running application here!)*

Example:

```
[Insert image: app_screenshot.png]
```

---

## ✨ Features

- **🎛 Interactive Sidebar:** Input all 19 required customer features easily.
- **⚡ Real-Time Prediction:** Instantly predicts churn using the trained model pipeline.
- **📈 Probability Score:** Displays model confidence via a dynamic progress bar.
- **🎨 Dynamic UI Feedback:** Color-coded message for CHURN (🔴) or STAY (🟢).
- **🔗 End-to-End Pipeline:** Preprocessing (imputation, scaling, encoding) handled inside the pipeline — no manual steps needed.

---

## 🛠️ Technology Stack

| Component | Description |
|------------|-------------|
| **Python 3.x** | Core programming language |
| **Streamlit** | For the interactive front-end and deployment |
| **Scikit-learn** | For building the ML pipeline and logistic regression model |
| **Pandas** | Data manipulation and input handling |
| **Joblib** | Model serialization (saving/loading trained pipeline) |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Create a `requirements.txt` file (if not already present) with the following:

```
streamlit
pandas
scikit-learn
```

Then install:
```bash
pip install -r requirements.txt
```

### 4. Ensure the Model Artifact Exists
The app requires the trained model file:  
`artifacts/model.pkl`

If missing, run the model training notebook to generate it:
```
notebooks/02-model-building.ipynb
```

### 5. Run the Application
```bash
streamlit run app.py
```

Your default browser should open automatically with the running app.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── .gitignore
├── README.md                <-- You are here!
├── requirements.txt         <-- All project dependencies
├── app.py                   <-- Streamlit application file
│
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   └── 02-model-building.ipynb  <-- Model training & pipeline creation
│
├── data/
│   └── raw/
│       └── churn_data.csv       <-- Raw dataset used for training
│
└── artifacts/
    └── model.pkl                <-- Trained scikit-learn pipeline
```

---

## 🧩 Model & Preprocessing Details

The `model.pkl` file is a **scikit-learn Pipeline** object that encapsulates all data transformations and the classifier.  
This ensures that the exact same preprocessing steps are applied to both training and live inputs.

### Pipeline Components

#### **1. Numerical Columns (`num_cols`)**
- **KNNImputer:** Fills missing values using 5 nearest neighbors  
- **StandardScaler:** Normalizes data (mean = 0, std = 1)

#### **2. Categorical Columns (`cat_cols`)**
- **SimpleImputer:** Replaces missing values with most frequent category  
- **OneHotEncoder:** Encodes categorical features (drops first category to avoid multicollinearity)

#### **3. Ordered Columns (`ordered_cols`)**
- **SimpleImputer:** Replaces missing values with most frequent category  
- **OneHotEncoder:** Encodes ordered categorical variables (e.g., contract duration)

#### **4. Classifier**
- **Model:** `LogisticRegression`  
- **Parameters:** `C=1`, `solver='liblinear'`

---

## 🚀 How to Use

1. **Enter Customer Information:**  
   Use the Streamlit sidebar to input all customer details.

2. **Get Prediction:**  
   Click **"Predict Churn"**.

3. **View Results:**  
   - Probability bar shows likelihood of churn  
   - Color-coded result:
     - 🟢 **STAY** — customer is likely to stay
     - 🔴 **CHURN** — customer is likely to leave

---

## 🧰 Troubleshooting

| Issue | Possible Fix |
|-------|---------------|
| `model.pkl not found` | Run the model training notebook or ensure it’s placed in `artifacts/`. |
| Missing library errors | Run `pip install -r requirements.txt` again. |
| Streamlit app not launching | Ensure the virtual environment is activated and port 8501 is free. |

---

## 👥 Contributors

- **Your Name** – *Developer & Maintainer*  
- *(Add additional contributors if applicable)*

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
