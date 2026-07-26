# 🌍 Earthquake Damage Prediction using Machine Learning

## 📌 Project Overview
This project predicts the level of damage to buildings after an earthquake using Machine Learning classification algorithms. The model is trained on historical earthquake building data and helps identify buildings that are likely to suffer higher damage.

---

## 📂 Dataset
The project uses two datasets:
- train_values.csv
- train_labels.csv

After merging both datasets, the final dataset contains building information and the corresponding damage grade.

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

---

## 📊 Project Workflow
1. Data Loading
2. Data Merging
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing
6. Model Building
7. Model Comparison
8. Hyperparameter Tuning
9. Feature Importance
10. Final Evaluation

---

## 🤖 Machine Learning Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

---

## 📈 Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance

---

## 📁 Project Structure

```
Earthquake-Damage-Prediction
│
├── Dataset
│   ├── train_values.csv
│   └── train_labels.csv
│
├── Notebook
│   └── Earthquake_Damage_Prediction.ipynb
│
├── model
│   ├── model.pkl
│   ├── encoders.pkl
│   ├── feature_columns.pkl
│   └── feature_types.pkl
│
├── Images
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 🚀 How to Run

1. Clone the repository
2. Install the required libraries

```
pip install -r requirements.txt
```

3. Run the notebook first to generate the model artifacts

```
jupyter notebook
```

Run all cells in `notebook/Earthquake_Damage_Prediction.ipynb`. The last cell must save `model.pkl`, `encoders.pkl`, `feature_columns.pkl`, and `feature_types.pkl` into a `model/` folder at the repo root.

4. Launch the Streamlit app

```
streamlit run app.py
```

---

## 📌 Results

The Random Forest model achieved the best performance among the tested models and was selected as the final model.

---

## 👨‍💻 Author

**Dileep Atluri**

GitHub: https://github.com/dileepatluri/Earthquake-Damage-Prediction

LinkedIn: https://www.linkedin.com/in/dileep-atluri-bab60932a/

## 🚀 Live Demo

🔗 **Live App:** https://earthquake-damage-prediction-ebwesyfp3mgf8zq2wmldsf.streamlit.app/
