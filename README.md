# Fake News Detection using Machine Learning (Python)

## 📌 Project Overview
Fake news spreads rapidly on social media and online platforms, often misleading people and influencing opinions. This project aims to detect whether a news article is **REAL** or **FAKE** using **Machine Learning and Natural Language Processing (NLP)** techniques.

The model analyzes news text and classifies it using **TF-IDF Vectorization** and a **Passive Aggressive Classifier**, achieving high accuracy on real-world data.

---

## 🎯 Objectives
- Understand the concept of fake news and text classification  
- Apply NLP techniques for feature extraction  
- Build and evaluate a machine learning model  
- Classify news articles as REAL or FAKE  

---

## 🧠 Concepts Used
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Supervised Machine Learning
- Text Classification
- Model Evaluation Metrics

---

## 🛠️ Technologies & Tools
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn  
- **Environment:** Jupyter Notebook  

---

## 📂 Dataset Information
- **File Name:** `news.csv`  
- **Dataset Size:** 7,796 rows × 4 columns  
- **Columns:**
  - `id` – Unique identifier
  - `title` – News title
  - `text` – News content
  - `label` – REAL or FAKE  

The dataset contains political news articles labeled as real or fake.

---

## ⚙️ Project Workflow
1. Import required libraries  
2. Load and explore the dataset  
3. Split data into training and testing sets  
4. Convert text data into numerical features using **TF-IDF Vectorizer**  
5. Train a **Passive Aggressive Classifier**  
6. Evaluate the model using accuracy and confusion matrix  

---

## 🧪 Model & Evaluation
- **Algorithm Used:** Passive Aggressive Classifier  
- **Feature Extraction:** TF-IDF Vectorization  
- **Accuracy Achieved:** ~92.8%  

### Confusion Matrix Results:
- True Positives: 589  
- True Negatives: 587  
- False Positives: 42  
- False Negatives: 49  

---

## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries
pip install numpy pandas scikit-learn


---


2️⃣ Launch Jupyter Notebook
jupyter lab

---

3️⃣ Run the Notebook
Load news.csv

Execute cells step by step using Shift + Enter

---

📈 Results
The model successfully classifies news articles with high accuracy, demonstrating the effectiveness of NLP and machine learning techniques in detecting misinformation.

---

🔮 Future Enhancements

Improve accuracy using advanced models (Logistic Regression, SVM, Deep Learning)

Deploy the model using Flask as a web application

Extend to multilingual fake news detection

Use real-time news data from APIs

---

📚 Learning Outcomes

Hands-on experience with NLP and text classification

Understanding of TF-IDF and online learning algorithms

Practical application of machine learning to real-world problems

---

🧑‍💻 Author

Rushikesh Rathod </br>
Computer Science Student | Machine Learning Enthusiast

---
