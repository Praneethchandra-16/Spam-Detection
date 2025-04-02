# 📬 Spam Detection using NLP

This project applies Natural Language Processing (NLP) techniques to classify SMS messages as **spam** or **ham** (not spam). The objective was to develop a robust pipeline that can handle imbalanced data and deliver reliable classification performance using traditional machine learning methods.

---

## 🧠 Objective
- Detect spam messages in a dataset of SMS messages.
- Use text vectorization and machine learning models to classify messages.
- Focus on F1-score as the primary evaluation metric due to data imbalance.

---

## 📊 Dataset
- Source: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- Total messages: ~5,500
- Class distribution:
  - **Ham**: ~87%
  - **Spam**: ~13%

---

## 🧪 Workflow

### Phase 1: Pipeline Prototyping (on small subset)
- Explored multiple pipelines using combinations of:
  - **CountVectorizer**, **TF-IDF**
  - **Logistic Regression**, **other baseline classifiers**
- Evaluated each pipeline using **F1-score**, especially for spam class
- Selected the pipeline with **TF-IDF + Logistic Regression** based on performance

### Phase 2: Full-Scale Training
- Trained the selected pipeline on the complete dataset
- Used **GridSearchCV** for hyperparameter tuning
- Evaluated performance using `classification_report`

---

## 🧱 Final Model Pipeline
```python
Pipeline([
  ('tfidf', TfidfVectorizer()),
  ('clf', LogisticRegression())
])
```

---

## 📈 Results
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Ham   | ~0.97     | ~0.99  | ~0.98    |
| Spam  | ~0.95     | ~0.89  | ~0.92    |

- **Precision (spam)**: Indicates a low false positive rate
- **Recall (spam)**: Shows strong ability to detect actual spam
- **F1-score**: Balanced and robust, suitable for imbalanced data scenarios

---

## 💾 Model Persistence
- Trained pipeline saved using `joblib`
- Ready for integration in applications or for further tuning

---

## ✅ Conclusion
This project demonstrates the effectiveness of classical machine learning techniques in spam detection when paired with thoughtful pipeline design and evaluation strategies. By addressing class imbalance and optimizing for relevant metrics, the final model delivers strong, generalizable performance.

---

## 📁 Files Included
- `spam.csv` — Dataset used
- `File1_HW2.ipynb` — Initial experiments and pipeline comparison
- `File2_HW2.ipynb` — Final model training and evaluation
- `Spam_Detection_Project_Summary.docx` — Project write-up

---

## 🚀 Future Improvements
- Explore deep learning methods like LSTMs or transformers
- Add more advanced preprocessing (lemmatization, stemming)
- Deploy as a web API or integrate into email/SMS platforms

