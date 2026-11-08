# SMS Spam Classification and Business-Text Spelling Correction

Two NLP systems for noisy real-world text: a spam classifier for SMS messages, deployed as a web app, and a spelling corrector for business and financial text with a desktop GUI. Built for the Natural Language Processing module (CT052-3-M) of the MSc in Artificial Intelligence at Asia Pacific University (2026).

## SMS spam classification

- **Data:** the UCI SMS Spam Collection, 5,169 messages after removing duplicates (87.4% ham, 12.6% spam).
- **Model:** a Linear SVM on word n-grams (1 to 2) and character n-grams (2 to 5), compared with Naive Bayes. It was tuned with 5-fold grid search on spam F1 rather than accuracy, because the classes are imbalanced.
- **Held-out results:**

| Metric | Value |
|---|---|
| Accuracy | 99.1% |
| Spam precision | 0.99 |
| Spam recall | 0.94 |
| Spam F1 | 0.96 |
| ROC-AUC | 0.999 |

- **Deployment:** a Streamlit app that loads the saved model and scores any message.

## Spelling correction

- A noisy-channel corrector: candidate words within a small edit distance, ranked by word frequencies from the NLTK Reuters business corpus.
- Edit distance implemented with dynamic programming.
- A Tkinter desktop GUI that highlights misspelt words and suggests corrections with their edit distance.

## Project structure

```
sms-spam-and-spelling-correction/
├── notebooks/
│   └── spam_and_spelling.ipynb   # both systems: EDA, training, evaluation, model export
├── app/
│   └── app_streamlit.py          # spam classifier web app
├── spelling/
│   ├── spell_core.py             # edit distance and noisy-channel corrector
│   └── spell_gui.py              # desktop GUI
├── data/
│   └── spam.csv                  # UCI SMS Spam Collection
├── models/                       # spam_model.joblib is saved here by the notebook
├── requirements.txt
└── README.md
```

## Run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook spam_and_spelling.ipynb   # run all cells to train and save models/spam_model.joblib
cd ..
streamlit run app/app_streamlit.py
python spelling/spell_gui.py
```

## Data licence

`data/spam.csv` is the UCI SMS Spam Collection by Almeida and Hidalgo, licensed CC BY 4.0.

## Authors

Sameena Siddiqua and Loula Sahal Ali
