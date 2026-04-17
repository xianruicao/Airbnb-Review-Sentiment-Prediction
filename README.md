# Airbnb Review Sentiment Prediction

This project applies Natural Language Processing (NLP) techniques to Airbnb guest reviews in order to understand the language associated with positive and negative experiences. The workflow combines exploratory text analysis, preprocessing, TF-IDF feature extraction, topic modeling with Latent Dirichlet Allocation (LDA), and supervised classification.

The main objective is not only to classify review sentiment, but also to interpret which themes and terms are most strongly connected to guest satisfaction.

## Project Overview

Online reviews contain far more detail than star ratings alone. In this project, written Airbnb reviews are analyzed to:

- explore the structure and vocabulary of the review corpus
- clean and normalize the text for downstream analysis
- identify informative words with TF-IDF
- uncover latent themes with LDA topic modeling
- generate sentiment labels from text polarity
- train classification models to predict positive versus negative reviews
- interpret which topics and words drive sentiment predictions

## Main Files

- `notebooks/airbnb_review_sentiment_prediction.ipynb`: main notebook with the full analysis pipeline
- `reports/airbnb_review_sentiment_prediction_report.pdf`: project report and supporting documentation
- `data/`: local data staging area for raw and processed files
- `outputs/`: local export area for figures and tables
- `requirements.txt`: Python dependencies needed to run the notebook
- `README.md`: repository overview and usage instructions

## Dataset

The original review data comes from the Inside Airbnb dataset.

This repository intentionally does **not** include the raw `reviews.csv` file:

- the file is large and would make the repository unnecessarily heavy
- the project is easier to clone and review when only code and documentation are versioned
- the notebook already uses hosted CSV files for translated and preprocessed text
- local raw datasets belong in `data/raw/`
- local processed datasets belong in `data/processed/`

At the moment, the notebook retrieves:

- translated reviews from an external GitHub-hosted CSV
- preprocessed review text from an external GitHub-hosted CSV

This means the analysis can be reproduced without committing the raw dataset to this repository. If you want to work with the raw file locally, keep it on your machine without adding it to Git.

## Methodology

### 1. Exploratory Data Analysis

The project begins with an exploratory analysis of the review corpus. This stage inspects:

- dataset size and structure
- examples of translated reviews
- the most frequent words in the corpus
- the distribution of review lengths
- the size of the vocabulary and the long-tail distribution of word frequencies

This step helps identify common review patterns and informs the preprocessing and modeling choices used later in the notebook.

### 2. Text Preprocessing

The reviews are cleaned and normalized before modeling. The preprocessing pipeline includes:

- punctuation removal
- tokenization
- stopword filtering
- lowercasing
- optional stemming or lemmatization

The notebook documents the preprocessing logic and relies on a preprocessed text file for faster execution.

### 3. TF-IDF Analysis

TF-IDF is used to identify words and n-grams that carry the most information across the review corpus. In the notebook, the vectorizer is configured with:

- `ngram_range=(1, 3)`
- `min_df=200`
- `max_df=0.8`

This stage highlights characteristic vocabulary and prepares the document-term representation used in later modeling steps.

### 4. Topic Modeling

LDA is used to identify latent themes discussed by guests. Several topic counts are evaluated using coherence scores, and the final notebook analysis uses **4 topics** for interpretability.

The four interpreted topics are:

- Topic 0: Comfort and Amenities
- Topic 1: Host Interaction
- Topic 2: Location and Accessibility
- Topic 3: Room Quality

### 5. Sentiment Label Generation

Because the dataset does not contain manual sentiment labels, the notebook generates binary labels using `TextBlob` polarity:

- positive sentiment -> `1`
- negative sentiment -> `0`

This produces a practical target for classification, but it is important to note that these are automatically generated labels rather than human-annotated ground truth.

### 6. Supervised Classification

Two classifiers are trained on TF-IDF features:

- Logistic Regression with class balancing
- Random Forest with class balancing

The project then compares model performance and uses the learned feature importance signals to interpret which terms and topics are most associated with positive or negative sentiment.

## Key Results

Based on the outputs stored in the notebook:

- translated review dataset after cleaning: `1,015,329` rows
- preprocessed text dataset used for modeling: `100,172` rows
- vocabulary size observed during EDA: `190,864`
- class distribution after sentiment labeling:
  - positive: `86,222`
  - negative: `13,950`

### Classification Performance

Logistic Regression:

- accuracy: `0.95`
- ROC-AUC: `0.9916`

Random Forest:

- accuracy: `0.97`
- ROC-AUC: `0.9877`

### Most Important Topic

When topic importance is aggregated from both models, **Host Interaction** emerges as the strongest theme associated with review sentiment in this project.

### Example High-Impact Positive Terms

Some of the strongest positive terms identified in the notebook include:

- `great`
- `good`
- `nice`
- `comfortable`
- `clean`
- `friendly`
- `perfect`
- `excellent`

### Example High-Impact Negative Terms

Some of the strongest negative terms identified in the notebook include:

- `bad`
- `dirty`
- `unfortunately`
- `difficult`
- `uncomfortable`
- `cold`
- `expensive`
- `small`

## Folder Structure

```text
Airbnb-Review-Sentiment-Prediction/
├── notebooks/
│   └── airbnb_review_sentiment_prediction.ipynb     <- Main analysis notebook
├── reports/
│   └── airbnb_review_sentiment_prediction_report.pdf <- Written project report
├── scripts/
│   └── .gitkeep                                     <- Placeholder for future Python scripts
├── data/
│   ├── README.md                                    <- Explains local data usage
│   ├── raw/                                         <- Local raw datasets such as reviews.csv
│   └── processed/                                   <- Local cleaned datasets and exports
├── outputs/
│   ├── README.md                                    <- Explains where generated artifacts go
│   ├── figures/                                     <- Saved plots and charts
│   └── tables/                                      <- Exported result tables
├── requirements.txt                                 <- Python dependencies
├── .gitignore                                       <- Ignore rules for data and outputs
└── README.md                                        <- Project overview and setup guide
```

The `data/raw/`, `data/processed/`, `outputs/figures/`, and `outputs/tables/` folders are ignored by Git apart from placeholder files, so classmates can keep local datasets and generated files without bloating the repository.

## How To Run

### Option 1: Open the Notebook Directly

If you already have a Python/Jupyter environment, open:

```bash
jupyter notebook notebooks/airbnb_review_sentiment_prediction.ipynb
```

or

```bash
jupyter lab
```

and run the notebook cell by cell.

### Option 2: Set Up a Fresh Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

Download the required language resources:

```bash
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

Then start Jupyter and open the notebook from the `notebooks/` folder.

Note: `requirements.txt` installs Python packages only. The spaCy English model still needs to be downloaded separately with `python -m spacy download en_core_web_sm`.

## Reproducibility Notes

- The notebook contains some inline `%pip install` commands.
- Parts of the workflow rely on externally hosted CSV files rather than local raw data.
- The preprocessing step is documented in the notebook, but the notebook uses a precomputed preprocessed dataset to avoid re-running the full cleaning pipeline every time.
- Results may vary slightly across library versions or environments.

## Limitations

- Sentiment labels are generated automatically with `TextBlob`, so the classification task is based on pseudo-labels rather than manually validated annotations.
- Review translation and preprocessing may introduce noise or lose nuance.
- The notebook depends on external hosted files, which is convenient for reproducibility but adds an external dependency.
- The review classes are imbalanced, with many more positive than negative examples.

## Possible Improvements

- replace pseudo-labels with manually annotated sentiment labels
- package the workflow into reusable Python modules instead of a single notebook
- add a `requirements.txt` or environment file for stricter reproducibility
- evaluate additional models such as linear SVM, gradient boosting, or transformer-based classifiers
- compare sentiment drivers across cities, time periods, or listing types

## Authors

Group members:

- Elvis Casco
- Xianrui Cao
- Román Feria

## License

No license file is currently included in this repository. If you plan to share or reuse the project publicly, adding a license is recommended.
