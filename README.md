# Twitter Sentiment Analysis with RoBERTa

A deep learning-based sentiment analysis pipeline for Twitter data using RoBERTa (Robustly Optimized BERT Approach). This project classifies tweets into four sentiment categories: Positive, Negative, Neutral, and Irrelevant.

## 📊 Dataset

The dataset used is the [Twitter Entity Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data) from Kaggle, containing:
- **Training Data**: 74,000+ tweets with sentiment labels
- **Validation Data**: 1,000+ tweets for model evaluation

### Dataset Features
- `Tweet_ID`: Unique identifier for each tweet
- `Entity`: Entity mentioned in the tweet (e.g., company, product, person)
- `Sentiment`: Target label (Positive, Negative, Neutral, Irrelevant)
- `Tweet_Content`: The actual tweet text

## 🚀 Features

- **Exploratory Data Analysis (EDA)**
  - Sentiment class distribution visualization
  - Word clouds for each sentiment type
  - Text length analysis with outlier detection
  - Duplicate and missing value handling

- **Text Preprocessing**
  - Removal of duplicates and null values
  - Outlier removal using IQR method
  - Stop word removal with NLTK

- **Model Architecture**
  - **Pre-trained Model**: RoBERTa-base (124M parameters)
  - **Custom Layers**: Dense layer with softmax activation
  - **Optimizer**: Adam (learning rate = 1e-5)
  - **Loss Function**: Categorical Crossentropy
  - **Metrics**: Categorical Accuracy

## 📈 Results

The model achieves:
- **Training Accuracy**: Evaluated with classification report and confusion matrix
- **Test Accuracy**: Performance metrics on 30% holdout test set
- **Validation Accuracy**: Evaluation on separate validation dataset

### Performance Metrics
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix visualization
- One-hot encoded multiclass predictions

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **Transformers (Hugging Face)** - RoBERTa implementation
- **Scikit-learn** - Data splitting and evaluation metrics
- **Pandas/NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Data visualization
- **WordCloud** - Text visualization
- **NLTK** - Text preprocessing
- **Statsmodels** - Statistical analysis (VIF)

## 📋 Requirements

```bash
nltk
pandas
numpy
matplotlib
seaborn
wordcloud
spacy
scikit-learn
tensorflow
transformers
statsmodels
```

twitter-sentiment-analysis/
│
├── twitter_sentiment_analysis.py    # Main code file
├── requirements.txt                  # Project dependencies
├── README.md                         # Project documentation
│
├── data/
│   ├── twitter_training.csv         # Training dataset
│   └── twitter_validation.csv       # Validation dataset
│
└── outputs/
    ├── confusion_matrices/          # Generated confusion matrices
    ├── wordclouds/                  # Word cloud visualizations
    └── reports/                     # Classification reports
