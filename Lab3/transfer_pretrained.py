from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report
from config import EMB_PATH, MAX_LENGTH, DATASET
import numpy as np

# DATASET = 'MR'
# PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'
# options: "MR", "Semeval2017A"s

DATASET = 'Semeval2017A'
PRETRAINED_MODEL = 'oliverguhr/german-sentiment-bert'

# 2-class models
# siebert/sentiment-roberta-large-english
# bert-base-uncased
# distilbert-base-uncased-finetuned-sst-2-english
# 3-class models
# cardiffnlp/twitter-roberta-base-sentiment
# mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
# oliverguhr/german-sentiment-bert

LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'bert-base-uncased': {
        'LABEL_0': 'positive',
        'LABEL_1': 'negative',
    },
    'distilbert-base-uncased-finetuned-sst-2-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis': {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive',
        'LABEL_2': 'neutral',
    },
    'oliverguhr/german-sentiment-bert': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    }
    
    #   'cardiffnlp/twitter-roberta-base-sentiment': {
    #       'POSITIVE': 'positive',
    #       'NEGATIVE': 'negative',
    #   }
}

if __name__ == '__main__':
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))

    # define a proper pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)

    y_pred = []
    for x in tqdm(X_test):
        # TODO: Main-lab-Q6 - get the label using the defined pipeline 
        result = sentiment_pipeline(x)[0]
        label = result['label']  
        y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label])
    y_pred = le.transform(y_pred)
    print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')