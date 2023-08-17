import numpy as np
import evaluate
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from utils.load_datasets import load_MR, load_Semeval2017A

# 2-class models
# siebert/sentiment-roberta-large-english
# bert-base-uncased
# distilbert-base-uncased-finetuned-sst-2-english
# 3-class models
# bert-base-cased
# cardiffnlp/twitter-roberta-base-sentiment
# mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
# oliverguhr/german-sentiment-bert


DATASET = 'MR'  # 'MR' or 'Semeval2017A'
PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'
# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def prepare_dataset(X, y):
    texts, labels = [], []
    for text, label in zip(X, y):
        texts.append(text)
        labels.append(label)

    return Dataset.from_dict({'text': texts, 'label': labels})


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

    # prepare datasets
    train_set = prepare_dataset(X_train, y_train)
    test_set = prepare_dataset(X_test, y_test)
    print("train.shape is: ",np.shape(train_set))

    # define model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=n_classes)
    # move the mode weight to cpu or gpu
    model.to(DEVICE)

    # tokenize datasets
    tokenized_train_set = train_set.map(tokenize_function)
    tokenized_test_set = test_set.map(tokenize_function)

    # TODO: Main-lab-Q7 - remove this section once you are ready to execute on a GPU
    #  create a smaller subset of the dataset
    n_samples = 500
    small_train_dataset = tokenized_train_set.shuffle(
        seed=42).select(range(n_samples))
    small_eval_dataset = tokenized_test_set.shuffle(
        seed=42).select(range(n_samples))

    # TODO: Main-lab-Q7 - customize hyperparameters once you are ready to execute on a GPU
    # training setup
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        num_train_epochs=20,
        per_device_train_batch_size=8 #prin 8
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # train
    trained_model = trainer.train()