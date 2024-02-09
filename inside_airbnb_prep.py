import pandas as pd
from tqdm import tqdm
import seaborn as sns
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from math import ceil
import numpy as np
import fasttext
import sys

import urllib.request

def preprocess(text):
    new_text = []

    for t in str(text).lower().split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = re.sub(r'<.*?>', " ", t)
        t = re.sub('\W+', ' ', str(t))
        t = t.strip()
        new_text.append(t)
    return re.sub("\s{2,}", " ", " ".join(new_text)).strip()

def batchify(text, max_length):
    batch_size = ceil(len(text.split(" ")) / max_length)
    start, end = 0, max_length
    sub_texts = []
    split_text = text.split(" ")
    
    for i in range(batch_size):
        sub_texts.append(" ".join(split_text[start:end]))
        start += max_length
        end += max_length

    return sub_texts

def calculate_score(scores):
    stacked_scores = np.array(scores[0])
    for score in scores[1:]:
        stacked_scores = np.vstack((stacked_scores, score))

    if len(scores) == 1: return stacked_scores
    return np.mean(stacked_scores, axis=0)

def get_sentiment_language(df_reviews):
    batch_size = 350

    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz", "lid.176.ftz")
    lid_model = fasttext.load_model("lid.176.ftz") 

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    tokenizer.model_max_length = 512
    
    df_reviews['sentiment'] = ""
    df_reviews['language'] = ""
    for i, v in tqdm(df_reviews.iterrows(), total=len(df_reviews)):
        text = v["review_treated"]
        labels = ['Negative', 'Neutral', 'Positive']

        outputs = []
        mult = False
        for batch in batchify(text, batch_size): # 350 < 512 - pode haver mais subwords que tokens
            encoded_tweet = tokenizer(batch, return_tensors='pt', truncation=True)
            
            if len(encoded_tweet['input_ids']) > batch_size: print(len(encoded_tweet['input_ids']), len(batch))
                
            output = model(**encoded_tweet)
            if len(outputs) == 0:
                outputs = np.array([softmax(output[0][0].detach().numpy())])
            else:
                outputs = np.vstack((outputs, softmax(output[0][0].detach().numpy())))
                mult = True

        scores = calculate_score(outputs)
        if mult == True:
            pass
            
        preds = lid_model.predict(text)
        language = preds[0][0].split("__label__")[1]

        df_reviews.at[i, "sentiment"] = labels[list(scores).index(max(scores))]
        df_reviews.at[i, "language"] = language
    
    return df_reviews


if __name__ == '__main__':
    df_reviews = pd.read_csv("reviews.csv")
    df_reviews['review_treated'] = df_reviews['comments'].apply(preprocess)
    df_reviews = get_sentiment_language(df_reviews)
    df_reviews.to_csv("reviews-airbnb-enriched.csv", index=None)