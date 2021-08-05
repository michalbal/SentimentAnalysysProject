import pandas as pd
import re
import os

HTML_TAG_RE = re.compile(r'<[^>]+>')
AT_RE = re.compile(r'@[\w]+')
HASHTAG_RE = re.compile(r'@[\w]+')
NON_ALPHANUM = re.compile(r'#[\W]')


def remove_tags(text):
    no_tags = HTML_TAG_RE.sub('', text)
    no_tags = AT_RE.sub('', no_tags)
    return HASHTAG_RE.sub('', no_tags)


def preprocess_text(sen):

    sentence = sen.lower()

    # Remove html tags and hashtags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers - also removes icons
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = NON_ALPHANUM.sub(' ', sentence)

    # Remove Single characters
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def load_imdb_data():
    movie_reviews = pd.read_csv("./data/IMDB Dataset.csv")
    movie_reviews['review'] = movie_reviews['review'].apply(preprocess_text)
    return movie_reviews



