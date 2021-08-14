import pandas as pd
import re
import os
import gensim.downloader as api
import tensorflow_datasets as tfds

HTML_TAG_RE = re.compile(r'<[^>]+>')
AT_RE = re.compile(r'@[\w]+')
HASHTAG_RE = re.compile(r'@[\w]+')
NON_ALPHANUM = re.compile(r'#[\W]')

W2V_EMBEDDING_DIM = 300


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

    # Needed for amazon reviews
    sentence = re.sub('b ', '', sentence)

    return sentence


def load_imdb_data():
    movie_reviews = pd.read_csv("./data/IMDB Dataset.csv")
    movie_reviews['review'] = movie_reviews['review'].apply(preprocess_text)
    return movie_reviews


def load_pretrained_w2v():
    w2v = api.load("word2vec-google-news-300")
    return w2v


def load_disneyland_data():
    disney_reviews = pd.read_csv("./data/DisneylandReviews.csv")
    needed_columns_dict = dict()
    needed_columns_dict['review'] = disney_reviews['Review_Text'].apply(preprocess_text)
    # TODO currently anything below 5 is considered negative since disneyland is supposed to be super exciting. Also created pretty equal datasets
    needed_columns_dict['sentiment'] = disney_reviews['Rating'].apply(lambda score: "positive" if score > 4 else "negative")
    needed_reviews = pd.DataFrame(needed_columns_dict)
    return needed_reviews


def load_tweets_data():
    tweets = pd.read_csv("./data/Tweets.csv")
    needed_columns_dict = dict()
    tweets = tweets[tweets['airline_sentiment'] != 'neutral']
    needed_columns_dict['review'] = tweets['text'].apply(preprocess_text)
    needed_columns_dict['sentiment'] = tweets['airline_sentiment']
    needed_reviews = pd.DataFrame(needed_columns_dict)
    return needed_reviews


def load_amazon_data():
    # Needed to download the first time
    # reviews_ds = tfds.load('amazon_us_reviews/Personal_Care_Appliances_v1_00', split='train', shuffle_files=True)
    # reviews = tfds.as_dataframe(reviews_ds)
    # reviews.to_csv("./data/amazon_reviews.csv")
    reviews = pd.read_csv("./data/amazon_reviews.csv")
    needed_columns_dict = dict()
    needed_columns_dict['review'] = reviews['data/review_body'].apply(preprocess_text)
    needed_columns_dict['sentiment'] = reviews['data/star_rating'].apply(lambda score: "positive" if score > 3 else "negative")
    needed_reviews = pd.DataFrame(needed_columns_dict)
    return needed_reviews