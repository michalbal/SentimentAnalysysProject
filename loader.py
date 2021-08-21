import pandas as pd
import re
import os
import gensim.downloader as api
import tensorflow_datasets as tfds
import pickle

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
    """
    Returns a pretrained w2v model, only on the words in the datasets.
    """
    w2v_model_path = "./models/w2v_model.pkl"
    if not os.path.exists(w2v_model_path):
        all_words = get_all_words()
        w2v = api.load("word2vec-google-news-300")
        w2v_embeddings = {word: w2v[word] for word in all_words if word in w2v}
        with open(w2v_model_path, "wb") as file:
            pickle.dump(w2v_embeddings, file)
        return w2v_embeddings
    else:
        with open(w2v_model_path, "rb") as file:
            return pickle.load(file)


def get_all_words():
    """
    Retrieve the words in all datasets
    :return: A set containing all the words in all the datasets
    """
    all_words = set()
    imdb_data = load_imdb_data()
    all_words = all_words.union(get_all_words_from_reviews(imdb_data))
    disney_data = load_disneyland_data()
    all_words = all_words.union(get_all_words_from_reviews(disney_data))
    amazon_data = load_amazon_data()
    all_words = all_words.union(get_all_words_from_reviews(amazon_data))
    tweets_data = load_tweets_data()
    all_words = all_words.union(get_all_words_from_reviews(tweets_data))
    return all_words


def get_all_words_from_reviews(dataset: pd.DataFrame):
    all_words = set()
    reviews = dataset['review']
    for review in reviews:
        words = set(review.split())
        all_words = all_words.union(words)

    return all_words


def load_disneyland_data():
    disney_reviews = pd.read_csv("./data/DisneylandReviews.csv")
    needed_columns_dict = dict()
    needed_columns_dict['review'] = disney_reviews['Review_Text'].apply(preprocess_text)
    # TODO currently anything below 5 is considered negative since disneyland is supposed to be super exciting. Also created pretty equal datasets
    needed_columns_dict['sentiment'] = disney_reviews['Rating'].apply(lambda score: "positive" if score > 3 else "negative")
    needed_reviews = pd.DataFrame(needed_columns_dict)
    return needed_reviews


def load_tweets_data():
    tweets = pd.read_csv("./data/Tweets.csv")
    needed_columns_dict = dict()

    # tweets['airline_sentiment'] = tweets['airline_sentiment'].apply(lambda sentiment: "positive" if sentiment == "neutral" or sentiment == "positive" else "negative")

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