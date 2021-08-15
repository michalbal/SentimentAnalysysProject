import loader
import pandas as pd
import numpy as np
import nltk
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, \
    plot_confusion_matrix
from sklearn.linear_model import LogisticRegression,SGDClassifier
import gensim.downloader as api
from sklearn.model_selection import train_test_split


# ---------------------------------------------------- Mutual to all
def turn_sentence_to_encoding(sentence: str):
    """
    Turns sentence into an average vector of the encodings of the words in the sentence.
    Not including stop words
    returns np array
    """
    words = sentence.split()
    embeddings_sum = np.zeros(loader.W2V_EMBEDDING_DIM)
    num_relevant_words = 0
    for word in words:
        if word not in GENERAL_STOPWORDS and word in w2v_model:
            embeddings_sum = embeddings_sum + w2v_model[word]
            num_relevant_words += 1

    if num_relevant_words == 0:
        return np.zeros(loader.W2V_EMBEDDING_DIM)

    return embeddings_sum / num_relevant_words


# Investigating the data
def show_word_cloud(texsts, stopwords = STOPWORDS ):
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color='black',
                          width=3000,
                          height=2500
                          ).generate(texsts)
    plt.figure(1, figsize=(12, 12))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


def create_model_and_split_data(reviews, data_name):
    reviews_encoded = [turn_sentence_to_encoding(sentance) for sentance in reviews['review']]

    sentiment = reviews['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        reviews_encoded, sentiment, test_size=0.2, random_state=42)

    svm_model = SGDClassifier()
    svm_model.fit(X_train, y_train)

    accuracy_svm_on_test = svm_model.score(X_test, y_test)
    print(data_name, " svm accuracy on test is: ", accuracy_svm_on_test)

    logistic_model = LogisticRegression(max_iter=300)
    logistic_model.fit(X_train, y_train)

    accuracy_logistic_on_test = logistic_model.score(X_test, y_test)
    print(data_name, " logistic accuracy on test is: ",
          accuracy_logistic_on_test)

    return svm_model, logistic_model, X_test, y_test


def show_models_results_on_data(first_model, second_model, third_model, fourth_model, x_test, y_test, comparison_name, data_name):
    print(comparison_name)

    show_model_success_on_dataset(first_model, "first_model", x_test, y_test)

    show_model_success_on_dataset(second_model, "second_model", x_test, y_test)

    show_model_success_on_dataset(third_model, "third_model", x_test, y_test)

    show_model_success_on_dataset(fourth_model, "fourth_model", x_test, y_test)

    # TODO add graph that shows all models together - Accuracy, recall, f1


def show_model_success_on_dataset(model, model_name, x_test, y_test):
    # Confusion Matrix is a table showing from left top: TP, FN, FP, TN
    y_predicted = model.predict(x_test)
    print(model_name, " accuracy: ",
          accuracy_score(y_predicted, y_test))
    cm = confusion_matrix(y_predicted, y_test)
    print(model_name, " confusion matrix is: ", cm)
    plt.figure()
    plot_confusion_matrix(model, x_test, y_test)
    plt.show()

# ---------------------------------------------------- IMDB data

def explore_imdb_data(movie_reviews):
    # Show how many positive and negative values exist
    print(movie_reviews['sentiment'].value_counts())

    positive_reviews = movie_reviews[movie_reviews['sentiment'] == 'positive']
    positive_words = ' '.join(positive_reviews['review'])
    show_word_cloud(positive_words)

    negative_reviews = movie_reviews[movie_reviews['sentiment'] == 'negative']
    negative_words = ' '.join(negative_reviews['review'])
    show_word_cloud(negative_words)

    # Now let's see the most common words without the words both Negative and Positive share
    imdb_stopwords = set(STOPWORDS).union(
        {'film', 'movie', 'one', 'character', 'time', 'see', 'make'})
    show_word_cloud(positive_words, imdb_stopwords)

    show_word_cloud(negative_words, imdb_stopwords)


def create_imdb_models_and_split_data():
    # Load imdb reviews data
    movie_reviews = loader.load_imdb_data()
    print(movie_reviews.head())
    # explore_imdb_data(movie_reviews)

    return create_model_and_split_data(movie_reviews, "IMDB")

# ---------------------------------------------------- Disney data


def explore_disney_data(disney_reviews):
    # Show how many positive and negative values exist
    print(disney_reviews['sentiment'].value_counts())

    positive_reviews = disney_reviews[disney_reviews['sentiment'] == 'positive']
    positive_words = ' '.join(positive_reviews['review'])
    show_word_cloud(positive_words)

    negative_reviews = disney_reviews[disney_reviews['sentiment'] == 'negative']
    negative_words = ' '.join(negative_reviews['review'])
    show_word_cloud(negative_words)

    # Now let's see the most common words without the words both Negative and Positive share
    disney_stopwords = set(STOPWORDS).union(
        {'ride', 'park', 'day', 'time', 'disneyland', 'disney', 'rides', 'one', 'go', 'kid'})
    show_word_cloud(positive_words, disney_stopwords)

    show_word_cloud(negative_words, disney_stopwords)


def create_disney_models_and_split_data():
    # Load disney reviews data
    disney_reviews = loader.load_disneyland_data()
    print(disney_reviews.head())
    # explore_disney_data(disney_reviews)

    return create_model_and_split_data(disney_reviews, "Disney")


# ---------------------------------------------------- Tweet data


def explore_tweets_data(tweets):
    # Show how many positive and negative values exist
    print(tweets['sentiment'].value_counts())

    positive_tweets = tweets[tweets['sentiment'] == 'positive']
    positive_words = ' '.join(positive_tweets['review'])
    show_word_cloud(positive_words)

    negative_tweets = tweets[tweets['sentiment'] == 'negative']
    negative_words = ' '.join(negative_tweets['review'])
    show_word_cloud(negative_words)


def create_tweets_models_and_split_data():
    # Load tweets data
    tweets = loader.load_tweets_data()
    print(tweets.head())
    explore_tweets_data(tweets)

    # return create_model_and_split_data(tweets, "Tweets")

# ---------------------------------------------------- Tweet data


def explore_amazon_data(reviews):
    # Show how many positive and negative values exist
    print(reviews['sentiment'].value_counts())

    positive_reviews = reviews[reviews['sentiment'] == 'positive']
    positive_words = ' '.join(positive_reviews['review'])
    show_word_cloud(positive_words)

    negative_reviews = reviews[reviews['sentiment'] == 'negative']
    negative_words = ' '.join(negative_reviews['review'])
    show_word_cloud(negative_words)

    # Now let's see the most common words without the words both Negative and Positive share
    amazon_stopwords = set(STOPWORDS).union(
        {'product', 'one', 'time'})
    show_word_cloud(positive_words, amazon_stopwords)

    show_word_cloud(negative_words, amazon_stopwords)


def create_amazon_models_and_split_data():
    # Load amazon reviews data
    reviews = loader.load_amazon_data()
    print(reviews.head())
    explore_amazon_data(reviews)

    # return create_model_and_split_data(reviews, "Amazon")

# ----------------------------------------------------

if __name__ == '__main__':
    # Get pretrained w2v model - it is of length 300, containing 100 million words
    w2v_model = api.load("word2vec-google-news-300")

    # Create general stopwords
    GENERAL_STOPWORDS = set(STOPWORDS)
    GENERAL_STOPWORDS.remove('not')
    GENERAL_STOPWORDS.remove('no')

    imdb_svm_model, imdb_logistic_model, imdb_x_test, imdb_y_test = create_imdb_models_and_split_data()

    disney_svm_model, disney_logistic_model, disney_x_test, disney_y_test = create_disney_models_and_split_data()
    show_model_success_on_dataset(imdb_svm_model, "imdb_svm_model", disney_x_test, disney_y_test)
    # create_tweets_models_and_split_data()
    # create_amazon_models_and_split_data()



