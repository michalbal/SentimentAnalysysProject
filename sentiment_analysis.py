import loader
import pandas as pd
import numpy as np
import nltk
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
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
    words = sentence.split(" ")
    embeddings_sum = np.zeros(loader.W2V_EMBEDDING_DIM)
    num_relevant_words = 0
    for word in words:
        if word not in GENERAL_STOPWORDS and word in w2v_model:
            embeddings_sum = np.add(embeddings_sum, w2v_model[word])
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

    movie_reviews_encoded = movie_reviews['review'].apply(
        turn_sentence_to_encoding)

    sentiment = movie_reviews['sentiment'].map({'positive': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(
        movie_reviews_encoded, sentiment, test_size=0.2, random_state=42)

    imdb_svm = SGDClassifier()
    imdb_svm.fit(X_train, y_train)

    accuracy_imdb_on_imdb = imdb_svm.score(X_test, y_test)
    print("Imdb svm accuracy on test is: ", accuracy_imdb_on_imdb)

    return imdb_svm, "soon_to_be_model" , X_test, y_test

# ----------------------------------------------------

if __name__ == '__main__':
    # Get pretrained w2v model - it is of length 300, containing 100 million words
    w2v_model = api.load("word2vec-google-news-300")

    # Create general stopwords
    GENERAL_STOPWORDS = set(STOPWORDS)
    GENERAL_STOPWORDS.remove('not')
    GENERAL_STOPWORDS.remove('no')

    imdb_svm_model, imdb_logistic_model, imdb_x_test, imdb_y_test = create_imdb_models_and_split_data()



