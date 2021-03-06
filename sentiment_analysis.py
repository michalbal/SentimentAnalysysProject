import loader
import pandas as pd
import numpy as np
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, \
    roc_curve, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
import pickle
import os


# ---------------------------------------------------- Mutual to all

MAX_SENTIMENT_SIZE = 2363
MAX_REVIEW_LENGTH = 50


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


def turn_sentence_to_encoding_limiting_length(sentence: str):
    """
    Turns sentence into an average vector of the encodings of the words in the sentence.
    Not including stop words
    Limits sentance length to 50
    returns np array
    """
    words = sentence.split()
    embeddings_sum = np.zeros(loader.W2V_EMBEDDING_DIM)
    num_relevant_words = 0
    length = min(MAX_REVIEW_LENGTH, len(words))
    for i in range(length):
        word = words[i]
        if word not in GENERAL_STOPWORDS and word in w2v_model:
            embeddings_sum = embeddings_sum + w2v_model[word]
            num_relevant_words += 1

    if num_relevant_words == 0:
        return np.zeros(loader.W2V_EMBEDDING_DIM)

    return embeddings_sum / num_relevant_words


# Investigating the data
def show_word_cloud(texsts, data_name, stopwords=STOPWORDS):
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color='black',
                          width=3000,
                          height=2500
                          ).generate(texsts)
    fig = plt.figure(1, figsize=(12, 12))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    plot_path = './plots/' + data_name + "_" + '_wordcloud.png'
    fig.savefig(plot_path)


def limit_dataset_size(reviews):
    """
    Returns a dataset where the amounts of samples that are positive
    and negative are limited by the number of positive and negative samples of
    the samllest dataset.
    """
    positive_reviews = reviews[reviews['sentiment'] == 'positive'].head(MAX_SENTIMENT_SIZE)
    negative_reviews = reviews[reviews['sentiment'] == 'negative'].head(MAX_SENTIMENT_SIZE)
    return pd.concat([positive_reviews, negative_reviews])


def make_positive_and_negative_equal_size(reviews, data_name):
    """
    Returns a dataset where the amounts of samples that are positive
    and negative are equal.
    """
    sentiment_size = min(reviews['sentiment'].value_counts())
    print("Both positive and negative amounts in ", data_name, " is: ", sentiment_size)
    positive_reviews = reviews[reviews['sentiment'] == 'positive'].head(
        sentiment_size)
    negative_reviews = reviews[reviews['sentiment'] == 'negative'].head(
        sentiment_size)
    return pd.concat([positive_reviews, negative_reviews])


def create_model_and_split_data(reviews, data_name):

    # limited_reviews = limit_dataset_size(reviews)

    # limited_reviews = make_positive_and_negative_equal_size(reviews, data_name)

    limited_reviews = reviews

    reviews_encoded = [turn_sentence_to_encoding(sentance) for sentance in limited_reviews['review']]

    sentiment = limited_reviews['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        reviews_encoded, sentiment, test_size=0.2, random_state=42)

    svm_model_path = ".\models\\"+ data_name + "_svm_model.pkl"
    if not os.path.exists(svm_model_path):
        svm_model = SGDClassifier()
        svm_model.fit(X_train, y_train)
        save_model(svm_model, svm_model_path)
    else:
        svm_model = load_model(svm_model_path)

    logistic_model_path = ".\models\\" + data_name + "_logistic_model.pkl"
    if not os.path.exists(logistic_model_path):
        logistic_model = LogisticRegression(max_iter=300)
        logistic_model.fit(X_train, y_train)
        save_model(logistic_model, logistic_model_path)
    else:
        logistic_model = load_model(logistic_model_path)

    return svm_model, logistic_model, X_test, y_test


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def show_models_results_on_data(models, model_names, x_test, y_test, comparison_name, data_name):
    print(comparison_name)

    for i in range(len(models)):
        show_model_success_on_dataset(models[i], model_names[i], x_test, y_test, data_name)

    compare_models_results_via_plot(models, model_names, x_test, y_test, data_name)


def show_model_success_on_dataset(model, model_name, x_test, y_test, data_name):
    y_predicted = model.predict(x_test)
    print(model_name, " accuracy: ",
          accuracy_score(y_predicted, y_test), " on ", data_name)
    print(classification_report(y_test, y_predicted))


def compare_models_results_via_plot(models, model_names, x_test, y_test, data_name):
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    for i in range(len(models)):
        model = models[i]
        model_name = model_names[i]
        yproba = model.decision_function(x_test)
        fpr, tpr, _ = roc_curve(y_test, yproba)
        auc = roc_auc_score(y_test, yproba)
        result_table = result_table.append(
            {'classifiers': model_name,
             'fpr': fpr,
             'tpr': tpr,
             'auc': auc}, ignore_index=True)

    # Set the names of the models as index labels
    result_table.set_index('classifiers', inplace=True)
    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    # Create the plot
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis ' + data_name, fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()
    plot_path = './plots/' + data_name +'_roc_curve.png'
    fig.savefig(plot_path)


# ---------------------------------------------------- IMDB data

IMDB_STOPWORDS = {'film', 'movie', 'one', 'character', 'time', 'see',
                      'make', 'director', 'play', 'scene'}


def explore_imdb_data(movie_reviews):
    positive_reviews = movie_reviews[movie_reviews['sentiment'] == 'positive']
    positive_words = ' '.join(positive_reviews['review'])
    show_word_cloud(positive_words, "IMDB_positive_all_stopwords")

    negative_reviews = movie_reviews[movie_reviews['sentiment'] == 'negative']
    negative_words = ' '.join(negative_reviews['review'])
    show_word_cloud(negative_words, "IMDB_negative_all_stopwords")

    # Now let's see the most common words without the words both Negative and Positive share
    imdb_stopwords = set(STOPWORDS).union(
        {'film', 'movie', 'one', 'character', 'time', 'see', 'make'})
    show_word_cloud(positive_words, "IMDB_positive_cleared_mutual_stopwords",imdb_stopwords)

    show_word_cloud(negative_words, "IMDB_negative_cleared_mutual_stopwords", imdb_stopwords)


def create_imdb_models_and_split_data():
    # Load imdb reviews data
    movie_reviews = loader.load_imdb_data()
    # print(movie_reviews.head())

    # Show how many positive and negative values exist
    print("IMDB counts:")
    print(movie_reviews['sentiment'].value_counts())
    # explore_imdb_data(movie_reviews)

    return create_model_and_split_data(movie_reviews, "IMDB")

# ---------------------------------------------------- Disney data


DISNEY_STOPWORDS = {'ride', 'park', 'day', 'time', 'disneyland', 'disney',
                        'rides', 'one', 'go', 'kid', 'kids'}


def explore_disney_data(disney_reviews):
    positive_reviews = disney_reviews[disney_reviews['sentiment'] == 'positive']
    positive_words = ' '.join(positive_reviews['review'])
    show_word_cloud(positive_words, "disney_positive_all_stopwords")

    negative_reviews = disney_reviews[disney_reviews['sentiment'] == 'negative']
    negative_words = ' '.join(negative_reviews['review'])
    show_word_cloud(negative_words, "disney_negative_all_stopwords")

    # Now let's see the most common words without the words both Negative and Positive share
    disney_stopwords = set(STOPWORDS).union(
        {'ride', 'park', 'day', 'time', 'disneyland', 'disney', 'rides', 'one', 'go', 'kid'})
    show_word_cloud(positive_words, "disney_positive_cleared_mutual_stopwords", disney_stopwords)

    show_word_cloud(negative_words, "disney_negative_cleared_mutual_stopwords", disney_stopwords)


def create_disney_models_and_split_data():
    # Load disney reviews data
    disney_reviews = loader.load_disneyland_data()
    # print(disney_reviews.head())

    # Show how many positive and negative values exist
    print("Disney counts:")
    print(disney_reviews['sentiment'].value_counts())
    # explore_disney_data(disney_reviews)

    return create_model_and_split_data(disney_reviews, "disney")


# ---------------------------------------------------- Tweet data

TWITTER_STOPWORDS = {'airline', 'airport', 'fly', 'plane', 'pilot', 'flight'}


def explore_tweets_data(tweets):

    positive_tweets = tweets[tweets['sentiment'] == 'positive']
    positive_words = ' '.join(positive_tweets['review'])
    show_word_cloud(positive_words, "tweets_positive_all_stopwords")

    negative_tweets = tweets[tweets['sentiment'] == 'negative']
    negative_words = ' '.join(negative_tweets['review'])
    show_word_cloud(negative_words, "tweets_negative_all_stopwords")


def create_tweets_models_and_split_data():
    # Load tweets data
    tweets = loader.load_tweets_data()
    # print(tweets.head())

    # Show how many positive and negative values exist
    print("Tweets counts:")
    print(tweets['sentiment'].value_counts())
    # explore_tweets_data(tweets)

    return create_model_and_split_data(tweets, "tweets")

# ---------------------------------------------------- Tweet data


AMAZON_STOPWORDS = {'product', 'one', 'time', 'use', 'amazon'}


def explore_amazon_data(reviews):
    positive_reviews = reviews[reviews['sentiment'] == 'positive']
    positive_words = ' '.join(positive_reviews['review'])
    show_word_cloud(positive_words, "amazon_positive_all_stopwords")

    negative_reviews = reviews[reviews['sentiment'] == 'negative']
    negative_words = ' '.join(negative_reviews['review'])
    show_word_cloud(negative_words, "amazon_negative_all_stopwords")

    # Now let's see the most common words without the words both Negative and Positive share
    amazon_stopwords = set(STOPWORDS).union(
        {'product', 'one', 'time', 'use', 'amazon'})
    show_word_cloud(positive_words, "amazon_positive_cleared_mutual_stopwords", amazon_stopwords)

    show_word_cloud(negative_words, "amazon_negative_cleared_mutual_stopwords", amazon_stopwords)


def create_amazon_models_and_split_data():
    # Load amazon reviews data
    reviews = loader.load_amazon_data()
    # print(reviews.head())

    # Show how many positive and negative values exist
    print("Amazon counts:")
    print(reviews['sentiment'].value_counts())
    # explore_amazon_data(reviews)

    return create_model_and_split_data(reviews, "amazon")

# ----------------------------------------------------


if __name__ == '__main__':

    # Get pretrained w2v model - length 300, containing 100 million words
    # but we only use the words in the datasets for easier use
    w2v_model = loader.load_pretrained_w2v()

    # Create general stopwords
    GENERAL_STOPWORDS = set(STOPWORDS)
    GENERAL_STOPWORDS.remove('not')
    GENERAL_STOPWORDS.remove('no')

    # Adding stopwords specific for these datasets
    GENERAL_STOPWORDS = GENERAL_STOPWORDS.union(IMDB_STOPWORDS)
    GENERAL_STOPWORDS = GENERAL_STOPWORDS.union(DISNEY_STOPWORDS)
    GENERAL_STOPWORDS = GENERAL_STOPWORDS.union(TWITTER_STOPWORDS)
    GENERAL_STOPWORDS = GENERAL_STOPWORDS.union(AMAZON_STOPWORDS)

    imdb_svm_model, imdb_logistic_model, imdb_x_test, imdb_y_test = create_imdb_models_and_split_data()

    disney_svm_model, disney_logistic_model, disney_x_test, disney_y_test = create_disney_models_and_split_data()

    tweets_svm_model, tweets_logistic_model, tweets_x_test, tweets_y_test = create_tweets_models_and_split_data()

    amazon_svm_model, amazon_logistic_model, amazon_x_test, amazon_y_test = create_amazon_models_and_split_data()

    svm_models = [imdb_svm_model, disney_svm_model, tweets_svm_model, amazon_svm_model]
    svm_models_names = ["imdb_svm_model", "disney_svm_model", "tweets_svm_model", "amazon_svm_model"]

    logistic_models = [imdb_logistic_model, disney_logistic_model, tweets_logistic_model,
                  amazon_logistic_model]

    logistic_models_names = ["imdb_logistic_model", "disney_logistic_model",
                       "tweets_logistic_model",
                       "amazon_logistic_model"]

    # Compare all svm models on IMDB data
    show_models_results_on_data(svm_models, svm_models_names, imdb_x_test, imdb_y_test, "Comparing SVM models on IMDB", "IMDB_SVM")

    # Compare all logistic models on IMDB data
    show_models_results_on_data(logistic_models, logistic_models_names, imdb_x_test,
                                imdb_y_test, "Comparing Logistic models on IMDB",
                                "IMDB_logistic")

    # Compare all svm models on disney data
    show_models_results_on_data(svm_models, svm_models_names, disney_x_test,
                                disney_y_test, "Comparing SVM models on Disney",
                                "Disney_SVM")

    # Compare all logistic models on disney data
    show_models_results_on_data(logistic_models, logistic_models_names,
                                disney_x_test,
                                disney_y_test,
                                "Comparing Logistic models on Disney",
                                "Disney_logistic")

    # Compare all svm models on tweets data

    show_models_results_on_data(svm_models, svm_models_names, tweets_x_test,
                                tweets_y_test, "Comparing SVM models on Tweets",
                                "Tweets_SVM")
    # Compare all logistic models on tweets data
    show_models_results_on_data(logistic_models, logistic_models_names,
                                tweets_x_test,
                                tweets_y_test,
                                "Comparing Logistic models on Tweets",
                                "Tweets_logistic")

    # Compare all svm models on amazon data

    show_models_results_on_data(svm_models, svm_models_names, amazon_x_test,
                                amazon_y_test, "Comparing SVM models on Amazon",
                                "Amazon_SVM")
    # Compare all logistic models on amazon data
    show_models_results_on_data(logistic_models, logistic_models_names,
                                amazon_x_test,
                                amazon_y_test,
                                "Comparing Logistic models on Amazon",
                                "Amazon_logistic")



