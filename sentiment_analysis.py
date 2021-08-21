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


def create_model_and_split_data(reviews, data_name):
    reviews_encoded = [turn_sentence_to_encoding(sentance) for sentance in reviews['review']]

    sentiment = reviews['sentiment'].map({'positive': 1, 'negative': 0})

    # TODO add limit to smallest dataset size if needed

    X_train, X_test, y_train, y_test = train_test_split(
        reviews_encoded, sentiment, test_size=0.2, random_state=42)

    svm_model_path = ".\models\\"+ data_name + "_svm_model.pkl"
    if not os.path.exists(svm_model_path):
        svm_model = SGDClassifier()
        svm_model.fit(X_train, y_train)
        save_model(svm_model, svm_model_path)
    else:
        svm_model = load_model(svm_model_path)

    accuracy_svm_on_test = svm_model.score(X_test, y_test)
    print(data_name, " svm accuracy on test is: ", accuracy_svm_on_test)

    logistic_model_path = ".\models\\" + data_name + "_logistic_model.pkl"
    if not os.path.exists(logistic_model_path):
        logistic_model = LogisticRegression(max_iter=300)
        logistic_model.fit(X_train, y_train)
        save_model(logistic_model, logistic_model_path)
    else:
        logistic_model = load_model(logistic_model_path)

    accuracy_logistic_on_test = logistic_model.score(X_test, y_test)
    print(data_name, " logistic accuracy on test is: ",
          accuracy_logistic_on_test)

    return svm_model, logistic_model, X_test, y_test


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def show_models_results_on_data(models, model_names, x_test, y_test, comparison_name, data_name):
    print(comparison_name)

    # Consider sending as two lists: models, model_names

    for i in range(len(models)):
        show_model_success_on_dataset(models[i], model_names[i], x_test, y_test)

    compare_models_results_via_plot(models, model_names, x_test, y_test, data_name)


def show_model_success_on_dataset(model, model_name, x_test, y_test):
    # Confusion Matrix is a table showing from left top: TP, FN, FP, TN
    y_predicted = model.predict(x_test)
    print(model_name, " accuracy: ",
          accuracy_score(y_predicted, y_test))
    cm = confusion_matrix(y_predicted, y_test)
    print(model_name, " confusion matrix is: \n", cm)
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
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()
    plot_path = './plots/' + data_name +'_roc_curve.png'
    fig.savefig(plot_path)


# ---------------------------------------------------- IMDB data

def explore_imdb_data(movie_reviews):
    # Show how many positive and negative values exist
    print(movie_reviews['sentiment'].value_counts())

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
    print(movie_reviews.head())
    # explore_imdb_data(movie_reviews)

    return create_model_and_split_data(movie_reviews, "IMDB")

# ---------------------------------------------------- Disney data


def explore_disney_data(disney_reviews):
    # Show how many positive and negative values exist
    print(disney_reviews['sentiment'].value_counts())

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
    print(disney_reviews.head())
    explore_disney_data(disney_reviews)

    return create_model_and_split_data(disney_reviews, "disney")


# ---------------------------------------------------- Tweet data


def explore_tweets_data(tweets):
    # Show how many positive and negative values exist
    print(tweets['sentiment'].value_counts())

    positive_tweets = tweets[tweets['sentiment'] == 'positive']
    positive_words = ' '.join(positive_tweets['review'])
    show_word_cloud(positive_words, "tweets_positive_all_stopwords")

    negative_tweets = tweets[tweets['sentiment'] == 'negative']
    negative_words = ' '.join(negative_tweets['review'])
    show_word_cloud(negative_words, "tweets_negative_all_stopwords")


def create_tweets_models_and_split_data():
    # Load tweets data
    tweets = loader.load_tweets_data()
    print(tweets.head())
    explore_tweets_data(tweets)

    return create_model_and_split_data(tweets, "tweets")

# ---------------------------------------------------- Tweet data


def explore_amazon_data(reviews):
    # Show how many positive and negative values exist
    print(reviews['sentiment'].value_counts())

    positive_reviews = reviews[reviews['sentiment'] == 'positive']
    positive_words = ' '.join(positive_reviews['review'])
    show_word_cloud(positive_words, "amazon_positive_all_stopwords")

    negative_reviews = reviews[reviews['sentiment'] == 'negative']
    negative_words = ' '.join(negative_reviews['review'])
    show_word_cloud(negative_words, "amazon_negative_all_stopwords")

    # Now let's see the most common words without the words both Negative and Positive share
    amazon_stopwords = set(STOPWORDS).union(
        {'product', 'one', 'time'})
    show_word_cloud(positive_words, "amazon_positive_cleared_mutual_stopwords", amazon_stopwords)

    show_word_cloud(negative_words, "amazon_negative_cleared_mutual_stopwords", amazon_stopwords)


def create_amazon_models_and_split_data():
    # Load amazon reviews data
    reviews = loader.load_amazon_data()
    print(reviews.head())
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
    # TODO remove cumulative undesirable stopwords revealed during word cloud - will improve performance

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
                                disney_y_test, "Comparing SVM models on disney",
                                "disney_svm")

    # Compare all logistic models on disney data
    show_models_results_on_data(logistic_models, logistic_models_names,
                                disney_x_test,
                                disney_y_test,
                                "Comparing Logistic models on disney",
                                "disney_logistic")

    # Compare all svm models on tweets data

    show_models_results_on_data(svm_models, svm_models_names, tweets_x_test,
                                tweets_y_test, "Comparing SVM models on tweets",
                                "tweets_svm")
    # Compare all logistic models on tweets data
    show_models_results_on_data(logistic_models, logistic_models_names,
                                tweets_x_test,
                                tweets_y_test,
                                "Comparing Logistic models on tweets",
                                "tweets_logistic")

    # Compare all svm models on amazon data

    show_models_results_on_data(svm_models, svm_models_names, amazon_x_test,
                                amazon_y_test, "Comparing SVM models on amazon",
                                "amazon_svm")
    # Compare all logistic models on amazon data
    show_models_results_on_data(logistic_models, logistic_models_names,
                                amazon_x_test,
                                amazon_y_test,
                                "Comparing Logistic models on amazon",
                                "amazon_logistic")



