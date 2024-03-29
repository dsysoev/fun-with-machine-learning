import pickle

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_test_split_data(categories, random_state=42):
    # get train data
    train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=random_state)
    x_train = train.data
    y_train = (train.target == 1).astype(int)
    # get test
    test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=random_state)
    x_test = test.data
    y_test = (test.target == 1).astype(int)

    return x_train, x_test, y_train, y_test


def preprocess(x, vector):
    # simple preprocessing
    x_train = pd.DataFrame(vector.fit_transform(x).todense(), columns=vector.get_feature_names_out())
    return x_train


def fit_and_save(model, x, y, top_words=None):
    # create vocabulary with order by tf-idf
    vector = TfidfVectorizer(stop_words='english')
    x_train = preprocess(x, vector)
    vocab = x_train.sum().sort_values(ascending=False).index.tolist()
    # select top words
    vocab = vocab if top_words is None else vocab[:top_words]
    # create vocabulary again based on top words
    vector = TfidfVectorizer(stop_words='english', vocabulary=vocab)
    x_train = preprocess(x, vector)
    model.fit(x_train, y)
    # save model and vector
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vector.pkl", "wb") as f:
        pickle.dump(vector, f)


def load_vector_and_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vector.pkl", "rb") as f:
        vector = pickle.load(f)

    return vector, model


def load_predict_and_score(x, y):
    # load
    vector, model = load_vector_and_model()
    x_test = preprocess(x, vector)
    # predict
    predict = model.predict_proba(x_test)[:, 1]
    # and score
    roc = roc_auc_score(y, predict)
    return roc


def main(top_words=60):
    categories = ['comp.graphics',
                  'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware',
                  'comp.windows.x']
    # set the model
    model = LogisticRegression(C=10)
    top_words = 1000 if top_words is None else top_words
    # load data
    x_train, x_test, y_train, y_test = train_test_split_data(categories=categories)
    # save fitted model
    fit_and_save(model, x_train, y_train, top_words)
    # calc score
    score = load_predict_and_score(x_test, y_test)
    print("model: {}, top_words: {}, roc auc: {:.2f}".format(model.__class__.__name__, top_words, score))
    return score


def select_optimal_top_words():
    top_words_list = [3, 10, 30, 50, 60, 70, 80, 100, 300, 1000]
    score = [{'score': main(i), 'top_word': i} for i in top_words_list]
    return pd.DataFrame(score)


def get_model_coef():
    vector, model = load_vector_and_model()
    s = pd.DataFrame(model.coef_, columns=vector.get_feature_names_out()).T
    s['abs'] = s[0].abs()
    s = s.sort_values('abs', ascending=False)
    s.drop('abs', axis=1, inplace=True)
    return s


if __name__ == '__main__':
    main()
