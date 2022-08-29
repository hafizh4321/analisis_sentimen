# app.py
from flask import Flask, render_template, request, jsonify
# from flask_mysqldb import MySQL, MySQLdb  # pip install
import tweepy
import re
# flask-mysqldb https://github.com/alexferl/flask-mysqldb

import os
import numpy as np
import tensorflow as tf

import official.nlp.bert.tokenization as tokenization
from official import nlp
from official.nlp import bert

from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# app.secret_key = "caircocoders-ednalan"

# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'table'
# app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# mysql = MySQL(app)


api_key = "PaO6a8FpsJLObvly8tKQeTUYJ"
api_secret_key = "ZWGEMqwd42eTi8OUVDPev3cmA15AlziNdQZrAFIRTZfbgM1XmU"
access_token = "1539800700133457920-KAAqfrUARmZG8Cg8A7VUgjDQHvIpV0"
access_token_secret = "NBylZNvqfz9QkAWfzXMj7s0NzYMtx3YUgzKo658NOnxqK"
consumer_key = "cjlLbm55OHM4c25RRUVrdVgyb1o6MT2pjaQ"
consumer_secret = "t6S4iNQxO3aDV9V2mHUumsog_z_zzZXO2z3to3YacklEK8vBjC"

model_fname = 'twitter_BERT'
my_wd = 'dataset31/'

new_model = tf.keras.models.load_model(os.path.join(my_wd, model_fname))


@app.route('/')
def index():
    return render_template('index.html')


def sentiment(tweet_text):
    global model_fname, my_wd, new_model

    tokenizerSaved = bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(my_wd, model_fname, 'assets/vocab.txt'),
        do_lower_case=False)

    encoder_fname = 'twitter_classes.npy'
    my_wd = 'dataset31/'

    encoder = LabelEncoder()
    encoder.classes_ = np.load(os.path.join(
        my_wd, encoder_fname), allow_pickle=True)

    tweet = [tweet_text]
    inputs = bert_encode(string_list=list(tweet),
                         tokenizer=tokenizerSaved,
                         max_seq_length=254)

    prediction = new_model.predict(inputs)

    output = {
        'prediction': 'positive' if encoder.classes_[
            np.argmax(prediction)] == 1 else 'negative',
        'confidence': prediction
    }
    return output


def encode_names(n, tokenizer):
    tokens = list(tokenizer.tokenize(n))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(string_list, tokenizer, max_seq_length):
    num_examples = len(string_list)

    string_tokens = tf.ragged.constant([
        encode_names(n, tokenizer) for n in np.array(string_list)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*string_tokens.shape[0]
    input_word_ids = tf.concat([cls, string_tokens], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor(
        shape=(None, max_seq_length))

    type_cls = tf.zeros_like(cls)
    type_tokens = tf.ones_like(string_tokens)
    input_type_ids = tf.concat(
        [type_cls, type_tokens], axis=-1).to_tensor(shape=(None, max_seq_length))

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(shape=(None, max_seq_length)),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs


@app.route("/api/tweet", methods=["GET"])
def index1():
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    hasilsearch = api.search_tweets(
        q=request.args.get('query'), lang="id", count=10)

    hasilanalisis = []

    for tweet in hasilsearch:
        tweet_properties = {}
        tweet_properties["isi_tweet"] = tweet.text
        tweet_bersih = ' '.join(re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text).split())

        tweet_sentiment = sentiment(tweet_bersih)

        tweet_json = {
            "tweet": tweet_bersih,
            "sentiment": tweet_sentiment["prediction"],
            "confidence": tweet_sentiment["confidence"]
        }

        if tweet.retweet_count > 0:
            if tweet_bersih not in hasilanalisis:
                hasilanalisis.append(tweet_json)
        else:
            hasilanalisis.append(tweet_json)

    return jsonify({'htmlresponse': render_template('response1.html', results=hasilanalisis)})

    # return jsonify(results=hasilanalisis)


# @app.route("/ajaxlivesearch", methods=["POST", "GET"])
# def ajaxlivesearch():
#     cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#     if request.method == 'POST':
#         search_word = request.form['query']
#         print(search_word)
#         if search_word == '':
#             query = "SELECT * from employee ORDER BY no"
#             cur.execute(query)
#             employee = cur.fetchall()
#         else:
#             query = "SELECT * from employee WHERE tweet LIKE '%{}%' OR sentiment LIKE '%{}%' OR confidence LIKE '%{}%' ORDER BY no DESC LIMIT 20".format(
#                 search_word, search_word, search_word)
#             cur.execute(query)
#             numrows = int(cur.rowcount)
#             employee = cur.fetchall()
#             print(numrows)
#     return jsonify({'htmlresponse': render_template('response1.html', employee=employee, numrows=numrows)})


if __name__ == "__main__":
    app.run(debug=True)
