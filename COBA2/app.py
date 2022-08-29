from flask import Flask, render_template, request, jsonify
import tweepy
import re

import os
import numpy as np
import tensorflow as tf

import official.nlp.bert.tokenization as tokenization
from official import nlp
from official.nlp import bert

from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

api_key = "PaO6a8FpsJLObvly8tKQeTUYJ"
api_secret_key = "ZWGEMqwd42eTi8OUVDPev3cmA15AlziNdQZrAFIRTZfbgM1XmU"
access_token = "1539800700133457920-KAAqfrUARmZG8Cg8A7VUgjDQHvIpV0"
access_token_secret = "NBylZNvqfz9QkAWfzXMj7s0NzYMtx3YUgzKo658NOnxqK"
consumer_key = "cjlLbm55OHM4c25RRUVrdVgyb1o6MT2pjaQ"
consumer_secret = "t6S4iNQxO3aDV9V2mHUumsog_z_zzZXO2z3to3YacklEK8vBjC"

model_fname = 'twitter_BERT'
my_wd = 'best/'

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
    my_wd = 'best/'

    encoder = LabelEncoder()
    encoder.classes_ = np.load(os.path.join(
        my_wd, encoder_fname), allow_pickle=True)

    tweet = [tweet_text]
    inputs = bert_encode(string_list=list(tweet),
                         tokenizer=tokenizerSaved,
                         max_seq_length=128)

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

    count_positif = 0
    count_negatif = 0

    for tweet in hasilsearch:
        tweet_bersih = ' '.join(re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text.lower()).split())

        tweet_sentiment = sentiment(tweet_bersih)

        tweet_json = {
            "tweet": tweet_bersih,
            "sentiment": tweet_sentiment["prediction"],
            "confidence": tweet_sentiment["confidence"]
        }

        positif = tweet_json["sentiment"].count("positif")
        if(tweet_json["sentiment"] == "positive"):
            count_positif += 1
        else:
            count_negatif += 1

        if tweet.retweet_count > 0:
            if tweet_bersih not in hasilanalisis:
                hasilanalisis.append(tweet_json)
        else:
            hasilanalisis.append(tweet_json)

    count_positif = int(count_positif/10*100)
    count_negatif = int(count_negatif/10*100)

    return jsonify({'htmlresponse': render_template('response1.html', results=hasilanalisis, positif=count_positif, negatif=count_negatif)})


if __name__ == "__main__":
    app.run(debug=True)
