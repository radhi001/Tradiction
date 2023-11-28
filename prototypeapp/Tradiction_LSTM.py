import datetime
import tweepy
from textblob import TextBlob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

access_token = '2385032364-hI4GtO7tHGcLVX9tnHRl6xvBPkZ0QcHKnOa2cYC'
access_token_secret = 'Evi0TRm5iWaMpiZUodgAtlhJnGjrGyxT90LEgeCY6xJCO'
consumer_key = 'HGmh0LcGCT5QkyW8vXa1O5tDc'
consumer_secret = 'vIcPL9iU9ag0sz0vQPVkiwwh3A4mGvRDujRlLnJBuHXrESA3eg'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def query_twitter(q, max_tweets=100, days_limit=1):
    posi, nega, neutral = 0, 0, 0

    for tweet in tweepy.Cursor(api.search, q=q).items(max_tweets):
        if (datetime.datetime.now() - tweet.created_at).days < days_limit:
            analysis = TextBlob(tweet.text)
            if analysis.sentiment.polarity > 0:
                posi += 1
            elif analysis.sentiment.polarity < 0:
                nega += 1
            elif analysis.sentiment.polarity == 0:
                neutral += 1

    total = posi + nega + neutral
    if total == 0:
        return 0
    sentiment_score = (posi - nega) / total
    return sentiment_score

data = pd.read_csv('your_stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

target_column = 'Close/Last'
data = data[[target_column]]

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

data['Sentiment'] = data.index.map(lambda x: query_twitter(f'{your_stock_symbol} stock', days_limit=1))

data['Sentiment'].fillna(data['Sentiment'].mean(), inplace=True)

train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

X_train_with_sentiment = np.concatenate((X_train, np.expand_dims(data['Sentiment'].values[sequence_length:], axis=1)), axis=2)

model.fit(X_train_with_sentiment, y_train, epochs=50, batch_size=64, verbose=1)

X_test_with_sentiment = np.concatenate((X_test, np.expand_dims(data['Sentiment'].values[-len(X_test):], axis=1)), axis=2)

predicted_values = model.predict(X_test_with_sentiment)

predicted_values = scaler.inverse_transform(predicted_values)
y_test = scaler.inverse_transform(y_test)

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Stock Prices')
plt.plot(predicted_values, label='Predicted Stock Prices')
plt.legend()
plt.show()
