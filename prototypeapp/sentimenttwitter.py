import datetime
import tweepy
from textblob import TextBlob

access_token = '2385032364-hI4GtO7tHGcLVX9tnHRl6xvBPkZ0QcHKnOa2cYC'
access_token_secret = 'Evi0TRm5iWaMpiZUodgAtlhJnGjrGyxT90LEgeCY6xJCO'
consumer_key = 'HGmh0LcGCT5QkyW8vXa1O5tDc'
consumer_secret = 'vIcPL9iU9ag0sz0vQPVkiwwh3A4mGvRDujRlLnJBuHXrESA3eg'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

[posi, nega, neutral] = [0, 0, 0]
sent1 = ''
result1 = 0


def query_twitter(q, max_tweets=100):
    for tweet in tweepy.Cursor(api.search, q=q).items(max_tweets):
        if (datetime.datetime.now() - tweet.created_at).days < 1:
            #sentiment_data(tweet.text)
            global posi
            global nega
            global neutral

            analysis = TextBlob(tweet.text)
            if analysis.sentiment.polarity > 0:
                print('Positive: ' + tweet.text + '\n')
                posi = posi + 1
            elif analysis.sentiment.polarity < 0:
                print('Negative: ' + tweet.text + '\n')
                nega = nega + 1
            elif analysis.sentiment.polarity == 0:
                print('Neutral' + tweet.text + '\n')
                neutral = neutral + 1
            #return sentiment_percent(posi, nega, neutral)

            global sent1
            global result1
            total = posi + nega + neutral
            if posi > nega and posi > neutral:
                sent1 = "Positive"
                try:
                    result1 = posi / total * 100
                except:
                    print("An error occurred")
            elif nega > posi and nega > neutral:
                sent1 = "Negative"
                try:
                    result1 = nega / total * 100
                except:
                    print("An error occurred")
            else:
                sent1 = "Neutral"
                try:
                    result1 = neutral / total * 100
                except:
                    print("An error occurred")

    print("in func",[posi, nega, neutral])
    print("in func Subject: " + "'" + q + "'" + " is " + str("{0:.2f}".format(result1)) + "% " + sent1)

    return [posi, nega, neutral, str("{0:.2f}".format(result1)) + "% " + sent1, "'" + q + "'" + " is " + str("{0:.2f}".format(result1)) + "% " + sent1]

# ans = []
# ans = query_twitter('TCS', 100)
# print('after calling',ans)

