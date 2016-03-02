import re
import csv
import pickle
from datetime import datetime
import pytz
import nltk


class Tweet(object):
    ID = 0
    topic = ""
    text = ""
    date = None
    sentiment = ""
    prediction = None

    def __init__(self, ID, topic, text, date, sentiment):
        self.ID = ID
        self.topic = topic
        self.text = text
        self.date = datetime.strptime(date,'%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=pytz.UTC)
        self.sentiment = sentiment


def parse_serialize_rawdata():
    # don't preprocess before serializing so we don't need to re-serialize every time we try something new
    # save all tweets in a single file so we reduce file reads
    tweets = []

    with open('full-corpus.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', escapechar='\\')
        for row in reader:
            try:
                # remove byte object notation, i.e. "b'text'" -> "text"
                row = [cell[2:-1] for cell in row]
                tweets.append(Tweet(ID=row[2], topic=row[0], text=row[4], date=row[3], sentiment=row[1]))
            except ValueError as e:
                # skip header row
                print(e)
                continue
        pickle.dump(tweets, open('tweets.p', 'wb'))


def preprocess(tweets):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')

    for t in tweets:
        t.text = t.text.lower()  # convert to lower case
        # replace user mentions (max 15 characters [A-Za-Z0-9_]) and hyperlinks, don't remove to preserve word order
        t.text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', 'USER_MENTION', t.text)
        t.text = re.sub(r'https?://\S+', 'URL', t.text)
        t.text = re.sub(r'www\.\S+', 'URL', t.text)

        # auto correct?

        # stem words
        t.text = stemmer.stem(t.text)

        # hesitant about removing stop words since they make more of a difference in sentiment analysis
        # examples off top of my head "the shit" vs "shit", "not good" vs "good", "fuck with it" vs "fuck it"
        # I'll try to remove some stop words selectively later on to see if it can improve accuracy
        # from nltk.corpus import stopwords
        # sw = set(stopwords.words("english"))  # set to prevent duplicates and faster "in" operations


class BayesSentimentClassifier(object):
    trained_ngram_sentiment_dictionary = {}
    positive_count = 0
    neutral_count = 0
    irrelevant_count = 0
    negative_count = 0

    def train(self, tweets):
        for t in tweets:
            words = nltk.word_tokenize(t.text)
            ngrams = nltk.ngrams(words, 2)  # ngrams, will adjust n to see which provides the most accuracy
            sentiment = t.sentiment
            for ngram in ngrams:
                if ngram not in self.trained_ngram_sentiment_dictionary:
                    self.trained_ngram_sentiment_dictionary[ngram] = [0, 0, 0, 0]
                if sentiment == 'positive':
                    self.trained_ngram_sentiment_dictionary[ngram][0] += 1
                    self.positive_count += 1
                elif sentiment == 'neutral':
                    self.trained_ngram_sentiment_dictionary[ngram][1] += 1
                    self.neutral_count += 1
                elif sentiment == 'irrelevant':
                    self.trained_ngram_sentiment_dictionary[ngram][2] += 1
                    self.irrelevant_count += 1
                elif sentiment == 'negative':
                    self.trained_ngram_sentiment_dictionary[ngram][3] += 1
                    self.negative_count += 1

    def predict(self, tweets):
        # implement Bayes Theorem: P(A|B) = (P(B|A) * P(A)) / P(B)

        # P(A) = probability that a tweet is <sentiment>
        total_tweets = self.positive_count + self.neutral_count + self.irrelevant_count + self.negative_count
        prob_positive = self.positive_count / total_tweets
        prob_neutral = self.neutral_count / total_tweets
        prob_irrelevant = self.irrelevant_count / total_tweets
        prob_negative = self.negative_count / total_tweets

        total_ngrams = 0
        for ngram_key in self.trained_ngram_sentiment_dictionary:
            total_ngrams += sum(self.trained_ngram_sentiment_dictionary[ngram_key])

        for t in tweets:
            words = nltk.word_tokenize(t.text)
            ngrams = nltk.ngrams(words, 2)

            cum_pos_prob = 0
            cum_neutral_prob = 0
            cum_irrelevant_prob = 0
            cum_negative_prob = 0

            tweet_ngrams = 0  # can't len(generator()) so just increment a variable

            for ngram in ngrams:

                prob_ngram_exists = 0

                if ngram in self.trained_ngram_sentiment_dictionary:
                    # P(B) = probability that an ngram exists in a tweet
                    prob_ngram_exists = sum(self.trained_ngram_sentiment_dictionary[ngram]) / total_ngrams

                    # P(B|A) = probability that an ngram is in a tweet given the tweet is <sentiment>
                    prob_ngram_given_positive = self.trained_ngram_sentiment_dictionary[ngram][0] / self.positive_count
                    prob_ngram_given_neutral = self.trained_ngram_sentiment_dictionary[ngram][1] / self.neutral_count
                    prob_ngram_given_irrelevant = self.trained_ngram_sentiment_dictionary[ngram][2] / self.irrelevant_count
                    prob_ngram_given_negative = self.trained_ngram_sentiment_dictionary[ngram][3] / self.negative_count
                else:
                    # we don't know enough about this ngram
                    pass

                if prob_ngram_exists != 0:
                    tweet_ngrams += 1  # otherwise the ngram didn't contribute to any probabilities
                    # P(A|B) = probability that a tweet is <sentiment> given a specific ngram is in it
                    cum_pos_prob += (prob_ngram_given_positive * prob_positive) / prob_ngram_exists
                    cum_neutral_prob += (prob_ngram_given_neutral * prob_neutral) / prob_ngram_exists
                    cum_irrelevant_prob += (prob_ngram_given_irrelevant * prob_irrelevant) / prob_ngram_exists
                    cum_negative_prob += (prob_ngram_given_negative * prob_negative) / prob_ngram_exists

            if tweet_ngrams > 0:
                cum_pos_prob /= tweet_ngrams
                cum_neutral_prob /= tweet_ngrams
                cum_irrelevant_prob /= tweet_ngrams
                cum_negative_prob /= tweet_ngrams
                probabilities = [cum_pos_prob, cum_neutral_prob, cum_irrelevant_prob, cum_negative_prob]

                highest_prob = max(probabilities)
                if highest_prob == cum_irrelevant_prob:
                    sentiment = 'irrelevant'
                elif highest_prob == cum_neutral_prob:
                    sentiment = 'neutral'
                elif cum_pos_prob == cum_negative_prob:
                    sentiment = 'indeterminate'
                elif highest_prob == cum_pos_prob:
                    sentiment = 'positive'
                elif highest_prob == cum_negative_prob:
                    sentiment = 'negative'

                t.prediction = Prediction(sentiment, probabilities)
            else:
                t.prediction = Prediction('indeterminate', None)
                print(t.text)


class Prediction(object):
    sentiment = ''
    probabilities = None

    def __init__(self, sentiment, probabilities):
        self.sentiment = sentiment
        self.probabilities = probabilities


def accuracy_score(tweets):
    correct = 0
    incorrect = 0
    for t in tweets:
        if t.sentiment == t.prediction.sentiment:
            correct += 1
        else:
            incorrect += 1
    return correct / (correct + incorrect)


def main():
    parse_serialize_rawdata()

if __name__ == '__main__':
    main()
