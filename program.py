import pickle
from utilities import Tweet  # necessary for loading tweet objects
from utilities import preprocess
from utilities import BayesSentimentClassifier
from utilities import accuracy_score
from sklearn.cross_validation import KFold


def main():
    # load the tweet objects
    tweets = pickle.load(open('tweets.p', 'rb'))

    # preprocess each tweet's text
    preprocess(tweets)

    # perform KFold cross-validation with 3 folds to get more accurate accuracy prediction
    kf = KFold(n=len(tweets), n_folds=3, shuffle=True)
    for train_indices, test_indices in kf:
        tweets_train = [tweets[i] for i in train_indices]
        tweets_test = [tweets[i] for i in test_indices]

        clf = BayesSentimentClassifier()
        clf.train(tweets_train)  # train the classifier, i.e. populate the sentiment dictionary
        clf.predict(tweets_test)  # predict sentiment of each tweet using Bayes Theorem

        # calculate accuracy
        print(accuracy_score(tweets_test))

        test = Tweet("", "", "I am skeptical about this result", "Tue Oct 18 18:05:50 +0000 2011", "")
        clf.predict([test])
        print(test.prediction.sentiment)
        print(test.prediction.probabilities)

        # TODO: try word boundaries
        # TODO: try tdidf


if __name__ == '__main__':
    main()
