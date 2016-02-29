# Modified Niek Sanders-Twitter Sentiment Corpus Install Script (http://www.sananalytics.com/lab/twitter-sentiment/)
# Pull tweets listed by ID in corpus.csv and writes to full-corpus.csv because Twitter ToS prevents direct distribution

from __future__ import unicode_literals
import json
import csv
import os
import time

from twitter_api_keys import CONSUMER_KEY
from twitter_api_keys import CONSUMER_SECRET
from twitter_api_keys import ACCESS_TOKEN_KEY
from twitter_api_keys import ACCESS_TOKEN_SECRET
from TwitterAPI import TwitterAPI

api = TwitterAPI(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)


def read_total_list(in_filename):
    with open(in_filename, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')

        total_list = []
        for row in reader:
            total_list.append(row)

        return total_list


def purge_already_fetched(fetch_list, raw_dir):
    rem_list = []

    for item in fetch_list:
        tweet_file = raw_dir + item[2] + '.json'
        if os.path.exists(tweet_file):
            try:
                parse_tweet_json(tweet_file)
                print('--> already downloaded #' + item[2])
            except RuntimeError:
                rem_list.append(item)
        else:
            rem_list.append(item)

    return rem_list


def get_time_left_str(cur_idx, fetch_list, download_pause):
    tweets_left = len(fetch_list) - cur_idx
    total_seconds = tweets_left * download_pause

    str_hr = int(total_seconds / 3600)
    str_min = int((total_seconds - str_hr*3600) / 60)
    str_sec = total_seconds - str_hr*3600 - str_min*60

    return '%dh %dm %ds' % (str_hr, str_min, str_sec)


def download_tweets(fetch_list, raw_dir):
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    max_tweets_per_hr = 710  # technically 720, 710 to be safe
    download_pause_sec = 3600 / max_tweets_per_hr

    for idx in range(0, len(fetch_list)):
        item = fetch_list[idx]

        time_remaining = get_time_left_str( idx, fetch_list, download_pause_sec)
        print('--> downloading tweet #%s (%d of %d) (%s left)' % (item[2], idx+1, len(fetch_list), time_remaining))

        r = api.request('statuses/show/:%d' % int(item[2]))

        if 'errors' not in r.json():
            with open(raw_dir + item[2] + '.json', 'w') as outfile:
                json.dump(r.json(), outfile)
        else:
            print('Tweet contains error, not saving')

        print('    pausing %d sec to obey Twitter API rate limits' % download_pause_sec)
        time.sleep(download_pause_sec)


def parse_tweet_json(filename):
    print('opening: ' + filename)

    with open(filename, 'r') as fp:
        try:
            tweet_json = json.load(fp)
        except ValueError:
            raise RuntimeError('error parsing json')

        return [tweet_json['created_at'], tweet_json['text']]


def build_output_corpus(out_filename, raw_dir, total_list):
    with open(out_filename, 'w') as fp:
        writer = csv.writer(fp, delimiter=',', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL)
        writer.writerow(['Topic', 'Sentiment', 'TweetId', 'TweetDate', 'TweetText'])

        missing_count = 0
        for item in total_list:
            if os.path.exists(raw_dir + item[2] + '.json'):
                try:
                    parsed_tweet = parse_tweet_json(raw_dir + item[2] + '.json')
                    full_row = item + parsed_tweet

                    for i in range(0, len(full_row)):
                        full_row[i] = full_row[i].encode("utf-8")

                    writer.writerow(full_row)

                except RuntimeError:
                    print('--> bad data in tweet #' + item[2])
                    missing_count += 1
            else:
                print('--> missing tweet #' + item[2])
                missing_count += 1

        print('\n%d of %d tweets downloaded!' % (len(total_list) - missing_count, len(total_list)))
        print('Output in: ' + out_filename + '\n')


def main():
    total_list = read_total_list('./corpus.csv')
    fetch_list = purge_already_fetched(total_list, './rawdata/')

    download_tweets(fetch_list, './rawdata/')

    # second pass for any failed downloads
    # print('\nStarting second pass to retry any failed downloads')
    # fetch_list = purge_already_fetched(total_list, './rawdata/')
    # download_tweets(fetch_list, './rawdata/')

    # Missing 998 of 5513 tweets, either because we didn't have permission to view it or the tweet was deleted
    build_output_corpus('./full-corpus.csv', './rawdata/', total_list)


if __name__ == '__main__':
    main()
