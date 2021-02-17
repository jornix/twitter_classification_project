import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

all_tweets = pd.read_json("random_tweets.json", lines=True)

# Define viral tweets
median_retweet = all_tweets["retweet_count"].median()
all_tweets["is_viral_median"] = np.where(
    all_tweets["retweet_count"] > median_retweet, 1, 0
)
all_tweets["is_viral_5"] = np.where(all_tweets["retweet_count"] > 5, 1, 0)
all_tweets["is_viral_1000"] = np.where(all_tweets["retweet_count"] > 1000, 1, 0)

# Make new features
all_tweets["tweet_length"] = all_tweets.apply(lambda tweet: len(tweet["text"]), axis=1)
all_tweets["followers_count"] = all_tweets.apply(
    lambda tweet: tweet["user"]["followers_count"], axis=1
)
all_tweets["friends_count"] = all_tweets.apply(
    lambda tweet: tweet["user"]["friends_count"], axis=1
)
all_tweets["hashtags"] = all_tweets.apply(
    lambda tweet: tweet["text"].count("#"), axis=1
)
all_tweets["links"] = all_tweets.apply(
    lambda tweet: tweet["text"].count("http"), axis=1
)
all_tweets["n_words"] = all_tweets.apply(
    lambda tweet: len(tweet["text"].split()), axis=1
)
all_tweets["mean_length_words"] = all_tweets.apply(
    lambda tweet: len(tweet["text"]) / len(tweet["text"].split()), axis=1
)
# Select and normalize data
labels = all_tweets["is_viral_median"]
data = all_tweets[
    [
        "tweet_length",
        "followers_count",
        "friends_count",
        "hashtags",
        "links",
        "n_words",
        "mean_length_words",
    ]
]

scaled_data = scale(data, axis=0)
# print(scaled_data[0])

# Create training and test set
train_data, test_data, train_labels, test_labels = train_test_split(
    scaled_data, labels, random_state=1
)

# Check for best K
scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))

plt.plot(range(1, 200), scores)
plt.show()
