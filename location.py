import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


# Import tweets from New York, London and Paris
new_york_tweets = pd.read_json("new_york.json", lines=True)
london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)

# Convert to lists and combine, add labels (0=New York,1=London,2=Paris)
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# Make training and test set
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels)

# Make count vectorizer
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

# Fit train data and predict test data
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predict = classifier.predict(test_counts)

# Evaluate predictions
# Accuracy score:
accuracy = accuracy_score(test_labels, predict)
print(accuracy)

# Confusion matrix:
confusion_mat = confusion_matrix(test_labels, predict, normalize='true')
print(confusion_mat)

# Graph confusion matrix
x_axis_labels = ["New York","London","Paris"] # labels for x-axis
y_axis_labels = ["New York","London","Paris"] # labels for y-axis

plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(confusion_mat, cmap = "Blues", xticklabels=x_axis_labels, yticklabels=y_axis_labels,annot = True,annot_kws={"size": 16})
plt.title('Tweet true and predicted location')
plt.show()
