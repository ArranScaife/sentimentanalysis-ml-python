# Load and prepare the dataset
import nltk
from nltk.corpus import movie_reviews
import random
# Create a tuple containing a category in the 1st index and the corresponding words 
# associated with the category in the 0th index.
# the two categories are neg and pos
documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Obtain the 2000 most common words in movie reviews
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

# Define the feature extractor
def document_features(document):
    document_words = set(document)
    features = {}
    # Build a dictionary mapping 'contains(<word>)' strings to a T/F bool
    # indicating if <word> is one of the 2000 most common movie review words
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


featuresets = [(document_features(d), c) for (d,c) in documents]
# Split into train and test sets on item 100
train_set, test_set = featuresets[100:], featuresets[:100]
# Train Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))

# Display most important features for classifying sentiment
classifier.show_most_informative_features(5)
