import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk 
import random
from nltk.corpus import movie_reviews
import pickle

# SklearnClassifier is an API to incorporate sklearn with nltk
from nltk.classify.scikitlearn import SklearnClassifier as skc
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers 

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf


documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

# Below lines means the same as above lines, we are creating documents to test and train data
# documents = []
# for category in movie_reviews.categories():
	# for fileid in movie_reviews.fileids(category):
		# documents.append(list(movie_reviews.words(fileids)), category)

random.shuffle(documents)

# print(documents[1])

all_words = []

# We will add all words to a single list, odd out the important most repeating words,
# then use those words to predict whether review is pos or neg. Hence, words are features of our text-classifier.

for w in movie_reviews.words():
	w = w.lower()
	all_words.append(w)

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(20))
# We see that most_common words(20) are just useless. 
# print(all_words["stupid"])

# We are using top 3000 common words to use as features
word_features = list(all_words.keys())[:3000]

# this funcion will compare each word of a review with words in word_features and return them if true
def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
# below line will process all documents and extract their features into individual tuples of the list
featuresets = [(find_features(rev), category) for (rev, category) in documents]

train_set = featuresets[:1900]
test_set = featuresets[1900:]

# using Naive Bayes algo
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("NaiveBayesClassifier algo: ",nltk.classify.accuracy(classifier, test_set)*100)
classifier.show_most_informative_features(15)

# saving classifier using pickle
# pickle_out = open("NaiveBayesClassifier.pickle", "wb")
# pickle.dump(classifier, pickle_in)
# pickle_out.close()

MNB_Classifier = skc(MultinomialNB())
MNB_Classifier.train(train_set)
print("MNB_Classifier algo: ",nltk.classify.accuracy(MNB_Classifier, test_set)*100)

# GNB_Classifier = skc(GaussianNB())
# GNB_Classifier.train(train_set)
# print("GNB_Classifier algo: ",nltk.classify.accuracy(GNB_Classifier, test_set)*100)

BNB_Classifier = skc(BernoulliNB())
BNB_Classifier.train(train_set)
print("BNB_Classifier algo: ",nltk.classify.accuracy(BNB_Classifier, test_set)*100)

LogisticRegression_Classifier = skc(LogisticRegression())
LogisticRegression_Classifier.train(train_set)
print("LogisticRegression_Classifier algo: ",nltk.classify.accuracy(LogisticRegression_Classifier, test_set)*100)

SGDClassifier_Classifier = skc(SGDClassifier())
SGDClassifier_Classifier.train(train_set)
print("SGDClassifier_Classifier algo: ",nltk.classify.accuracy(SGDClassifier_Classifier, test_set)*100)

SVC_classifier = skc(SVC())
SVC_classifier.train(train_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)

LinearSVC_classifier = skc(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

NuSVC_classifier = skc(NuSVC())
NuSVC_classifier.train(train_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100)

# combining algos with a vote, creating our own classifier
voted_classifier = VoteClassifier(classifier, BNB_Classifier, MNB_Classifier, 
								LogisticRegression_Classifier, SGDClassifier_Classifier, 
								LinearSVC_classifier, NuSVC_classifier)

print("voted_classifier accuracy percent: ",nltk.classify.accuracy(voted_classifier, test_set)*100)

print("Classification:", voted_classifier.classify(test_set[0][0]), "confidence %=", voted_classifier.confidence(test_set[0][0])*100)
print("Classification:", voted_classifier.classify(test_set[1][0]), "confidence %=", voted_classifier.confidence(test_set[1][0])*100)
print("Classification:", voted_classifier.classify(test_set[2][0]), "confidence %=", voted_classifier.confidence(test_set[2][0])*100)
print("Classification:", voted_classifier.classify(test_set[3][0]), "confidence %=", voted_classifier.confidence(test_set[3][0])*100)
print("Classification:", voted_classifier.classify(test_set[4][0]), "confidence %=", voted_classifier.confidence(test_set[4][0])*100)
print("Classification:", voted_classifier.classify(test_set[5][0]), "confidence %=", voted_classifier.confidence(test_set[5][0])*100)

