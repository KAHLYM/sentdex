import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuressets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuressets[:1900]
testing_set =  featuressets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set)*100))
# classifier.show_most_informative_features(15)

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set)*100))

# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("GNB_classifier accuracy percent: ", (nltk.classify.accuracy(GNB_classifier, testing_set)*100))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy percent: ", (nltk.classify.accuracy(BNB_classifier, testing_set)*100))

LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print("LR_classifier accuracy percent: ", (nltk.classify.accuracy(LR_classifier, testing_set)*100))

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD_classifier accuracy percent: ", (nltk.classify.accuracy(SGD_classifier, testing_set)*100))

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier, testing_set)*100))

LSVC_classifier = SklearnClassifier(LinearSVC())
LSVC_classifier.train(training_set)
print("LSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LSVC_classifier, testing_set)*100))

NSVC_classifier = SklearnClassifier(NuSVC())
NSVC_classifier.train(training_set)
print("NSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NSVC_classifier, testing_set)*100))
