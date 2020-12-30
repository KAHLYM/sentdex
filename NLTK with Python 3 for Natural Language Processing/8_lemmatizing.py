from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("cacti"))
# print(lemmatizer.lemmatize("geese"))
# print(lemmatizer.lemmatize("rocks"))
# print(lemmatizer.lemmatize("python"))

# default pos is noun "n"

print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a"))

print(lemmatizer.lemmatize("best"))
print(lemmatizer.lemmatize("best", pos="a"))

print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run", pos="v"))