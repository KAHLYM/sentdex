from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK
# nltk.download()

example_text = "Hello there, how are you doing today? The weather is great and Python awesome. The sky is pinkish-blue. You should not eat cardboard."

# Tokenize sentences
print(sent_tokenize(example_text))

# Tokenize words
print(word_tokenize(example_text))
