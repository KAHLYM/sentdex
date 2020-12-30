from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

# %appdata%\\nltk_data\corpora

# import nltk
# nltk.download("gutenberg")

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

print(tok[5:15])
