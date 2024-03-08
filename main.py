from pprint import pprint
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle
from statistics import mean

nltk.download([
     "movie_reviews",
     "vader_lexicon",
     "punkt",
 ])

sia = SentimentIntensityAnalyzer()

def is_positive(review_text: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(review_text)
    ]
    return mean(scores) > 0

positive_reviews = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_reviews = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_reviews = positive_reviews + negative_reviews

shuffle(all_reviews)
correct = 0
for review_id in all_reviews:
    review_text = nltk.corpus.movie_reviews.raw(review_id)
    if is_positive(review_text):
        if review_id in positive_reviews:
            correct += 1
    else:
        if review_id in negative_reviews:
            correct += 1


unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]
negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]


positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}


unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
    if w.isalpha() and w not in unwanted
])
negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
    if w.isalpha() and w not in unwanted
])


print(F"{correct / len(all_reviews):.2%} correct")
