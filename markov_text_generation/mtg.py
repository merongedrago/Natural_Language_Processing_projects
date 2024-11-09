import nltk
import numpy as np

# dataset
alice = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()
corpus_austen = nltk.word_tokenize(alice)


# function to get my key when i have a value
def get_key(my_dict, val):
    keys = []
    for key, value in my_dict.items():
        if val == value:
            keys.append(key)
            if len(keys) == 1:
                final_key = key
            elif len(keys) > 1:
                final_key = keys
    return final_key


# Create an n-gram index for fast lookup of n-gram counts
def create_ngram_index(corpus, n_max):
    ngram_index = {}
    for n in range(1, n_max + 1):
        for i in range(len(corpus) - n + 1):
            ngram = tuple(corpus[i : i + n])
            if ngram in ngram_index:
                ngram_index[ngram] += 1
            else:
                ngram_index[ngram] = 1
    return ngram_index


# Count occurrences of an n-gram using the index
def count_occ(sent_count, ngram_index):
    sent_tuple = tuple(sent_count)
    return ngram_index.get(sent_tuple, 0)


# Calculate probability using multiple n-grams with backoff
def calculate_score(sentence, n, word, corpus, ngram_index):
    word_score = 0
    backoff_num = 0

    for i in range(n, 0, -1):
        seed_sentence = sentence[-i:]
        new_sentence = seed_sentence + [word]

        # Get counts from the n-gram index
        count_new = count_occ(new_sentence, ngram_index)
        count_seed = count_occ(seed_sentence, ngram_index)

        if count_new > 0 and count_seed > 0:

            word_score = (0.4**backoff_num) * (count_new / count_seed)
            return word_score

        backoff_num += 1

    word_score = (0.4**backoff_num) * (corpus.count(word) / len(corpus))
    return word_score


# Get most common words since it was taking my computer long to computer - you can adjust this number if computer is also too slow
def get_most_common_words(corpus, top_n=800):
    word_freq = {}
    for word in corpus:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]


def predict_dict(sentence, n, corpus, ngram_index, top_n_words=800):
    common_words = get_most_common_words(corpus, top_n_words)
    score_dict = {}

    for word in common_words:

        score_dict[word] = calculate_score(sentence, n, word, corpus, ngram_index)
    return max(score_dict, key=score_dict.get)


def predict_dict_random(sentence, n, corpus, ngram_index, top_n_words=700):
    common_words = get_most_common_words(corpus, top_n_words)
    score_dict = {}
    my_list = []

    for word in common_words:

        score_dict[word] = calculate_score(sentence, n, word, corpus, ngram_index)
        my_list.append(word)

    return my_list


# Function to construct the sentence when deterministstic
def construct_sentence(sentence, n, corpus, ngram_index):
    while len(sentence) < 10:
        word = predict_dict(sentence, n, corpus, ngram_index)
        sentence.append(word)
        if word in [".", "?", "!"]:
            break
    return sentence


# Function to predict based on the random input
def finish_sentence(sentence, n, corpus, random):
    if random == True:
        ngram_index = create_ngram_index(corpus_austen, n)
        while len(sentence) < 10:
            word = np.random.choice(
                predict_dict_random(sentence, n, corpus, ngram_index, top_n_words=700)
            )
            sentence.append(str(word))
            if word in [".", "?", "!"]:
                pass
    else:
        ngram_index = create_ngram_index(corpus_austen, n)
        sentence = construct_sentence(test_sent_token, n, corpus_austen, ngram_index)

    return sentence


# print(np.unique(new_test_corpus))
# Example usage
n = 3  # Set n-gram level
ngram_index = create_ngram_index(corpus_austen, n)
test_sentence = "she was not".split()
new_sentence = construct_sentence(test_sentence, n, corpus_austen, ngram_index)

# print("Generated Sentence:", " ".join(new_sentence))


test_sentence = "it did not"
test_sent_token = nltk.word_tokenize(test_sentence)
test_corpus = "She was not to, she was not to decided not eating go to sleep"
test_token = nltk.word_tokenize(test_corpus.lower())


print(finish_sentence(test_sent_token, 5, corpus_austen, False))
