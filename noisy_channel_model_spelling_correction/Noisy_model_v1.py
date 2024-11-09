# import package and datasets
import numpy as np

# loading datasets
word_count = np.loadtxt("https://norvig.com/ngrams/count_1w.txt", dtype=str)
addition_csv = np.genfromtxt("additions.csv", delimiter=",", dtype=str)
deletions_csv = np.genfromtxt("deletions.csv", delimiter=",", dtype=None, comments=None)
substitutions_csv = np.genfromtxt("substitutions.csv", delimiter=",", dtype=str)
bigrams_csv = np.genfromtxt("bigrams.csv", delimiter=",", dtype=str, comments=None)
unigrams_csv = np.genfromtxt("unigrams.csv", delimiter=",", dtype=str)
alphabet_list = list(unigrams_csv[1:, 0])


def wrong_insert(corrputed_word, poss_word_dict):
    """creates a list of words where a letter is deleted from a word and the erorr probability of that edit"""
    for i in range(
        1, len(corrputed_word)
    ):  # assuming that the first character of the word is correct(please see assumptions pdf)
        new_word = corrputed_word[:i] + corrputed_word[i + 1 :]
        prefix_array = addition_csv[addition_csv[:, 0] == corrputed_word[i - 1], :]
        mistake_score = prefix_array[prefix_array[:, 1] == corrputed_word[i], 2]
        if len(prefix_array[prefix_array[:, 1] == corrputed_word[i], 2]) == 0:
            pass  # assuming that if can't find the prefix and added letter in the addition csv, we don't consider the word
        else:
            final_score = int(mistake_score) / int(
                unigrams_csv[unigrams_csv[:, 0] == corrputed_word[i - 1], 1]
            )
            poss_word_dict[new_word] = final_score
        new_word = ""  # reintialize the word that was created
    return poss_word_dict


def wrong_delete(corrputed_word, poss_word_dict):
    """creates a list of words where a letter from the alphabet is added to the wrong word and the erorr probability of that edit"""
    for i in range(1, len(corrputed_word) + 1):
        for alphabet in alphabet_list:
            # assuming that the first character of the word is correct(please see assumptions pdf)
            new_word = corrputed_word[:i] + alphabet + corrputed_word[i:]
            char_bigram = corrputed_word[i - 1] + alphabet
            prefix_array = deletions_csv[
                deletions_csv[:, 0] == corrputed_word[i - 1], :
            ]

            if len(prefix_array[prefix_array[:, 1] == alphabet, 2]) == 0:
                pass  # assuming that if can't find the of the prefix and the alphabet in the deletions csv, we don't consider the word
            else:
                mistake_score = prefix_array[prefix_array[:, 1] == alphabet, 2]
                final_score = int(mistake_score) / int(
                    bigrams_csv[bigrams_csv[:, 0] == char_bigram, 1]
                )
                poss_word_dict[new_word] = final_score
            new_word = ""
            char_bigram = ""
    return poss_word_dict


def wrong_subst(corrputed_word, poss_word_dict):
    """creates a list of words where a letter from the alphabet is substituted by a letter in wrong word and the erorr probability of that edit"""
    for i in range(0, len(corrputed_word)):
        for alphabet in alphabet_list:
            if alphabet == corrputed_word[i]:
                pass
            elif i == 0:  # case when it is the first letter that has been substituted
                new_word = alphabet + corrputed_word[i + 1 :]
                prefix_array = substitutions_csv[substitutions_csv[:, 0] == alphabet, :]

                if len(prefix_array[prefix_array[:, 1] == corrputed_word[i], 2]) == 0:
                    pass  # assumption that if cant find that combination of substituion in the csv, then we pass
                else:
                    mistake_score = prefix_array[
                        prefix_array[:, 1] == corrputed_word[i], 2
                    ]
                    final_score = int(mistake_score) / int(
                        unigrams_csv[unigrams_csv[:, 0] == alphabet, 1]
                    )
                    poss_word_dict[new_word] = final_score

            else:
                new_word = corrputed_word[:i] + alphabet + corrputed_word[i + 1 :]
                prefix_array = substitutions_csv[substitutions_csv[:, 0] == alphabet, :]
                if len(prefix_array[prefix_array[:, 1] == corrputed_word[i], 2]) == 0:
                    pass  # assumption that if cant find that combination of substituion in the csv, then we pass
                else:
                    mistake_score = prefix_array[
                        prefix_array[:, 1] == corrputed_word[i], 2
                    ]
                    final_score = int(mistake_score) / int(
                        unigrams_csv[unigrams_csv[:, 0] == alphabet, 1]
                    )
                    poss_word_dict[new_word] = final_score

            new_word = ""
    return poss_word_dict


def get_key(my_dict, val):
    """gets a key of a dictionary when given a value"""
    for key, value in my_dict.items():
        if val == value:
            return key


def check_exist(possible_dictionary, real_words):
    """check if the word generated from the deletion, addition and susbstituion are in the dictionary provided"""
    poss_real_word = {}
    for keys in possible_dictionary.keys():
        if keys in real_words[:, 0]:
            poss_real_word[keys] = possible_dictionary[keys] * int(
                word_count[word_count[:, 0] == keys, 1]
            )
    return poss_real_word


def most_likely(possible_real_words):
    """finds the most likely word from a dictionary of words and their associated error probability"""
    if len(possible_real_words) == 0:
        return "No possible word found"
    else:
        max_likelihood = max(possible_real_words.values())
    return get_key(possible_real_words, max_likelihood)


# final function to call to correct word
def correct_word(corrupted_word):
    """corrects a word when given a wrong word"""
    corrupted_word = corrupted_word.lower()
    if corrupted_word in word_count[:, 0]:
        return "this word already exists "
    word_dict = {}
    word_dict1 = wrong_insert(corrupted_word, word_dict)
    word_dict2 = wrong_delete(corrupted_word, word_dict1)
    word_dict2 = wrong_subst(corrupted_word, word_dict2)
    real_word = check_exist(word_dict2, word_count)
    corrected_word = most_likely(real_word)
    return corrected_word
