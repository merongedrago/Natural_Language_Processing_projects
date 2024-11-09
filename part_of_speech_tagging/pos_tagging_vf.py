# import necessary packages and dataset
import nltk
import numpy as np
from viterbi import viterbi  # you might need to specify the float type for the matrices

brown_training = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]


def get_key(my_dict, val):
    # simple function to get my key from a dictionary
    for key, value in my_dict.items():
        if val == value:
            return key


def list_tag(training_data):
    # creates a list of states that exist in the dataset
    tag_list = []
    for sentence in training_data:
        for i in range(0, len(sentence)):
            if sentence[i][1] in tag_list:
                pass
            else:
                tag_list.append(sentence[i][1])
    return tag_list


def state_mapping(tag_list):
    # create a mapping of states to indices
    state_index = 0
    mapping_index = {}
    for tag in tag_list:
        mapping_index[tag] = state_index
        state_index += 1
    return mapping_index


def observation_mapping(training_data):
    # create a mapping of observation/words with an index
    obs_index = 0
    obs_mapping = {}
    for sentence in training_data:
        for word in sentence:
            if word[0] in obs_mapping.keys():
                pass
            else:
                obs_mapping[word[0]] = obs_index
                obs_index += 1
    obs_mapping["UNK"] = obs_index
    return obs_mapping


def create_A(training_data, tag_list, state_index):
    # creating matrix A of states to states
    A_matrix = np.full(
        (len(tag_list), len(tag_list)), 1.0
    )  # create a matrix with 1s for smoothing
    row_total = np.full((len(tag_list), 1), 0)
    for sentence in training_data:
        for i in range(1, len(sentence)):
            A_matrix[state_index[sentence[i - 1][1]], state_index[sentence[i][1]]] += 1
    # summing over row to find the probabilites associated with each transition
    for i in range(0, len(tag_list)):
        row_total[i] = sum(A_matrix[i, :])
    for i in range(0, len(tag_list)):
        for j in range(0, len(tag_list)):
            A_matrix[i, j] = A_matrix[i, j] / row_total[i]
    return A_matrix


def create_B(training_data, tag_list, state_index, obs_index):
    # creating matrix B of observations and states
    B_matrix = np.full(
        (len(tag_list), len(obs_index)), 1.0
    )  # create a matrix with 1s for smoothing
    row_total = np.full((len(tag_list), 1), 0)
    for sentence in training_data:
        for words in sentence:
            B_matrix[state_index[words[1]], obs_index[words[0]]] += 1
    for i in range(0, len(tag_list)):
        row_total[i] = sum(B_matrix[i, :])
    for i in range(0, len(tag_list)):
        for j in range(0, len(obs_index)):
            B_matrix[i, j] = B_matrix[i, j] / row_total[i]

    return B_matrix


def create_pi(training_data, tag_list, state_index):
    # creating pi matrix for intial states
    pi_matrix = np.full((len(tag_list), 1), 0.0)
    for sentence in training_data:
        # print(state_index[sentence[0][1]])
        pi_matrix[state_index[sentence[0][1]]] += 1
    # print(pi_matrix)
    total = sum(pi_matrix[:, :])
    for i in range(0, len(tag_list)):
        pi_matrix[i] = pi_matrix[i] / total
    return pi_matrix


# using my functions and assigning to variables so i can use them as an input for viterbi algorithm
brown_states = list_tag(brown_training)
state_index = state_mapping(brown_states)
obs_index = observation_mapping(brown_training)
A_matrix = create_A(brown_training, brown_states, state_index)
B_matrix = create_B(brown_training, brown_states, state_index, obs_index)
pi_matrix = create_pi(brown_training, brown_states, state_index)
pi_matrix = pi_matrix.flatten()

# import test data and translate the sentences to indexes
test_data = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
observation_list = []
observed_states = []
predicted_raw = []
predicted_states = []
for sentences in test_data:
    for words in sentences:
        if words[0] in obs_index.keys():
            observation_list.append(obs_index[words[0]])
        else:
            observation_list.append(obs_index["UNK"])
        # print(observation_list)
        observed_states.append(words[1])

    answer = viterbi(observation_list, pi_matrix, A_matrix, B_matrix)
    for i in answer[0]:
        predicted_states.append(get_key(state_index, i))
    observation_list = []  # reintiallize the observation list

# print result
print(f"The observed states from the training data are:  {observed_states}")
print("*********************************************************************")
print(f"The predicted states from our viterbi model are: {predicted_states}")
