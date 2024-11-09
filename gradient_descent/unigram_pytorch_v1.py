"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)

    except ValueError:
        idx = len(vocabulary) - 1

    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 500
    learning_rate = 0.5

    # train model
    loss_dict = {}  # creating a dictionary to store num iteration and loss for the plot
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_dict[_] = loss.item()

    # pulling the outputs of the model and converting to probabilites
    logits = model.s.detach().numpy()
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    # create a unigram model to calcuate the optimal probabilities for comparison
    char_dict = {}
    char_dict["None"] = 0
    for element in tokens:
        if element in char_dict:
            char_dict[element] += 1
        elif element.isalpha() or element == " ":
            char_dict[element] = 1
        else:
            char_dict["None"] += 1
    vocab_total = sum(char_dict.values())

    # calculate the minimum loss using the optimal probabilites calculated above
    minim_loss = 0
    for element in char_dict:
        minim_loss += (
            abs((np.log(char_dict[element] / vocab_total))) * char_dict[element]
        )
        char_dict[element] = char_dict[element] / vocab_total

    # reorder the optimal probabilites dictionary and prepare for plotting
    myKeys = list(char_dict.keys())
    myKeys.sort()
    sorted_dict = {i: char_dict[i] for i in myKeys}
    first_two = {" ": sorted_dict.pop(" "), "None": sorted_dict.pop("None")}
    reordered_dict = {**sorted_dict, **first_two}
    vocabulary_for_plotting = [
        str(char) if char is not None else "None" for char in vocabulary
    ]

    # plotting of loss function over time/iteration with the minimum loss line included
    plt.figure(figsize=(10, 6))
    plt.plot(loss_dict.keys(), loss_dict.values(), color="blue")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss function")
    plt.title("Loss over number of iterations / time")
    plt.axhline(y=minim_loss, color="r", linestyle="-")
    plt.xticks(rotation=90)
    plt.legend(
        [
            "Loss from the trained model",
            "Minimum possible loss based on optimal probabilites",
        ]
    )
    plt.tight_layout()
    plt.show()

    # plotting the predicted probabilities from the learned model and optimal probabilities calculated from our unigram model
    plt.figure(figsize=(10, 6))
    plt.scatter(
        vocabulary_for_plotting,
        probs.flatten(),
        color="blue",
        marker="x",
    )
    plt.scatter(
        reordered_dict.keys(),
        reordered_dict.values(),
        color="red",
        alpha=0.3,
        marker="o",
    )
    plt.xlabel("Characters")
    plt.ylabel("Probability")
    plt.title("Optimal probabilities vs. probabilities from a trained model")
    plt.legend(
        [
            "Probability from the trained model",
            "Optimal probability from calculated unigram model",
        ]
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gradient_descent_example()
