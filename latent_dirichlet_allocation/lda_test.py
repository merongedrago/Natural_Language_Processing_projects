"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""

from typing import List
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np


def lda_gen(
    vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int
) -> List[str]:
    """Generate words based on LDA inputs alpha, beta, xi and vocabulary"""
    # create a list of topics based on alpha
    document_size = np.random.poisson(xi)
    alpha_norm = alpha / sum(alpha)
    topic_list = list(range(len(alpha)))
    chosen_topics = np.random.choice(
        topic_list, p=alpha_norm, size=document_size, replace=True
    )

    # create a dictionary with key as the topic and values as a list of topic
    word_topic_dict = {}
    for topic in topic_list:
        words_per_topic = np.random.choice(
            vocabulary, p=beta[topic], size=document_size, replace=True
        )
        word_topic_dict[topic] = words_per_topic
        words_per_topic = []

    # choose a word for each topic in the chosen topic list
    words = []
    for element in chosen_topics:
        words.append(np.random.choice(word_topic_dict[element]))
    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass",
        "pike",
        "deep",
        "tuba",
        "horn",
        "catapult",
    ]
    beta = np.array(
        [
            [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
            [0.3, 0.0, 0.2, 0.3, 0.2, 0.0],
        ]
    )

    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [lda_gen(vocabulary, alpha, beta, xi) for _ in range(100)]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(corpus, id2word=dictionary, num_topics=3)
    print(model.alpha)
    print(model.show_topics())


if __name__ == "__main__":
    test()
