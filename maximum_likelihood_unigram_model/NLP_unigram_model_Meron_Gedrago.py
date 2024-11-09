from matplotlib import pyplot as plt


def apples_unigram():
    unigram_accuracy = {}
    for apples in range(0, 101):
        # simulating with 100 elements instead of 10 to create a general case a smoother
        prob_apple = apples / 100
        prob_banana = 1 - prob_apple
        # calculating the accuracy of the model by muliplying the probability of each element in the list
        inst_accuracy = (prob_apple**6) * (prob_banana**4)
        unigram_accuracy[prob_apple] = inst_accuracy
    return unigram_accuracy


# storing the dictionary of the probability and the accuracy of the model at each probability to app_dict
app_dict = apples_unigram()

# pulling and storing the x and y values from the dictionary
x_plot = app_dict.keys()
y_plot = app_dict.values()

# plot and show the plot of probability and the accuracy of the model
plt.plot(x_plot, y_plot)
plt.xlabel("Probability of apple")
plt.ylabel("Accuracy of the model")
plt.title("Plot of probability of apple vs accuracy")
plt.grid()
plt.show()
