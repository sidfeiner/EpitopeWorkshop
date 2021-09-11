import matplotlib.pyplot as plt


def plot_training_data(test_accs, test_losses, train_accs, train_losses, title=None):
    fig = plt.figure()
    plt.plot(train_accs)
    plt.plot(train_losses)
    plt.plot(test_accs)
    plt.plot(test_losses)
    plt.xlabel('# Batch')
    plt.legend(['Train accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss'], fontsize=13)
    if title is not None:
        plt.title(title)
    fig.tight_layout()
    plt.show()
