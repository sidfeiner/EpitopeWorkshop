import matplotlib.pyplot as plt


def plot_training_data(test_accs, test_losses, validations_accs, validation_losses, train_accs, train_losses,
                       title=None):
    fig = plt.figure()
    plt.plot(train_accs)
    plt.plot(train_losses)
    plt.plot(validations_accs)
    plt.plot(validation_losses)
    plt.plot(test_accs)
    plt.plot(test_losses)
    plt.xlabel('# Batch')
    plt.legend(['Train accuracy', 'Train Loss', 'Validation accuracy', 'Validation Loss', 'Test Accuracy', 'Test Loss'],
               fontsize=13)
    if title is not None:
        plt.title(title)
    fig.tight_layout()
    plt.show()
