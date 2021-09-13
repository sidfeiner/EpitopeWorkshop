import os

import matplotlib.pyplot as plt
import re

from EpitopeWorkshop.common import conf

CONSEC_NON_WORD_PATTERN = re.compile(r"\W+")


def clean_title(s: str) -> str:
    return CONSEC_NON_WORD_PATTERN.sub('_', s)


def plot_training_data(test_accs, test_losses, validations_accs, validation_losses, train_accs, train_losses,
                       title):
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
    file_name = clean_title(title)
    file_path = os.path.join(conf.PLOTS_DIR, f"{file_name}.jpg")
    fig.savefig(file_path, dpi=fig.dpi, bbox_inches='tight')
    return file_path
