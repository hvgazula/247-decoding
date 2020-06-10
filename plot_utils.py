import os

import matplotlib.pyplot as plt
import pandas as pd


def figure5(SAVE_DIR, lengths, string):
    bins = [0, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000, 7500, 10000]

    categories = pd.cut(lengths, bins)
    price_binned = pd.value_counts(categories).reindex(categories.categories)

    fig, ax = plt.subplots()
    ax.bar(range(0, len(bins) - 1), price_binned, width=1, align='edge')
    plt.xticks(range(len(bins)), labels=bins)

    for i, v in enumerate(price_binned.values):
        ax.text(i + 0.25, v + 5, str(v), color='blue', fontweight='bold')

    plt.title(f'Distribution of Seq lengths ({string})', fontsize=14)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig(os.path.join(SAVE_DIR, string + '_signal_len_dist.png'))
    plt.close()
    return


def plot_training(history, save_dir, title='', val=True):
    '''Plot train/val loss and accuracy and save figures'''
    plt.plot(history["train_loss"])
    if val:
        plt.plot(history["valid_loss"])
    plt.title('Model loss: %s' % title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_dir + 'loss.png')
    plt.clf()
    plt.plot(history["train_acc"])
    if val:
        plt.plot(history["valid_acc"])
    plt.title('Model accuracy: %s' % title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))

    return
