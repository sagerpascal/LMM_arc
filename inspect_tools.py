from arcdsl.main import get_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys

# Visualisation taken from: https://www.kaggle.com/code/dipamc77/arc-simple-dsl-and-search
def plot_one(input_matrix=None, ax=None, title=''):
    ax = ax or plt.axes()
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)


def plot_task(taskdata, title):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(taskdata[0])
    num_test = len(taskdata[1])

    num_tot = num_train + num_test

    fig, axs = plt.subplots(2, num_tot, figsize=(3 * num_tot, 3 * 2))
    fig.suptitle(title)
    if num_tot == 1:
        axs = axs[..., None]
    for i in range(num_train):
        plot_one(taskdata[0][i]['input'], axs[0, i], 'Train Input')
        plot_one(taskdata[0][i]['output'], axs[1, i], 'Train Output')

    if num_tot == 1:
        axs = axs[..., None]
    for i in range(num_test):
        plot_one(taskdata[1][i]['input'], axs[0, num_train+i], 'Test Input')
        plot_one(taskdata[1][i]['output'], axs[1, num_train+i], 'Test Output')
    plt.tight_layout()
    plt.show()

def main():
    data = get_data(base_path='./data/')
    train_keys = list(data['train'].keys())
    ids = np.arange(len(train_keys))
    np.random.shuffle(ids)

    for id in ids:
        key = train_keys[id]
        plot_task([data['train'][key], data['test'][key]], key)


        inp = input('press c for the next example, any other key to exit')
        if not inp.lower() == 'c':
            sys.exit(0)





if __name__ == '__main__':
    main()