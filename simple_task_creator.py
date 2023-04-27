# Create an infinite amount of samples of task 08ed6ac7
# i.e. four vertical bars in a 9x9 grid, color bar according to their size (yellow, green, red, blue -> from small to big)

import random
import warnings
from pathlib import Path

import matplotlib
import skimage.measure
from skimage.draw import random_shapes
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=50, help='Number of samples to generate')
args = parser.parse_args()


def create_single_task_08ed6ac7():
    def get_colors(bar_sizes):
        colors = np.zeros((4), dtype=int)
        bar_order = np.argsort(bar_sizes)[::-1]
        for i, size in enumerate(bar_order):
            colors[size] = i + 1
        return colors

    def color_bars(data, bar_sizes, bar_colors):
        for i, size in enumerate(bar_sizes):
            data[9 - size: 9, 2 * i + 1: 2 * i + 2] = bar_colors[i]
        return data

    bar_sizes = random.sample(range(1, 9), 4)
    bar_colors = get_colors(bar_sizes)

    # Create the input data
    input_data = np.zeros((9, 9), dtype=int)
    input_data = color_bars(input_data, bar_sizes, [5] * 4)

    # Create the output data
    output_data = np.zeros((9, 9), dtype=int)
    output_data = color_bars(output_data, bar_sizes, bar_colors)

    task_dict = {
        "input": input_data.tolist(),
        "output": output_data.tolist()
    }

    return task_dict


def create_single_task_477d2879(image=None):
    def remove_single_pixels(data):
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]):
                if ((data[max(i - 1, 0), j] != data[i, j] or i == 0) and (
                        data[min(i + 1, data.shape[0] - 1), j] != data[i, j] or i == data.shape[0] - 1)) or (
                        (data[i, max(j - 1, 0)] != data[i, j] or j == 0) and (
                        data[i, min(j + 1, data.shape[1] - 1)] != data[i, j] or j == data.shape[1] - 1)):
                    data[i, j] = 0
        return data


    def remove_small_components(data, min_size=5):
        for v in np.unique(data):
            if v != 0 and np.sum(data == v) < min_size:
                data[data == v] = 0
        return data


    def mark_border(data):
        borders = np.zeros(data.shape, dtype=int)
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]):
                if data[i, j] != 0:
                    if data[max(i - 1, 0), j] != data[i, j] or data[min(i + 1, data.shape[0] - 1), j] != data[i, j] or \
                            data[i, max(j - 1, 0)] != data[i, j] or data[i, min(j + 1, data.shape[0] - 1)] != data[
                        i, j] or data[max(i - 1, 0), max(j - 1, 0)] != data[i, j] or data[
                        max(i - 1, 0), min(j + 1, data.shape[0] - 1)] != data[i, j] or data[
                        min(i + 1, data.shape[0] - 1), max(j - 1, 0)] != data[i, j] or data[
                        min(i + 1, data.shape[0] - 1), min(j + 1, data.shape[0] - 1)] != data[i, j]:
                        borders[i, j] = 1
        return borders

    def color_image(connected_components_no_border, border_components):
        values_no_border = np.unique(connected_components_no_border)[1:]
        values_border = np.unique(border_components)[1:]
        colors = random.sample(range(2, 20), len(values_no_border) + len(values_border))
        result = np.zeros(connected_components_no_border.shape, dtype=int)

        # color fields
        for i in range(len(values_no_border)):
            result[connected_components_no_border == values_no_border[i]] = colors[i]

        # color borders
        for i in range(len(values_border)):
            result[border_components == values_border[i]] = colors[i + len(values_no_border)]

        return result

    def get_input(output, border):
        input = np.zeros(output.shape, dtype=int)
        input[border == 1] = 1

        for v in np.unique(output):
            index_v = np.where(output == v)
            if len(index_v[0]) > 1:
                i = random.randint(0, len(index_v[0]) - 1)
                i1, i2 = index_v[0][i], index_v[1][i]
            else:
                i1, i2 = index_v[0], index_v[1]
            input[i1, i2] = output[i1, i2]

        return input

    def check_if_ok(borders):
        color_bg = 1
        for i in range(1, borders.shape[0] - 1):
            for j in range(1, borders.shape[1] - 1):
                if (borders[i - 1, j] == color_bg or i - 1 == 0) and (
                        borders[i + 1, j] == color_bg or i + 1 == borders.shape[0]) and (
                        borders[i, j - 1] == color_bg or j - 1 == 0) and (
                        borders[i, j + 1] == color_bg or j + 1 == borders.shape[1]):
                    return False
        return True

    if image is None:
        # lets make the image a big bigger and crop it afterwards so that some shapes are touching the border
        image, _ = random_shapes((15, 15), min_shapes=2, max_shapes=6,
                                 min_size=4, max_size=9, allow_overlap=True, num_channels=1)
        image = image[:13, :13, 0]
        image[image == 255] = 0.
        image[image > 0] = 1.

    # generate initial image with shapes and borders
    connected_components = remove_single_pixels(image)
    connected_components = remove_small_components(connected_components)
    connected_components = skimage.measure.label(connected_components, connectivity=2)
    borders = mark_border(connected_components)

    # extract connected components of shapes and borders
    connected_components_no_border = skimage.measure.label(1 - borders, connectivity=1)
    border_components = skimage.measure.label(borders, connectivity=1)

    # generate input and output
    result = color_image(connected_components_no_border, border_components)
    input_data = get_input(result, borders)

    if not check_if_ok(borders):
        return None

    # fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    # ax[0].imshow(input_data, cmap=matplotlib.colormaps["tab20"], vmin=0, vmax=20, interpolation='none')
    # ax[1].imshow(result, cmap=matplotlib.colormaps["tab20"], vmin=0, vmax=20, interpolation='none')
    # for i in range(2):
    #     ax[i].set_xticks(np.arange(-0.5, 12, 1), minor=True)
    #     ax[i].set_yticks(np.arange(-0.5, 12, 1), minor=True)
    #     ax[i].set_xticks(np.arange(0, 13, 1))
    #     ax[i].set_yticks(np.arange(0, 13, 1))
    #     ax[i].grid(which="minor")
    #     ax[i].set_xticklabels([])
    #     ax[i].set_yticklabels([])
    # plt.tight_layout()
    # plt.show()
    # pass
    # again = input("Again? (y/n)")
    # if again == "y":
    #     create_single_task_477d2879(image)

    task_dict = {
        "input": input_data.tolist(),
        "output": result.tolist()
    }
    return task_dict



def create_task(task_f, n_samples, path):
    if path.exists():
        answer = input(f"File {path} already exists. Overwrite? (y/n)")
        if answer == "y":
            path.unlink()
        else:
            warnings.warn(f"File {path} already exists. Not overwritten.")
            return

    tasks = []
    while len(tasks) < n_samples:
        task_dict = task_f()
        if task_dict is not None:
            tasks.append(task_dict)

    data = {"train": tasks}

    with open(path, 'w') as fp:
        json.dump(data, fp)



def create_task_08ed6ac7(n_samples, path=Path('08ed6ac7_v2.json')):
    create_task(create_single_task_08ed6ac7, n_samples, path)


def create_task_477d2879(n_samples, path=Path('477d2879_v2.json')):
    create_task(create_single_task_477d2879, n_samples, path)


if __name__ == '__main__':
    create_task_08ed6ac7(n_samples=args.n_samples)
    create_task_477d2879(n_samples=args.n_samples)

