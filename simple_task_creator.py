# Create an infinite amount of samples of task 08ed6ac7
# i.e. four vertical bars in a 9x9 grid, color bar according to their size (yellow, green, red, blue -> from small to big)

import random
from pathlib import Path
from skimage.draw import random_shapes
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to generate')
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


def create_single_task_477d2879():
    # lets make the image a big bigger and crop it afterwards so that some shapes are touching the border
    image, _ = random_shapes((18, 18), min_shapes=2, max_shapes=5,
                             min_size=4, allow_overlap=True)
    image = image[:13, :13]
    plt.imshow(image)
    plt.show()

def create_single_task_5daaa586():
    print("asdf")
    # 12 to 25 grid with and height
    h, w = np.random.randint(12, 25, 2)
    grid = np.zeros((h,w))

    # 4 colors
    cols = np.random.randint(1, 9, 4)
    # choose sprinkle color
    sp_col = np.random.choice(cols)
    # add sprinkles
    # determine P(flip)
    p_flip = 1/np.random.randint(3, 10)
    # flip corresponding cells
    grid[np.random.uniform(0,1,(h,w)) < p_flip] = sp_col

    offsets = {}
    # size horizontal seg
    size_hor = np.random.randint(3, w-4)
    # left offset
    offsets["left_off"] = int((w-size_hor)/2)
    offsets["right_off"] = offsets["left_off"]+size_hor
    # size vert seg
    size_ver = np.random.randint(3, h - 4)
    # top offset
    offsets["top_off"]
    top_off = int((h - size_ver) / 2)
    bot_off = top_off+size_ver

    # randomly select color and line -> draw





    input_data = None
    output_data = None
    task_dict = {
        "input": input_data.tolist(),
        "output": output_data.tolist()
    }
    return task_dict


def create_task_samples(task_fn, n_samples, path):
    assert not path.exists()

    tasks = []
    for _ in range(n_samples):
        task_dict = task_fn()
        tasks.append(task_dict)
    data = {"train": tasks}

    with open(path, 'w') as fp:
        json.dump(data, fp)


# def create_task_08ed6ac7(n_samples, path=Path('08ed6ac7_v2.json')):
#     assert not path.exists()
#
#     tasks = []
#     for _ in range(n_samples):
#         task_dict = create_single_task_08ed6ac7()
#         tasks.append(task_dict)
#     data = {"train": tasks}
#
#     with open(path, 'w') as fp:
#         json.dump(data, fp)
#
#
# def create_task_477d2879(n_samples, path=Path('477d2879_v2.json')):
#     assert not path.exists()
#
#     tasks = []
#     for _ in range(n_samples):
#         task_dict = create_single_task_477d2879()
#         tasks.append(task_dict)
#     data = {"train": tasks}
#
#     with open(path, 'w') as fp:
#         json.dump(data, fp)


if __name__ == '__main__':
    # create_task_08ed6ac7(n_samples=args.n_samples)
    # create_task_477d2879(n_samples=args.n_samples)

    #create_task_samples(create_single_task_477d2879, n_samples=args.n_samples, path=Path('477d2879_v2.json'))
    create_task_samples(create_single_task_5daaa586, n_samples=args.n_samples, path=Path('5daaa586.json'))