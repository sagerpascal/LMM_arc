import numpy as np
import pandas as pd
import os, json, cv2, time
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

from matplotlib import colors
from copy import deepcopy

from multiprocessing import Pool as MultiProcessingPool
import multiprocessing as mp
mp.cpu_count()



data_path = '/kaggle/input/abstraction-and-reasoning-challenge/'
os.listdir(data_path)


hexcols = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
           '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
           '#467272']  ## Last color will be used for padding


def hex2numcolor(color):
    h2n = lambda x: int(x, 16)
    r, g, b = h2n(color[1:3]), h2n(color[3:5]), h2n(color[5:7])
    return np.array([r, g, b])


npcols = np.array([hex2numcolor(col) for col in hexcols])


def colorize(img, cols):
    newimg = np.empty((*img.shape, 3), np.uint8)
    for k, v in zip(np.arange(len(cols)), cols): newimg[img == k, :] = v
    return newimg


def imshow_colored(img):
    colimg = colorize(img, npcols)
    plt.imshow(colimg)


def get_tasks(data_path='../data/data/'):
    training_path = data_path + 'training/'
    evaluation_path = data_path + 'evaluation/'
    test_path = data_path + 'test/'

    training_tasks = sorted(os.listdir(training_path))
    evaluation_tasks = sorted(os.listdir(evaluation_path))
    test_tasks = sorted(os.listdir(test_path))

    train_tasks = [Task(tnum, training_path + tfile) for tnum, tfile in enumerate(training_tasks)]
    eval_tasks = [Task(tnum, evaluation_path + tfile) for tnum, tfile in enumerate(evaluation_tasks)]
    test_tasks = [Task(tnum, test_path + tfile) for tnum, tfile in enumerate(test_tasks)]

    return train_tasks, eval_tasks, test_tasks


def get_train_test(task):
    train_in, train_out = [], []
    for t in task['train']:
        train_in.append(np.uint8(np.array(t['input'])))
        train_out.append(np.uint8(np.array(t['output'])))

    test_in, test_out = [], []
    for t in task['test']:
        test_in.append(np.uint8(np.array(t['input'])))
        if 'output' in t:
            test_out.append(np.uint8(np.array(t['output'])))

    return train_in, train_out, test_in, test_out


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
    fig, axs = plt.subplots(2, num_train, figsize=(3 * num_train, 3 * 2))
    fig.suptitle(title, fontsize=16, y=1.08)
    if num_train == 1:
        axs = axs[..., None]
    for i in range(num_train):
        plot_one(taskdata[0][i], axs[0, i], 'Train Input')
        plot_one(taskdata[1][i], axs[1, i], 'Train Output')
    plt.tight_layout()
    plt.show()

    num_test = len(taskdata[2])
    fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2))
    if num_test == 1:
        axs = axs[..., None]
    for i in range(num_test):
        plot_one(taskdata[2][i], axs[0, i], 'Test Input')
        plot_one(taskdata[3][i], axs[1, i], 'Test Output')
    plt.tight_layout()
    plt.show()

class Task():
    def __init__(self, tasknum, taskfile):
        self.taskfile = taskfile
        self.tasknum = tasknum
        with open(taskfile, 'r') as f:
            self.task = json.load(f)
        self.origdata = get_train_test(self.task)
        self.update_data(self.origdata)

    def _colorpermute(self, img, swaps, kb):
        newimg = img.copy()
        for k, v in zip(np.arange(kb, 10), swaps):
            newimg[img == k] = v
        return newimg

    def update_data_with_colormap(self, colormap, data=None, kb=0):
        if data is None:
            data = self.origdata
        tartgetdata = []
        for image_list in data:
            dlist = []
            for img in image_list:
                dlist.append(self._colorpermute(img, colormap, kb))
            tartgetdata.append(dlist)
        self.update_data(tartgetdata)

    def swap_colorpairs(self, colorpairs):
        """ Input a list of colors to swap """
        if len(np.unique([c[0] for c in colorpairs])) != len(np.unique([c[0] for c in colorpairs])):
            print("Color mapping not one to one skipping")
            return
        colormap = np.arange(10)
        for c in colorpairs:
            colormap[c[0]] = c[1]
            colormap[c[1]] = c[0]
        self.update_data_with_colormap(colormap, self.data, 0)

    def randomize_colors(self, keep_black=False):
        kb = int(keep_black)
        colormap = np.random.permutation(np.arange(kb, 10))
        self.update_data_with_colormap(colormap, self.origdata, kb)

    def update_data(self, data):
        self.data = data
        self.io = []
        train_in, train_out, self.test_in, self.test_out = self.data
        for i, o in zip(train_in, train_out):
            self.io.append([i, o])

    def reset_to_original(self):
        self.update_data(self.origdata)

    def show_task(self):
        """Show the tasks data"""
        plot_task(self.data, str(self.tasknum) + ' ' + self.taskfile)


dslfuncs = {}
imageproperties = {}


class DSLFunction():
    def __init__(self, func, in_types, out_types, num_repeats):
        self.name = func.__name__
        self.func = func
        self.in_types = in_types if not isinstance(in_types, str) else (in_types,)
        self.out_types = out_types if not isinstance(out_types, str) else (out_types,)
        self.variable_length_outputs = np.any(['-multiple' in t for t in self.out_types])
        self.num_repeats = num_repeats

    def __call__(self, *args):
        return self.func(*args)

    def info(self):
        print('%s : %s -> %s | Repeat %d' % (self.name, self.in_types, self.out_types, self.num_repeats))


def register(in_types, out_types, num_repeats=1):
    def _thunk(func):
        name = func.__name__
        dslfuncs[name] = DSLFunction(func, in_types, out_types, num_repeats=num_repeats)
        return func

    return _thunk


def imageproperty(func):
    name = func.__name__
    imageproperties[name] = func
    return func


@register("Image", "ImageList")
def split_by_color(img, crop=True):
    """ Input an image, splits image list of images by different colors"""
    if img.max() == 0:
        return None
    cols = np.unique(img[img > 0])
    outs = []
    for c in cols:
        colimg = np.zeros_like(img)
        colimg[img == c] = img[img == c]
        outs.append(colimg)
    if crop:
        return _composite_imagelist_to_imagelist(outs, _crop_nonzero)
    else:
        return outs


@register("Image", "ImageList")
def split_by_blobs(img, conn=8, crop=True):
    """ splits image based on connected components (8 connected) """
    if img.max() == 0:
        return None
    ncon, cons, stats, cent = cv2.connectedComponentsWithStats(np.uint8(img > 0), connectivity=conn)
    sortedcons = [c for _, c in sorted(zip(stats[1:, cv2.CC_STAT_AREA], np.arange(1, ncon)), reverse=True)]
    outs = []
    for c in sortedcons:
        consimg = np.zeros_like(img)
        consimg[cons == c] = img[cons == c]
        outs.append(consimg)
    if crop:
        return _composite_imagelist_to_imagelist(outs, _crop_nonzero)
    else:
        return outs


@register("Image", "ImageList")
def split_by_blobs_4connected(img):
    return split_by_blobs(img, 4)


@register("Image", "ImageList")
def split_by_color_and_blobs(img):
    """ splits image by color then splits by connected components (8 connected) """
    lst = split_by_color(img)
    if lst is None:
        return None
    return _composite_imagelist_to_imagelist(lst, split_by_blobs)


@register(("Image", "Shape"), "ImageList")
def split_by_shape(img, targetshape):
    """ splits image by size if image size is integer multiple of target size """
    (rd, cd), mods = np.divmod(img.shape, targetshape)
    if rd + cd <= 2 or np.sum(mods) > 0:
        return None
    outs = []
    ts = targetshape
    for r in range(rd):
        for c in range(cd):
            crop = img[r * ts[0]:(r + 1) * ts[0], c * ts[1]:(c + 1) * ts[1]]
            outs.append(crop)
    return outs


@register(("Image", "Color"), "ImageList")
def split_by_lines_with_color(img, color):
    """Splits based on horizontal and vertical lines accross the image"""
    vlines = list(np.where(np.all(img == img[0, :], axis=0) & (img[0, :] == color))[0])
    hlines = list(np.where(np.all(img == img[:, :1], axis=1) & (img[:, 0] == color))[0])
    if len(hlines) == 0 and len(vlines) == 0:
        return None
    hlines = [-1] + hlines + [img.shape[0]]
    vlines = [-1] + vlines + [img.shape[1]]
    outs = []
    for r1, r2 in zip(hlines[:-1], hlines[1:]):
        for c1, c2 in zip(vlines[:-1], vlines[1:]):
            crop = img[r1 + 1:r2, c1 + 1:c2]
            if crop.size > 0:
                outs.append(crop)
    return outs if len(outs) > 0 else None


@register("Image", "ImageList")
def split_by_lines(img):
    """Splits based on horizontal and vertical lines accross the image"""
    vlines = list(np.where(np.all(img == img[0, :], axis=0) & (img[0, :] > 0))[0])
    hlines = list(np.where(np.all(img == img[:, :1], axis=1) & (img[:, 0] > 0))[0])
    if len(hlines) == 0 and len(vlines) == 0:
        return None
    hlines = [-1] + hlines + [img.shape[0]]
    vlines = [-1] + vlines + [img.shape[1]]
    outs = []
    for r1, r2 in zip(hlines[:-1], hlines[1:]):
        for c1, c2 in zip(vlines[:-1], vlines[1:]):
            crop = img[r1 + 1:r2, c1 + 1:c2]
            if crop.size > 0:
                outs.append(crop)
    return outs if len(outs) > 0 else None


########################################################
############## Image to Image ##########################
########################################################

@register("Image", "Image")
def crop_nonzero(img):
    """ Input an image, crops the non zero bounding box"""
    return _crop_nonzero(img)


@register("Image", "Image")
def pixelwise_not(img):
    cols = np.unique(img)
    if not len(cols) == 2:
        #         return None
        img = img.copy()
        img[img > cols[1]] = cols[1]  # TODO: Minor hack, check out how it turns out else revert
    newimg = np.zeros_like(img)
    newimg[img == cols[0]] = cols[1]
    newimg[img == cols[1]] = cols[0]
    return newimg


########################################################
############# List to Image ############################
########################################################

@register("ImageList", "Image")
def pixelwise_and(lst):
    """ If colors are different, uses color of img1 """
    if not len(lst) == 2:
        return None
    img1, img2 = lst
    if img1.max() == 0 or img2.max() == 0:
        return None
    if not np.all(img1.shape == img2.shape):
        return None
    andmap = np.logical_and(img1 > 0, img2 > 0)
    newimg = np.zeros_like(img1)
    newimg[andmap] = img1[andmap]
    return newimg


@register("ImageList", "Image")
def pixelwise_overlap(lst):
    """ Overlaps all images from last to first order """
    if len(lst) == 1:
        return None
    if not np.all([lst[-1].shape == l.shape for l in lst[:-1]]):
        return None
    newimg = lst[-1].copy()
    for img in reversed(lst[:-1]):
        nz = img > 0
        newimg[nz] = img[nz]
    return newimg


@register("ImageList", "Image")
def pixelwise_xor(lst):
    """ Pixelwise XOR based on non zero locations """
    if not len(lst) == 2:
        return None
    img1, img2 = lst
    if img1.max() == 0 or img2.max() == 0:
        return None
    if not np.all(img1.shape == img2.shape):
        return None
    newimg = np.zeros_like(img1)
    xormap = np.logical_xor(img1 > 0, img2 > 0)
    newimg[xormap] = img2[xormap]
    img1map = np.logical_and(xormap, img1 > 0)
    newimg[img1map] = img1[img1map]
    return newimg


# @register("ImageList", "ImageList")
# def crop_nonzero_multi(lst):
#     return _composite_imagelist_to_imagelist(lst, _crop_nonzero)

# @register(("ImageList", "ImageOperator",), "ImageList", 2) ### Makes things too slow
def _composite_imagelist_to_imagelist(lst, func):
    outs = []
    for img in lst:
        res = func(img)
        if res is None:
            return None
        if isinstance(res, list):
            outs.extend(res)
        else:
            outs.append(res)
    return outs


# @register("ImageList", "Image")
# def non_repeated_image_in_list(lst):   ################## Very Niche, need to check for bugs
#     """Returns the unique image if others are repeated multiple times"""
#     unique_images, reps = _unique_image_with_counts(lst)
#     is_unique = [r==1 for r in reps]
#     if not np.sum(is_unique) == 1:
#         return None
#     else:
#         return unique_images[np.where(is_unique)[0][0]]

@register("ImageList", "Image")
def least_repeated_image_in_list(lst):  ################## Very Niche
    """Returns the least repeated image if it is uniquely lowest count"""
    unique_images, reps = _unique_image_with_counts(lst)
    argmin = np.argwhere(reps == np.min(reps))
    if not len(argmin) == 1:
        return None
    else:
        return unique_images[argmin[0][0]]


@register("ImageList", "Image")
def most_repeated_image_in_list(lst):  ################## Very Niche
    """Returns the most repeated image if it is uniquely highest count"""
    unique_images, reps = _unique_image_with_counts(lst)
    argmax = np.argwhere(reps == np.max(reps))
    if not len(argmax) == 1:
        return None
    else:
        return unique_images[argmax[0][0]]


# @register("ImageList", "Image")
# def access_first(lst):
#     return lst[0]

# @register("ImageList", "Image")
# def access_last(lst):
#     return lst[-1]

@register(("ImageList", "ImageProperty"), "Image", 2)
def min_by_prop(lst, key):
    return min(lst, key=key)


@register(("ImageList", "ImageProperty"), "Image", 2)
def max_by_prop(lst, key):
    return max(lst, key=key)


# @register("ImageList", "Image")
# def symmetrical_image_in_list(lst):
#     is_sym = [_is_symmetrical(img) for img in lst]
#     if not np.any(is_sym):
#         return None
#     for img, sym in zip(lst, is_sym):
#         if sym:
#             return img

########################################################
############# List to List #############################
########################################################

# @register(("ImageList", "ImageProperty"), "ImageList")
# def sort_by_prop(lst, key):
#     return sorted(lst, key=key)

# @register("ImageList", "ImageList")
# def reverse_list(lst):
#     return list(reversed(lst))

########################################################
############# Image to Color ###########################
########################################################

# @register("Image", "Color-multiple")
# def image_colors_raster(img):
#     """ Return unique colors in raster scan order """
#     _, unique_idx = np.unique(img, return_index=True)
#     return list(img.ravel()[np.sort(unique_idx)])

# @register("Image", "Color-multiple")
# def image_colors_sorted(img):
#     """ Return unique colors by sort order """
#     return np.unique(img)

# @register("Image", "Color-multiple")
# def line_colors(img):
#     """ Returns colors of lines in raster scan order """
#     vlines = np.where(np.all(img == img[0,:], axis=0))
#     hlines = np.where(np.all(img == img[:,:1], axis=1))
#     if len(hlines) == 0 and len(vlines) == 0:
#         return None
#     cols = np.concatenate([img[0, vlines].squeeze(0), img[hlines, 0].squeeze(0)])
#     _, unique_idx = np.unique(cols, return_index=True)
#     return list(cols[np.sort(unique_idx)])


########################################################
############# Image Properties #########################
########################################################

@imageproperty
def count_colors(img):
    return len(np.unique(img))


@imageproperty
def count_nonzero(img):
    return np.sum(img > 0)


@imageproperty
def is_symmetrical(img):
    return np.any([np.array_equal(img, sym) \
                   for sym in [np.flipud(img), np.fliplr(img), img.T, np.fliplr(img.T)]])


#### Create new properties #####
@register("Color", "ImageProperty")
def create_color_counter(color):
    return lambda img: _count_single_color(img, color)


################## Other helpers #######################

def _unique_images(lst):
    unique_list = []
    for img in lst:
        if not np.any([np.array_equal(img, u) for u in unique_list]):
            unique_list.append(img)
    return unique_list


def _unique_image_with_counts(lst):
    unique_list, rep_count = [], []
    for img in lst:
        match = []
        for i, x in enumerate(unique_list):
            eq = np.array_equal(img, x)
            rep_count[i] += int(eq)
            match.append(eq)
        if not np.any(match):
            unique_list.append(img)
            rep_count.append(1)
    return unique_list, rep_count


def _crop_nonzero(img):
    """ Input an image, crops the non zero bounding box"""
    if img.max() == 0:
        return None
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax + 1, cmin:cmax + 1]


def _is_symmetrical(img):
    return np.any([np.array_equal(img, sym) \
                   for sym in [np.flipud(img), np.fliplr(img), img.T, np.fliplr(img.T)]])


def _image_colors_sorted(img):
    """ Return unique colors by sort order """
    return np.unique(img)


def _count_single_color(img, color):
    return np.sum(img == color)


#### Helper functions

def ravel(tupleOfTuples):
    return itertools.chain(*tupleOfTuples)


def pretty_print_program(program, poolnames, postprocess=None):
    for stage in program:
        funcname, args_idx = stage
        stagetext = ''
        for name, args in zip(poolnames, args_idx):
            for idx in args:
                if name == 'ImageProperty' and name in imageproperties:
                    ## Little cosmetic but depends on imageproperty injection style to the pools
                    stagetext += list(imageproperties.keys())[idx] + ', '
                #                 elif name == 'ImageOperator':
                #                     stagetext += list(imageoperators.keys())[idx] + ', '
                else:
                    stagetext += f'{name}[{idx}], '
        print(f'{funcname}({stagetext})')
    if postprocess is not None:
        print(postprocess)
    print()


def reverse_range(top, bottom=0):
    " Creates a reversed range because using python 'reversed' is not a persistant iterator"
    return range(top - 1, bottom - 1, -1)


def serialize_multiple(data, data_types):
    if not isinstance(data, tuple):
        data = (data,)
    new_data, new_data_types = [], []
    for di, dt in enumerate(data_types):
        #         try:
        if '-multiple' in dt:
            dt = dt[:-9]
            new_data.extend(data[di])
            new_data_types.extend([dt for _ in data[di]])
        else:
            new_data_types.append(dt)
            new_data.append(data[di])
    #         except:
    #             global debug
    #             debug = data, data_types
    #             print(len(data), data_types)
    #             assert False
    return tuple(new_data), new_data_types


class DataPool():
    def __init__(self, data_type, num_readback_entries=None):
        self.data_type = data_type
        self.data = []
        self.entry_lengths = [0]
        self.datalen = 0
        self.num_readback_entries = num_readback_entries
        self.index = reverse_range(len(self.data))
        self.update_range()

    def iterator(self, num_data):
        return itertools.combinations(self.index, num_data)  ## TODO: Might need change to permutations

    def append_data(self, data):
        self.data.append(data)

    def remove_last_data(self):
        self.data.pop(-1)

    def update_range(self):
        diff = len(self.data) - self.datalen
        if diff == 0:
            return
        elif diff > 0:
            self.datalen = len(self.data)
            self.entry_lengths.append(diff)
        else:
            self.datalen = len(self.data)
            self.entry_lengths.pop(-1)

        if self.num_readback_entries is not None:
            bottom = 0 if len(self.entry_lengths) <= self.num_readback_entries \
                else self.entry_lengths[-(self.num_readback_entries + 1)]
            self.index = reverse_range(len(self.data), bottom)
        else:
            self.index = reverse_range(len(self.data))  ## Range is reversed to prioritize last


class OutputPool():
    def __init__(self, poolnames, num_readback_entries):
        self.poolnames = poolnames
        num_readback_entries = num_readback_entries or [None for _ in self.poolnames]
        self.pools = {name: DataPool(name, nre) for name, nre in zip(self.poolnames, num_readback_entries)}

    def valid_data_iterator(self, input_types):
        iters = []
        for name, pool in self.pools.items():
            ### IMPORTANT TODO - all DSL functions must have be written in poolnames order to support this
            num_data = sum([it == name for it in input_types])
            iters.append(pool.iterator(num_data))
        return itertools.product(*iters)

    def index2data(self, index):
        data = []
        for poolidx, dataidx in zip(self.poolnames, index):
            for idx in dataidx:
                data.append(self.pools[poolidx].data[idx])
        return data

    def update(self, data_tuple, data_types):
        if not isinstance(data_tuple, tuple):
            data_tuple = (data_tuple,)
        #         print("Updating", len(data), data_types)
        for data, dtype in zip(data_tuple, data_types):
            self.pools[dtype].append_data(data)

        for name in self.pools:
            self.pools[name].update_range()

    def remove_last_entry(self, data_types):
        for dtype in data_types:
            self.pools[dtype].remove_last_data()

        for name in self.pools:
            self.pools[name].update_range()

    def _len(self):
        return {name: len(pool.data) for name, pool in self.pools.items()}

    def get_pool(self, name):
        return self.pools[name].data


class PostprocessChecker():
    def __init__(self, task):
        self.task = task
        self.name = 'undefined'

    def _is_valid(self):
        return True

    def matches_basic_conditions(self, func_outs):
        return True

    def all_match(self, func_outs):
        for io, out in zip(self.task.io, func_outs):
            img, target = io
            if not np.array_equal(self.postprocess_program(out), target):
                return False
        return True

    def postprocess_program(self, img):
        return img

    def check(self, func_outs):
        if not self.matches_basic_conditions(func_outs):
            return False
        return self.all_match(func_outs)


class SetOutputColor(PostprocessChecker):
    def __init__(self, task):
        super().__init__(task)
        self.name = 'set_output_color'
        self.setcolor, self.valid = self._is_valid()

    def _is_valid(self):
        """ Valid to use only all outputs have single color """
        allimgs_flat = []
        for io in self.task.io:
            allimgs_flat.extend(io[1].ravel())
        cols = list(np.unique(allimgs_flat))
        if 0 in cols:
            cols.pop(0)
        if len(cols) == 1:
            return cols[0], True
        else:
            return None, False

    def matches_basic_conditions(self, func_outs):
        return np.all([np.array_equal(fo.shape, io[1].shape) \
                       for fo, io in zip(func_outs, self.task.io)])

    def postprocess_program(self, img):
        resimg = img.copy()
        resimg[resimg > 0] = self.setcolor
        return resimg


class ProgramRunner():
    def __init__(self, program, initial_pool, dsl, postprocess=None):
        self.program = program
        self.initial_pool = initial_pool
        self.dsl = dsl
        self.postprocess = postprocess

    def __call__(self, img):
        return self.run_program(img)

    def show(self):
        pretty_print_program(self.program, self.initial_pool.poolnames, self.postprocess)

    def run_program(self, img, diagnose=False):
        "Diagnose is to be used to debug programs"
        pool = deepcopy(self.initial_pool)
        pool.update(img, ['Image'])
        func_output = None
        for stage in self.program:
            funcname, data_idx = stage
            func = self.dsl[funcname]
            output_types = func.out_types
            try:  ## These can occur when a program works on an image but not another of the same task
                data = pool.index2data(data_idx)
                func_output = func(*data)
                if func.variable_length_outputs:
                    func_output, serial_output_types = serialize_multiple(func_output, output_types)
                else:
                    serial_output_types = output_types
                pool.update(func_output, serial_output_types)
            except:
                #                 global debugpool
                #                 debugpool = pool
                #                 global debugprog
                #                 debugprog = self.program
                #                 func_output = func(*data)
                #                 pool.update(func_output, output_types)
                #                 print('Error')
                return None
            if diagnose:  ### To be implemented
                pass
        return func_output


class ProgramSearcher():
    def __init__(self, task, initial_pool, dsl, timeout=None, postprocessors=[]):
        self.task = task
        self.dsl = dsl
        self.initial_pool = initial_pool
        self.timeout = timeout or np.inf
        self.test_outs = []

        self.postprocess = None
        self.postprocessors = []
        for pp in postprocessors:
            postp = pp(task)
            if postp.valid:
                self.postprocessors.append(postp)

    def progrunner(self, program):
        return ProgramRunner(program, self.initial_pool, self.dsl, self.postprocess)

    def postprocess_program_check(self, func_outs):
        for pp in self.postprocessors:
            if pp.check(func_outs):
                return True, pp.name, pp.postprocess_program
        return False, None, None

    def all_match(self, program, current_out):
        ## TODO: Clean up this mess
        if len(self.postprocessors) > 0:  # Exectute only if postprocessors available
            if not isinstance(current_out, np.ndarray):
                return False
            ## TODO: This is only valid for color based postprocessing, need to standardize this
            if not np.array_equal(current_out.shape, self.target.shape):
                return False
        else:
            if not np.array_equal(current_out, self.target):  ## Prevent building program when no match
                return False

        progrunner = self.progrunner(program)
        task = self.task
        train_outs = []
        matched = True
        for io in task.io:
            img, target = io
            out = progrunner(img)
            train_outs.append(out)
            if out is None:
                return False
            elif not np.array_equal(out, target):
                matched = False

        self.test_outs = []
        for test_img in task.test_in:
            out = progrunner(test_img)
            self.test_outs.append(out)
            if out is None:
                return False

        if not matched:
            matched, self.postprocess, pp_func = self.postprocess_program_check(train_outs)
            if self.postprocess is not None:
                self.test_outs = [pp_func(tout) for tout in self.test_outs]

        return matched

    def should_terminate(self, prog):
        if (len(prog) == self.maxlen) or (time.time() - self.starttime > self.timeout):
            return True

    def search(self, maxlen, prevent_duplicates=False):
        if maxlen < 3:
            prevent_duplicates = False  ## For smaller programs, the overhead makes it slower overall
        self.starttime = time.time()
        self.run_count = 0
        self.maxlen = maxlen
        self.match_condition = self.all_match
        #         self.terminate = lambda prog: len(prog) == self.maxlen
        funcnames = []
        for key in self.dsl:
            funcnames.extend([key for _ in range(self.dsl[key].num_repeats)])
        pool = deepcopy(self.initial_pool)
        img, self.target = self.task.io[0]
        pool.update(img, ['Image'])
        nodes_visited = {} if prevent_duplicates else None
        program = self._programsearch(pool=pool,
                                      out=None,
                                      program=[],
                                      function_names=funcnames,
                                      nodes_visited=nodes_visited)
        time_elapsed = time.time() - self.starttime
        if program is None:
            return False, self.run_count, time_elapsed, None, self.test_outs
        else:
            return True, self.run_count, time_elapsed, self.progrunner(program), self.test_outs

    def _programsearch(self, pool, out, program, function_names, nodes_visited):

        ## Match Check
        if self.match_condition(program, out):
            return program

        ## Termination
        if self.should_terminate(program):
            return None

        ## DFS
        for funcidx, funcname in enumerate(function_names):
            new_function_names = function_names[:funcidx] + function_names[funcidx + 1:]
            func = self.dsl[funcname]
            input_types = func.in_types
            output_types = func.out_types
            for valid_data_index in pool.valid_data_iterator(input_types):
                if nodes_visited is not None:
                    node_key = funcname + str(valid_data_index)
                    if node_key in nodes_visited:
                        continue
                    else:
                        nodes_visited[node_key] = True  ## Cannot avoid this double access
                self.run_count += 1
                data = pool.index2data(valid_data_index)
                try:
                    func_output = func(*data)
                    if func_output is None:
                        continue
                except:
                    #                     print(input_types)
                    #                     pretty_print_program(program + [(func.name, valid_data_index)], pool.poolnames)
                    #                     print([len(p) for p in pool.pools])
                    #                     global debugpool
                    #                     debugpool = pool
                    #                     func_output = func(*data)
                    continue

                if func.variable_length_outputs:
                    func_output, serial_output_types = serialize_multiple(func_output, output_types)
                else:
                    serial_output_types = output_types

                program.append((func.name, valid_data_index))
                pool.update(func_output, serial_output_types)

                nv_copy = None if nodes_visited is None else nodes_visited.copy()
                finalprogram = self._programsearch(pool,
                                                   func_output,
                                                   program,
                                                   new_function_names,
                                                   nv_copy)

                if finalprogram is not None:
                    return finalprogram

                if nodes_visited is not None:
                    nodes_visited.pop(node_key, None)
                program.pop(-1)
                pool.remove_last_entry(serial_output_types)
        return None

train_tasks, eval_tasks, test_tasks = get_tasks(data_path)
allpoolnames = ['Image', 'ImageList', 'ImageProperty', 'Color', 'Shape']
readback = [2, 2, None, 1, 1]

print("%d DSL Functions" % len(dslfuncs))
for f in dslfuncs:
    dslfuncs[f].info()
print()
print("%d Image Properties" % len(imageproperties))
for f in imageproperties:
    print(f)


def progsearch_check():
    # Sanity Check
    run_count = 0
    maxlen = 3
#     task = train_tasks[71] # solvable with len 3
#     task = train_tasks[215] # solvable with len 4
#     task = train_tasks[110] # solvable with len 5
    task = train_tasks[78] # solvable with len 3
#     task = train_tasks[130] # unsolvable
    initpool = OutputPool(poolnames=allpoolnames, num_readback_entries=readback)
    props, propdt = serialize_multiple( ([f for _, f in imageproperties.items()],), ['ImageProperty-multiple'])
    initpool.update(props, propdt)
    colors = _image_colors_sorted(task.io[0][0])
    colors, dt = serialize_multiple( colors, ['Color-multiple'])
    initpool.update(colors, dt)
    outshape = list(task.io[0][1].shape)
    initpool.update(outshape, ['Shape'])
    searcher = ProgramSearcher(task=task, initial_pool=initpool, dsl=dslfuncs, postprocessors=[SetOutputColor])
    solved, run_count, time_elapsed, progrunner, test_outs = searcher.search(maxlen=maxlen, prevent_duplicates=True)
    print(task.tasknum, solved, run_count, '%0.3fs' % time_elapsed)
    if solved:
        print()
        progrunner.show()
progsearch_check()


def search_on_task(arg):
    task, maxlen, timeout, dsldict, try_backgroundswap = arg
    if try_backgroundswap:
        allpix = []
        for io in task.io:
            allpix.extend(np.ravel(io[0]))
            allpix.extend(np.ravel(io[1]))
        cols = np.unique(allpix)
    else:
        cols = [0]
    for c in cols:
        task.swap_colorpairs([(c, 0)])
        pool = OutputPool(poolnames=allpoolnames, num_readback_entries=readback)
        props, propdt = serialize_multiple(([f for _, f in imageproperties.items()],), ['ImageProperty-multiple'])
        pool.update(props, propdt)
        colors = _image_colors_sorted(task.io[0][0])
        colors, dt = serialize_multiple(colors, ['Color-multiple'])
        pool.update(colors, dt)
        outshape = list(task.io[0][1].shape)
        pool.update(outshape, ['Shape'])
        searcher = ProgramSearcher(task=task, initial_pool=pool, dsl=dsldict,
                                   postprocessors=[SetOutputColor], timeout=timeout)
        solved, run_count, time_elapsed, progrunner, test_outs = searcher.search(maxlen=maxlen, prevent_duplicates=True)
        task.swap_colorpairs([(c, 0)])
        if solved:
            for idx, img in enumerate(test_outs):
                newimg = img.copy()
                for k, v in zip([0, c], [c, 0]):
                    newimg[img == k] = v
                test_outs[idx] = newimg
            break

    return (task.tasknum, solved, run_count, time_elapsed, progrunner, test_outs)


def multiprocess_run(tasks, df, task_type, lenvals, excludelist, timeout, dsldict, validate_test=False,
                     try_bgswap=False):
    alltasks = deepcopy(tasks)
    #     alltasks = [task for task in alltasks if df.loc[task.tasknum, 'Type'] == task_type
    #                                              and task.tasknum not in excludelist]

    print(f'Searching {len(alltasks)} tasks with timeout {timeout}')
    foundprogs = {}
    for maxlen in lenvals:
        tic = time.time()
        search_args = [(task, maxlen, timeout, dsldict, try_bgswap) for task in alltasks]
        with MultiProcessingPool(4) as p:
            for sr in p.imap_unordered(search_on_task, search_args):
                if sr[1]:
                    foundprogs[sr[0]] = sr[2:]
        alltasks = [task for task in alltasks if task.tasknum not in foundprogs]
        print('Maxlen %d Total time %fs' % (maxlen, time.time() - tic))
        print(foundprogs.keys())
        print("Tasks solved:", len(foundprogs.keys()))

    if validate_test:
        for tnum in foundprogs:
            target_outs = tasks[tnum].test_out
            prog_outs = foundprogs[tnum][-1]
            if not np.all([np.array_equal(po, to) for po, to in zip(prog_outs, target_outs)]):
                print(f'Task {tnum} got false solution')
    return foundprogs

runtime_error = False
try:
    if not np.all(test_tasks[0].io[0][0] == eval_tasks[0].io[0][0]):
        test_foundprogs = multiprocess_run(test_tasks, 'test_task_df', 'Cut and Transform', lenvals=[1,2,3,4],
                                  timeout=None, excludelist=[], dsldict=dslfuncs, try_bgswap=True)
    else:
        print('Skipping commit time run')
        for task in test_tasks:
            task.randomize_colors(keep_black=False)
        test_foundprogs = multiprocess_run(test_tasks, 'test_task_df', 'Cut and Transform', lenvals=[1,2],
                                  timeout=None, excludelist=[], dsldict=dslfuncs, try_bgswap=True)
        for task in test_tasks:
            task.reset_to_original()
except:
    print("Runtime error")
    runtime_error = True


def flattener(pred):
    str_pred = str([list(row) for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='output_id')

if not runtime_error:

    for prognum in test_foundprogs:
        task_id = test_tasks[prognum].taskfile.split('/')[-1].split('.')[0]
        pred_outs = test_foundprogs[prognum][-1]
        for idx, img in enumerate(pred_outs):
            output_id = f'{task_id}_{idx}'
            pred_1 = flattener(img)
            pred = pred_1 + ' ' + pred_1 + ' ' + pred_1 + ' '
            submission.loc[output_id, 'output'] = pred
            display(submission.loc[output_id])
    if len(test_foundprogs) > 0:
        submission.to_csv('submission.csv')