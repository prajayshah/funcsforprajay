import os
import sys
import re
import pickle
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats, ndimage, io
from scipy.optimize import curve_fit
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import random

from skimage.io import imread
from sklearn.decomposition import PCA
import tifffile as tf
import math
import csv

from funcsforprajay.wrappers import plot_piping_decorator


############### GENERALLY USEFUL FUNCTIONS #############################################################################

# retrieve the last modified time for a file path
def get_last_modified_time(file_path: str):
    return os.path.getmtime(file_path)


# return the parent directory of a file:
def return_parent_dir(file_path: str):
    return file_path[:[(s.start(), s.end()) for s in re.finditer('/', file_path)][-1][0]]


def list_in_dir(dir_path: str):
    assert os.path.exists(dir_path)
    return os.listdir(dir_path)


def timer(start, end):
    """source: https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes
    -seconds-and-milliseco"""
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds) + ' hours, mins, seconds')


# report sizes of variables
def _sizeof_fmt(num, suffix='B'):
    """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# report sizes of variables
def print_size_of(var):
    print(_sizeof_fmt(sys.getsizeof(var)))


# report sizes of variables
def print_size_vars():
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, _sizeof_fmt(size)))


# finding paths to files with a certain extension
def path_finder(umbrella, *args, is_folder=False):
    '''
    returns the path to the single item in the umbrella folder
    containing the string names in each arg
    is_folder = False if args is list of files
    is_folder = True if  args is list of folders
    '''
    # list of bools, has the function found each argument?
    # ensures two folders / files are not found
    found = [False] * len(args)
    # the paths to the args
    paths = [None] * len(args)

    if is_folder:
        for root, dirs, files in os.walk(umbrella):
            for folder in dirs:
                for i, arg in enumerate(args):
                    if arg in folder:
                        assert not found[i], 'found at least two paths for {},' \
                                             'search {} to find conflicts' \
                            .format(arg, umbrella)
                        paths[i] = os.path.join(root, folder)
                        found[i] = True

    elif not is_folder:
        for root, dirs, files in os.walk(umbrella):
            for file in files:
                for i, arg in enumerate(args):
                    if arg in file:
                        assert not found[i], 'found at least two paths for {},' \
                                             'search {} to find conflicts' \
                            .format(arg, umbrella)
                        paths[i] = os.path.join(root, file)
                        found[i] = True

    print(paths)
    for i, arg in enumerate(args):
        if not found[i]:
            raise ValueError('could not find path to {}'.format(arg))

    return paths


# progress bar
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# find the closest value in a list to the given input
def findClosest(arr, input):
    if type(arr) == list:
        arr = np.array(arr)
    subtract = arr - input
    positive_values = abs(subtract)
    # closest_value = min(positive_values) + input
    index = np.where(positive_values == min(positive_values))[0][0]
    closest_value = arr[index]

    return closest_value, index


# flatten list of lists
def flattenOnce(list: Union[list, tuple, np.ndarray], asarray=False):
    """ flattens a nested list by one nesting level (should be able to run multiple times to get further down if needed for
     deeper nested lists) """
    # if not asarray:
    #     isnot_list = [False for i in list if type(i) != list]
    #     if len(isnot_list) > 0:
    #         print('not a nested list, so returning original list.')
    #         return list
    #     l_ = []
    #     return [l_.extend() for i in list]
    # elif asarray:
    #     return np.asarray([x for i in list for x in i])
    if not type(list):
        raise TypeError('input must be a list or tuple')
    if not asarray:
        return [x for i in list for x in i]
    elif asarray:
        return np.asarray([x for i in list for x in i])



# save .pkl files from the specified pkl_path
def save_pkl(obj, pkl_path: str):
    if os.path.exists(return_parent_dir(pkl_path)):
        os.makedirs(return_parent_dir(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f".. saved to {pkl_path} -- ")
    else:
        raise NotADirectoryError(f'parent directory of {pkl_path} cannot be reached.')


# load .pkl files from the specified pkl_path
def load_pkl(pkl_path: str):
    if os.path.exists(pkl_path):
        f = pickle.load(open(pkl_path, 'rb'))
        return f
    else:
        raise FileNotFoundError(f"{pkl_path} not found")


def ImportTiff(tiff_path, frames: Union[tuple, int] = None) -> np.ndarray:
    """
    Import multi-frame tiff file from provided `tiff_path`, between specified frames (if provided, optional).

    :param tiff_path: path to multi-frame tiff file to load
    :param frames: optional, load frames between frames specified, or load single frame if int provided.
    :return: stack of tiff images
    """
    if frames and type(frames) == tuple:
        im_stack = tf.imread(tiff_path, key=range(frames[0], frames[1]))
    elif frames and type(frames) == int:
        im_stack = tf.imread(tiff_path, key=frames)
    else:
        # import cv2
        # ret, images = cv2.imreadmulti(tiff_path, [], cv2.IMREAD_ANYCOLOR)
        # if len(images) > 0:
        #     im_stack = np.asarray(images)
        # im_stack = tf.imread(tiff_path)
        try:
            stack = []
            with tf.TiffFile(tiff_path) as tif:
                for page in tif.pages:
                    image = page.asarray()
                    stack.append(image)
            im_stack = np.array(stack)
            if len(im_stack) == 1: im_stack = im_stack[0]
        except Exception as ex:
            try:
                im_stack = imread(tiff_path, plugin='pil')
            except Exception as ex:
                raise ImportError('unknown error in loading tiff stack.')

    return im_stack

def makeFrameAverageTiff(frames: Union[int, list, tuple], tiff_path: str = None, stack: np.ndarray = None,
                         peri_frames: int = 100, save_dir: str = None, to_plot=False, **kwargs) -> np.ndarray:
    """Creates, plots and/or saves an average image of the specified number of peri-key_frames around the given frame from either the provided tiff_path or the stack array.

    :param frames: key frames to create peri-average frames.
    :param tiff_path: path to tiff file for collecting images.
    :param stack: image stack array to use for collecting peri-average images
    :param peri_frames: number of frames to collect pre- and post- from key frame
    :param save_dir: directory to save tiff images to
    :param to_plot: if true, show peri-frame average image
    :param kwargs: see kwargs under imagingplus.plotting.plotting.plotImg
    :return: peri-frame averaged array images
    """

    if type(frames) == int:
        frames = [frames]

    stack = ImportTiff(tiff_path) if not stack else stack

    imgs = []
    for idx, frame in enumerate(frames):
        # im_batch_reg = tf.imread(tif_path, key=range(0, self.output_ops['batch_size']))

        if 0 > frame - peri_frames // 2:
            peri_frames_low = frame
        else:
            peri_frames_low = peri_frames // 2
        if stack.shape[0] < frame + peri_frames // 2:
            peri_frames_high = stack.shape[0] - frame
        else:
            peri_frames_high = peri_frames // 2
        im_sub_reg = stack[frame - peri_frames_low: frame + peri_frames_high]

        avg_sub = np.mean(im_sub_reg, axis=0)

        # convert to 8-bit
        avg_sub = convert_to_8bit(avg_sub, 0, 255)

        if save_dir:
            if '.tif' in save_dir: save_dir = os.path.dirname(save_dir) + '/'
            save_path = save_dir + f'/{frames[idx]}_s2preg_frame_avg.tif'
            os.makedirs(save_dir, exist_ok=True)

            print(f"\t.. Saving averaged s2p registered tiff for frame: {frames[idx]}, to: {save_path}")
            tf.imwrite(save_path, avg_sub, photometric='minisblack')

        imgs.append(avg_sub)

    return np.asarray(imgs)



############### STATS/DATA ANALYSIS FUNCTIONS ##########################################################################

def eq_line_2points(p1, p2):
    """
    returns the y = mx + b equation given two points.
    :param p1: point 1, x and y coord
    :param p2: point 2, x and y coord
    """
    from numpy import ones, vstack
    from numpy.linalg import lstsq
    x_coords, y_coords = zip(*[p1, p2])
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    print(f'x: {x_coords}')
    print(f'y: {y_coords}')
    print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
    return m, c


def moving_average(a, n=4):
    """
    a: array to process
    n: window over which to collect moving average
    """
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# calculate correlation across all cells
def corrcoef_array(array):
    df = pd.DataFrame(array)
    correlations = {}
    columns = df.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[str(col_a) + '__' + str(col_b)] = stats.pearsonr(df.loc[:, col_a], df.loc[:, col_b])

    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']
    corr = result['PCC'].mean()

    print('Correlation coefficient: %.2f' % corr)

    return corr, result


def points_in_circle_np(radius, x0=0, y0=0):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    for x, y in zip(x_[x], y_[y]):
        yield x, y


# calculate distance between 2 points on a cartesian plane
def calc_distance_2points(p1: tuple, p2: tuple):
    """
    uses the hypothenus method to calculate the straight line distance between two given points on a 2d cartesian plane.
    :param p1: point 1
    :param p2: point 2
    :return:
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def lin_regression(x: list, y: list):
    return np.poly1d(np.polyfit(x, y, 1))(range(np.min(x), np.max(x)))


# retrieve x, y points from csv
def xycsv(csvpath):
    xline = []
    yline = []
    with open(csvpath) as csv_file:
        csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')
        for row in csv_file:
            xline.append(int(float(row['xcoords'])))
            yline.append(int(float(row['ycoords'])))
    # assumption = line is monotonic
    line_argsort = np.argsort(yline)
    xline = np.array(xline)[line_argsort]
    yline = np.array(yline)[line_argsort]
    return xline, yline


# read matlab array
def load_matlab_array(path):
    """
    Returns a matlab array read in from the path given in path.
    :param path: path to the matlab output file ending in .mat
    :return: array
    """
    return io.loadmat(path)


# read csv
def read_csv(csvpath, as_pandas = True, sep=None):
    """
    Import a csv file. Optinally return as pandas dataframe.
    :param csvpath: path to the .csv file to import.
    :param as_pandas:
    :param sep: separator to use when reading in to a pandas dataframe.
    :return:
    """
    if as_pandas:
        csv_file = pd.read_csv(csvpath, sep=sep)
    else:
        with open(csvpath) as csv_file:
            csv_file = csv.DictReader(csv_file, fieldnames=None, dialect='excel')
    return csv_file


# find percentile of a value within an array
def find_percentile(d, threshold):
    return sum(np.abs(d) < threshold) / float(len(d)) * 100

# calc z score, given std and mean
def zscore(dat: float, std: float, mean: float):
    """Calc z score for an input score, given std and mean of the overall distribution.
    :param score: score to conver to z score
    :param std:
    :param mean:
    :return:
    """
    return (dat - mean) / std


def convert_to_positive(arr):
    min_value = min(arr)
    if min_value >= 0:
        return arr
    else:
        return np.array([x + abs(min_value) for x in arr])


def decay_constant(arr: np.ndarray, threshold: float = None, signal_rate: Union[float, int] = 1):
    """measure the timeconstant of decay of a signal array.
    If signal rate is provided, will return in units of time, otherwise will return as the index of the array.
    """
    if not type(arr) is np.ndarray:
        raise TypeError('provide `arr` input as type = np.array')
    max_value = arr.max()  # peak Flu value after stim
    max_index = arr.argmax()  # peak Flu value after stim
    threshold = (1 - np.exp(-1)) * max_value if not threshold else threshold  # set threshold to be at 1/e x peak
    try:
        x_ = np.where(arr[max_index:] < threshold)[0][0]  # find index AFTER the index of the max value of the trace, where the trace decays to the threshold value
        return x_ / signal_rate  # convert frame # to time
    except Exception:
        print(f'Could not find decay below the maximum value of the trace provided. max: {max_value}, max index: {max_index}, decay threshold: {threshold}')


def decay_constant_logfit_method(arr):
    """use the polyfit on the logarithm of the signal to calculate the decay coefficient

    >>> r = 0.5
    >>> a = 10
    >>> n = 10
    >>> arr = np.array([a*np.exp((-r)*i) for i in range(n)])
    >>> decay_constant = decay_constant_logfit_method(arr=arr)
    """
    coeffs = np.polyfit(range(len(arr)), np.log(arr), 1)
    decay_constant = -coeffs[0]
    return decay_constant


def decay_timescale(arr, decay_constant=None, signal_rate=1):
    """
    Calculation of the decay timescale (optionally adjusting for signal collection data rate).
    Decay timescale is defined as (1 - 1/e) * initial [max] value of the signal.

    :param arr:
    :param decay_constant:
    :param signal_rate:
    :return:

    >>> r = 0.5
    >>> a = 10
    >>> n = 10
    >>> arr = np.array([a*np.exp((-r)*i) for i in range(n)])
    >>> decay_constant = decay_constant_logfit_method(arr=arr)
    >>> decay_timescale(arr=arr, decay_constant=decay_constant, signal_rate=30)
    """

    max_value = np.max(arr)

    if decay_constant is None:
        decay_constant = decay_constant_logfit_method(arr=arr)

    timescale = -(1 / decay_constant) * np.log(1 - 1 / np.e)
    half_life = -(1 / decay_constant) * np.log(0.5)

    plot = False
    if plot:
        time_steps = np.arange(0, len(arr))
        decay = max_value * np.exp(-decay_constant * time_steps)
        plt.plot(decay)
        plt.axhline(arr.max() * 0.5)
        plt.axvline(half_life)
        plt.suptitle('half life')
        plt.show()

        plt.plot(decay)
        plt.axhline(arr.max() * (1 - 1 / np.e))
        plt.axvline(timescale)
        plt.suptitle('timescale value')
        plt.show()

    return timescale / signal_rate


# logaritmic regression fit function
def logarithmic_regression_fit(x, y):
    """
    Fit a logarithmic regression function to the data
    :param x: x data
    :param y: y data
    :return: a, b
    """
    def log_func(x, a, b):
        return a + b * np.log(x)

    # Finding the optimal parameters :
    popt, pcov = curve_fit(log_func, x, y)
    print("a  = ", popt[0])
    print("b  = ", popt[1])

    # Predicting values:
    y_pred = log_func(x, popt[0], popt[1])

    # Check the accuracy :
    from sklearn.metrics import r2_score
    Accuracy = r2_score(y, y_pred)
    print(f'R**2: {Accuracy}')

    return popt[0], popt[1], y_pred

############### HELPFUL FUNCTIONS FOR PLOTTING ##########################################################################

# random func for rotating images and calculating the image intensity along one axis of the image
def rotate_img_avg(input_img, angle):
    """this function will be used to rotate the input_img (ideally will be the avg seizure image) at the given angle.
    The function also will return the 1 x n length average across non-zero values along the x axis.

    :param input_img: ndarray comprising the image
    :param angle: the angle to rotate the image with (in degrees), +ve = counter-clockwise
    """
    full_img_rot = ndimage.rotate(input_img, angle, reshape=True)

    return full_img_rot


# PCA decomposition(/compression) of an image
def pca_decomp_image(input_img, components: int = 3, plot_quant: bool = False):
    """
    the method for PCA based decomposition/compression of an image, and also (optional) quantification of the resulting
    image across the x axis

    :param input_img: ndarray; input image
    :param components: int; # of principle components to use for the PCA decomposition (compression) of the input_img
    :param plot_quant: bool; plot quantification of the average along x-axis of the image
    :return: ndarray; compressed image, imshow plots of the original and PCA compressed images, as well as plots of average across the x-axis
    """

    print("Extracting the top %d eigendimensions from image" % components)
    pca = PCA(components)
    img_transformed = pca.fit_transform(input_img)
    img_compressed = pca.inverse_transform(img_transformed)

    if plot_quant:
        # quantify the input image
        fig = plt.figure(figsize=(15, 5))
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.imshow(input_img, cmap='gray')
        ax2.imshow(img_compressed, cmap='gray')

        img_t = input_img.T
        avg = np.zeros([img_t.shape[0], 1])
        for i in range(len(img_t)):
            x = img_t[i][img_t[i] > 0]
            if len(x) > 0:
                avg[i] = x.mean()
            else:
                avg[i] = 0

        ax3.plot(avg)
        ax3.set_xlim(20, len(img_t) - 20)
        ax3.set_title('average plot quantification of the input img', wrap=True)
        plt.show()

        # quantify the PC reconstructed image
        fig = plt.figure(figsize=(15, 5))
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.imshow(input_img, cmap='gray')
        ax2.imshow(img_compressed, cmap='gray')

        img_compressed = img_compressed.T
        avg = np.zeros([img_compressed.shape[0], 1])
        for i in range(len(img_compressed)):
            x = img_compressed[i][img_compressed[i] > 0]
            if len(x) > 0:
                avg[i] = x.mean()
            else:
                avg[i] = 0

        ax3.plot(avg)
        ax3.set_xlim(20, len(img_compressed.T) - 20)
        ax3.title.set_text('average plot quantification of the PCA compressed img - %s dimensions' % components)

        plt.show()

    return img_compressed


# grouped average / smoothing of a 1dim array (basically the same as grouped average on imageJ)
def smoothen_signal(signal, w):
    return np.convolve(signal, np.ones(w), 'valid') / w


############### PLOTTING FUNCTIONS #####################################################################################
# general plotting function for making plots quickly (without having to write out a bunch of lines of code)
# custom colorbar for heatmaps
from matplotlib.colors import LinearSegmentedColormap


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return LinearSegmentedColormap('CustomMap', cdict)


# generate an array of random line_colors
def _get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def _color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def _generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = _get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([_color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def make_random_color_array(n_colors):
    """
    Generates a list of random line_colors for an input number of line_colors required.

    :param n_colors: # of line_colors required
    :return: list of line_colors in RGB
    """
    colors = []
    for i in range(0, n_colors):
        colors.append(_generate_new_color(colors, pastel_factor=0.2))
    return colors


@plot_piping_decorator(verbose=False)
def make_general_scatter(x_list: list, y_data: list, fig=None, ax=None,
                         **kwargs):  ## TODO remove the double plotting, just give option to plot all individual as stamps or together!
    """
    General function for quick, simple plotting of data lists as scatters. NOTE: THIS FUNC MAKES TWO SEPARATE PLOTS if given >1 dataset to plot.

    :param x_list: list of x_points for plots, must match one to one to y_data
    :param y_data: list of y_data for plots, must match one to one to x_list
    :param kwargs: (optional)
        line_colors: list, line_colors to use to plot >1 data sets
        ax_y_labels: list, y_labels to use to plot >1 data sets
        ax_x_labels: list, x_labels to use to plot >1 data sets
        y_label: str, y_labels to use to plot the combined main plot
        x_label: str, x_labels to use to plot the combined main plot
        legend_labels: list[str], legend_labels to use to plot the combined main plot
        ax_titles: list of ax_titles to use to plot >1 data traces
        x_lim: tuple, used to set x_lim of plot
        suptitle: str, used for suptitle of fig
    :return None
    """

    assert len(y_data) == len(x_list), 'y_data length does not match x_list length'

    num_plots = len(x_list)

    if 'line_colors' not in kwargs.keys():
        colors = make_random_color_array(num_plots)
    else:
        assert type(kwargs['line_colors']) is list and len(kwargs['line_colors']) == len(
            x_list), 'provide line_colors argument in list form matching number of traces to plot'
        colors = kwargs['line_colors']

    edgecolors = colors if 'edgecolors' not in [*kwargs] else kwargs['edgecolors']

    # set plotting properties
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 0.8
    if 's' in kwargs.keys():
        size = kwargs['s']
    else:
        size = 50
    lw = 0 if 'lw' not in [*kwargs] else kwargs['lw']

    # check integrity of function call arguments
    if 'ax_y_labels' in kwargs.keys() and type(kwargs['ax_y_labels']) is list: assert len(
        kwargs['y_labels']) == num_plots
    if 'ax_x_labels' in kwargs.keys() and type(kwargs['ax_x_labels']) is list: assert len(
        kwargs['x_labels']) == num_plots
    if 'ax_titles' in kwargs.keys() and type(kwargs['ax_titles']) is list: assert len(kwargs['ax_titles']) == num_plots

    if 'legend_labels' in kwargs.keys() and type(kwargs['legend_labels']) is list:
        assert len(kwargs[
                       'legend_labels']) == num_plots, 'legend_labels len does not match number of plots to make (len of x_list)'
        label = kwargs['legend_labels']
    else:
        label = ['']

    if num_plots > 1:
        ncols = 4
        nrows = len(x_list) // ncols
        if len(x_list) % ncols > 0:
            nrows += 1

        fig2, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[(ncols * 4), (nrows * 3)])
        counter = 0

        axs[0, 0].set_xlabel(kwargs['ax_x_labels'][0]) if 'ax_x_labels' in kwargs.keys() else None
        axs[0, 0].set_ylabel(kwargs['ax_y_labels'][0]) if 'ax_y_labels' in kwargs.keys() else None

    # prep for single small plot with all plots
    # fig, ax = plt.subplots(figsize=(4, 3))
    # fig, ax = kwargs['fig'], kwargs['ax']

    for i in range(num_plots):
        if 'supress_print' in [*kwargs] and kwargs['supress_print'] != True: print(
            f"plotting plot # {i + 1} out of {num_plots}, {len(x_list[i])} points")
        ax.scatter(x=x_list[i], y=y_data[i], facecolors=colors[i], edgecolors=edgecolors[i], alpha=alpha, lw=lw, s=size,
                   label=label[i])

        if num_plots > 1:
            a = counter // ncols
            b = counter % ncols

            # make plot for individual key/experiment trial
            ax2 = axs[a, b]
            ax2.scatter(x=x_list[i], y=y_data[i], facecolors=colors[i], edgecolors=edgecolors[i], alpha=alpha, lw=lw,
                        s=size, label=label[i])
            ax2.set_xlim(-50, 50)
            ax2.set_title(f"{kwargs['ax_titles'][i]}") if 'ax_titles' in kwargs.keys() else None
            counter += 1
        else:
            ax.set_title(f"{kwargs['ax_titles'][i]}") if 'ax_titles' in kwargs.keys() else None

    ax.set_xlim(kwargs['x_lim'][0], kwargs['x_lim'][1]) if 'x_lim' in kwargs.keys() else None
    ax.set_ylim(kwargs['y_lim'][0], kwargs['y_lim'][1]) if 'y_lim' in kwargs.keys() else None
    ax.set_xlabel(kwargs['x_label']) if 'x_label' in kwargs.keys() else None
    ax.set_ylabel(kwargs['y_label']) if 'y_label' in kwargs.keys() else None

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left") if 'legend_labels' in kwargs.keys() else None

    fig.tight_layout(pad=1.8)

    if num_plots > 1:
        fig2.suptitle(f"all plots individual")
        fig2.tight_layout(pad=1.8)
        # print(f"\nsaving multi-axes plot to {kwargs['save_path_full']}") if 'save_path_full' in kwargs.keys() else None
        # fig2.savefig(kwargs['save_path_full']) if 'save_path_full' in kwargs.keys() else None
        fig2.show()
        # print(f'trying to return {fig2}')
        # return fig2


# @print_start_end_plot
@plot_piping_decorator()
def make_general_plot(data_arr, x_range=None, twin_x: bool = False, plot_avg: bool = True, plot_std: bool = True,
                      **kwargs):
    """
    General function for quick, simple plotting of arbritary data arrays.

    :param data_arr: list of data-traces to plot, or np.ndarray containing data-traces
    :param x_range: list of x-ranges to plot, or np.ndarray containing x-ranges
    :param twin_x: iff two traces, option to plot on same axis
    :param plot_avg: if more than two traces, whether to plot average of the data traces
    :param plot_std: if more than two traces, whether to plot std of the data traces, if false will plot individual data traces in random color
    :param kwargs: (optional)
        line_colors: list, line_colors to use to plot >1 data traces
        y_labels: list, y_labels to use to plot >1 data traces
        x_labels: list, x_labels to use to plot >1 data traces
        ax_titles: list of ax_titles to use to plot >1 data traces
        title: str, one title for one ax plotting
        y_label: str, one y label for one ax plotting
        x_label: str, one x label for one ax plotting
        fontsize: float, fontsizes within the plot of text labels
        v_span: tuple, vertical span fill - will be same for each axis
        suptitle: str, used for suptitle of fig
    :return None
    """

    f, axs = kwargs['fig'], [kwargs['ax']]
    # prepare for plotting over multiple axes if called for
    if len(axs) > 1:
        num_axes = len(axs)
    else:
        num_axes = 1
        # axs = np.array([axs])

    # create data arrays in the correct format for plotting
    if type(data_arr) is list:
        num_traces = len(data_arr)
    elif type(data_arr) is np.ndarray:
        num_traces = data_arr.shape[0]
    else:
        raise Exception('data_arr must be of type list of np.ndarray')

    # check if plotting multi-traces on 1 axis (but not twinx style!):
    if num_traces > num_axes and num_axes == 1:
        alpha = 0.3
    else:
        alpha = 1
        plot_avg = False  # turn off plotting of average trace
        plot_std = False  # turn off plotting of std trace from data

    # add twin x if called for:
    if num_traces == 2 and twin_x is True:
        ax = axs[0]
        ax2 = ax.twinx()
        axs = np.array([ax, ax2])
        num_axes = 2

    print(f'\nPlotting {num_traces} data traces across {num_axes} axes') if not twin_x else print(
        f'\nPlotting {num_traces} data traces across 1 axes (with twin_x)')

    # create x_range to use for plotting
    if x_range is not None:
        if type(x_range) is list:
            x_range = np.asarray(x_range)
        assert x_range.shape == data_arr.shape, print(
            '|- AssertionError: mismatch between data to plot and x_range provided for this data')
    else:
        x_range = np.empty_like(data_arr)
        for i in range(num_traces):
            x_range[i] = range(len(data_arr[i]))

    # make random line_colors for plotting
    if 'line_colors' not in kwargs.keys():
        colors = make_random_color_array(num_traces) if num_traces > 1 else ['black']
    else:
        assert type(kwargs['line_colors']) is list, print(
            '|- AssertionError: provide line_colors argument in list form')
        assert len(kwargs['line_colors']) == num_traces, print(
            '|- AssertionError: provide enough line_colors as number of traces to plot')
        colors = kwargs['line_colors']

    # check integrity of function call arguments
    if 'y_labels' in kwargs.keys() and len(kwargs['y_labels']) > 1: assert len(kwargs['y_labels']) == num_traces
    if 'x_labels' in kwargs.keys() and len(kwargs['x_labels']) > 1: assert len(kwargs['x_labels']) == num_traces
    if 'ax_titles' in kwargs.keys(): assert len(kwargs['ax_titles']) == num_traces

    # shrink or enlarge the fontsize option:
    fontsize = kwargs['fontsize'] if 'fontsize' in kwargs.keys() else 10

    # make the plot using each provided data trace
    ax_counter = 0

    if 'v_span' in kwargs.keys() and type(kwargs['v_span']) is tuple:
        axs[ax_counter].axvspan(kwargs['v_span'][0], kwargs['v_span'][1], color='indianred', zorder=1)

    if plot_std is False or num_traces == 1:  # only plot individual lines if plot_std is inactive
        print(f'.. plotting {num_traces} individual traces on {num_axes} axes')
        for i in range(num_traces):
            axs[ax_counter].plot(x_range[i], data_arr[i], color=colors[i], alpha=alpha)
            if num_axes > 1:
                axs[ax_counter].set_xlabel(kwargs['ax_titles'][i],
                                           fontsize=fontsize) if 'ax_titles' in kwargs.keys() else None
                axs[ax_counter].set_xlabel(kwargs['x_labels'][i],
                                           fontsize=fontsize) if 'x_labels' in kwargs.keys() else None
                axs[ax_counter].set_ylabel(kwargs['y_labels'][i],
                                           fontsize=fontsize) if 'y_labels' in kwargs.keys() else None
                ax_counter += 1
    if num_axes == 1 and twin_x is False and num_traces > 1:
        if plot_avg:
            print(f'.. plotting average trace of {data_arr.shape[0]} traces on 1 axis')
            axs[ax_counter].plot(x_range[0], np.mean(data_arr, axis=0), color='black', alpha=1,
                                 zorder=data_arr.shape[0] + 1)
        if plot_std:
            print(f'.. plotting std trace of {data_arr.shape[0]} traces on 1 axis')
            std_low = np.mean(data_arr, axis=0) - np.std(data_arr, axis=0)
            std_high = np.mean(data_arr, axis=0) + np.std(data_arr, axis=0)
            axs[ax_counter].fill_between(x_range[0], std_low, std_high, color='gray', alpha=0.5, zorder=0)

        axs[ax_counter].set_title(kwargs['title'], fontsize=fontsize * 1.1, wrap=True) if 'title' in kwargs.keys() else \
            axs[ax_counter].set_title(f"{num_traces} traces")
        axs[ax_counter].set_ylabel(kwargs['y_label'], fontsize=fontsize) if 'y_label' in kwargs.keys() else None
        axs[ax_counter].set_xlabel(kwargs['x_label'], fontsize=fontsize) if 'x_label' in kwargs.keys() else None
        axs[ax_counter].set_ylabel(kwargs['y_labels'], fontsize=fontsize) if 'y_labels' in kwargs.keys() else None
        axs[ax_counter].set_xlabel(kwargs['x_labels'], fontsize=fontsize) if 'x_labels' in kwargs.keys() else None

    return None


### plot the location of provided coordinates
@plot_piping_decorator(figsize=(5, 5), verbose=False)
def plot_coordinates(coords: list, frame_x: int, frame_y: int, background: np.ndarray = None, fig=None, ax=None,
                     **kwargs):
    """
    plot coordinate locations

    :param targets_coords: ls containing (x,y) coordinates of targets to plot
    :param background: np.array on which to plot coordinates, default is black background (optional)
    :param kwargs:
    """
    if background is None:
        background = np.zeros((frame_x, frame_y), dtype='uint16')
        ax.imshow(background, cmap='gray')
    else:
        ax.imshow(background, cmap='gray')

    if 'edgecolors' in kwargs.keys():
        edgecolors = kwargs['edgecolors']
    else:
        edgecolors = 'yellowgreen'

    # set facecolors of the plotted coordinates
    facecolors = kwargs['facecolors'] if 'facecolors' in kwargs.keys() else 'none'
    # shrink or enlarge the fontsize option:
    fontsize = kwargs['fontsize'] if 'fontsize' in kwargs.keys() else 10

    for (x, y) in coords:
        ax.scatter(x=x, y=y, edgecolors=edgecolors, facecolors=facecolors, linewidths=2.0)

    ax.set_title(kwargs['title'], fontsize=fontsize * 1.1, wrap=True) if 'title' in kwargs.keys() else ax.set_title(
        f"{len(coords)} coordinates")

    ax.margins(0)
    fig.tight_layout()


# plot a 2d histogram density plot
@plot_piping_decorator(figsize=(5, 5), verbose=False)
def plot_hist2d(data: np.array, fig=None, ax=None, **kwargs):
    """
    plot 2d histogram

    :param data: data array to be plotted
    :param kwargs:
    """
    # check data structure:
    assert data.shape[1] == 2 and data.ndim == 2, "data np.array shape must be (n, 2)"

    # set colormap for the 2d density plot
    cmap = kwargs['cmap'] if 'cmap' in kwargs.keys() else 'inferno'

    # set colormap for the 2d density plot
    bins = kwargs['bins'] if 'bins' in kwargs.keys() and len(kwargs['bins']) == 2 else [100, 100]
    print(f"|- plotting with: {bins} (Nx, Ny) bins [.1]")

    # shrink or enlarge the fontsize option:
    fontsize = kwargs['fontsize'] if 'fontsize' in kwargs.keys() else 10

    ax.hist2d(data[:, 0], data[:, 1], bins=bins, cmap=cmap)

    ax.set_title(kwargs['title'], fontsize=fontsize * 1.1, wrap=True) if 'title' in kwargs.keys() else ax.set_title(
        f"2d density plot, {bins} bins")
    ax.set_ylabel(kwargs['y_label'], fontsize=fontsize) if 'y_label' in kwargs.keys() else None
    ax.set_xlabel(kwargs['x_label'], fontsize=fontsize) if 'x_label' in kwargs.keys() else None

    ax.set_ylim(kwargs['y_lim'][0], kwargs['y_lim'][1]) if 'y_lim' in kwargs.keys() else None
    ax.set_xlim(kwargs['x_lim'][0], kwargs['x_lim'][1]) if 'x_lim' in kwargs.keys() else None

    ax.margins(0)
    fig.tight_layout()


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    from: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib/49601444

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# # plotting function for plotting a bar graph with the individual data points shown as well
# def plot_bar_with_points(data, title='', x_tick_labels=[], legend_labels: list = [], points: bool = True,
#                          bar: bool = True, colors: list = ['black'], ylims=None, xlims=True, text_list=None,
#                          x_label=None, y_label=None, alpha=0.2, savepath=None, expand_size_x=1, expand_size_y=1,
#                          shrink_text: float = 1, show_legend=False,
#                          paired=False, title_pad=20, **kwargs):
#     """
#     all purpose function for plotting a bar graph of multiple categories with the option of individual datapoints shown
#     as well. The individual datapoints are drawn by adding a scatter plot with the datapoints randomly jittered around the central
#     x location of the bar graph. The individual points can also be paired in which case they will be centered. The bar can also be turned off.
#
#     :param data: list; provide data from each category as a list and then group all into one list
#     :param title: str; title of the graph
#     :param x_tick_labels: labels to use for categories on x axis
#     :param legend_labels:
#     :param points: bool; if True plot individual data points for each category in data using scatter function
#     :param bar: bool, if True plot the bar, if False plot only the mean line
#     :param colors: line_colors (by category) to use for each x group
#     :param ylims: tuple; y axis limits
#     :param xlims: the x axis is used to position the bars, so use this to move the position of the bars left and right
#     :param x_label: x axis label
#     :param y_label: y axis label
#     :param text_list: list of text to add to each category of data on the plot
#     :param text_shift: float; number between 0.5 to 1 used to adjust precise positioning of the text in text_list
#     :param alpha: transparency of the individual points when plotted in the scatter
#     :param savepath: .svg file path; if given, the plot will be saved to the provided file path
#     :param expand_size_x: factor to use for expanding figure size
#     :param expand_size_y: factor to use for expanding figure size
#     :param paired: bool, if True then draw lines between data points of the same index location in each respective list in the data
#     :return: matplotlib plot
#     """
#
#     # collect some info about data to plot
#     w = 0.3  # mean bar width
#     xrange_ls = list(range(len(data)))
#     y = data
#     if len(colors) != len(xrange_ls):
#         colors = colors * len(xrange_ls)
#
#     # initialize plot
#     if 'fig' in kwargs.keys():
#         f = kwargs['fig']
#         ax = kwargs['ax']
#     else:
#         f, ax = plt.subplots(figsize=((5 * len(xrange_ls) / 2) * expand_size_x, 3 * expand_size_y))
#
#     if paired:
#         assert len(xrange_ls) > 1
#
#     # start making plot
#     if not bar:
#         for i in xrange_ls:
#             ## plot the mean line
#             ax.plot(np.linspace(xrange_ls[i] * w * 2.5 - w / 2, xrange_ls[i] * w * 2.5 + w / 2, 3), [np.mean(y[i])] * 3,
#                     color='black')
#         lw = 0,
#         edgecolor = None
#         # since no bar being shown on plot (lw = 0 from above) then use it to plot the error bars
#         ax.bar([x * w * 2.5 for x in xrange_ls],
#                height=[np.mean(yi) for yi in y],
#                yerr=[np.std(yi, ddof=1) for yi in y],  # error bars
#                capsize=4.5,  # error bar cap width in points
#                width=w,  # bar width
#                linewidth=lw,  # width of the bar edges
#                edgecolor=edgecolor,
#                color=(0, 0, 0, 0),  # face edgecolor transparent
#                zorder=2
#                )
#     elif bar:
#         if 'edgecolor' not in kwargs.keys():
#             edgecolor = 'black',
#             lw = 1
#         else:
#             edgecolor = kwargs['edgecolor'],
#             lw = 1
#         # plot bar graph
#         ax.errorbar([x * w * 2.5 for x in xrange_ls], [np.mean(yi) for yi in y], fmt='none',
#                     yerr=np.asarray([np.asarray([0, np.std(yi, ddof=1)]) for yi in y]).T, ecolor='gray',
#                     capsize=5, zorder=0)
#         ax.bar([x * w * 2.5 for x in xrange_ls],
#                height=[np.mean(yi) for yi in y],
#                # yerr=np.asarray([np.asarray([0, np.std(yi, ddof=1)]) for yi in y]).T,  # error bars
#                capsize=4.5,  # error bar cap width in points
#                width=w,  # bar width
#                linewidth=lw,  # width of the bar edges
#                edgecolor=edgecolor,
#                color=(0, 0, 0, 0),  # face edgecolor transparent
#                zorder=2
#                )
#     else:
#         AttributeError('something wrong happened with the bar bool parameter...')
#
#     ax.set_xticks([x * w * 2.5 for x in xrange_ls])
#     if len(xrange_ls) > 1:
#         ax.set_xticklabels(x_tick_labels, fontsize=10 * shrink_text, rotation=45)
#     else:
#         ax.set_xticklabels(x_tick_labels, fontsize=10 * shrink_text)
#
#     if xlims:
#         ax.set_xlim([(xrange_ls[0] * w * 2) - w * 1.20, (xrange_ls[-1] * w * 2.5) + w * 1.20])
#     elif len(xrange_ls) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
#         xlims_ = [-1, 1]
#         ax.set_xlim(xlims_)
#
#     if len(legend_labels) == 0:
#         if len(x_tick_labels) == 0:
#             x_tick_labels = [None] * len(xrange_ls)
#         legend_labels = x_tick_labels
#
#     if points:
#         if not paired:
#             for i in xrange_ls:
#                 # distribute scatter randomly across whole width of bar
#                 ax.scatter(xrange_ls[i] * w * 2.5 + np.random.random(len(y[i])) * w - w / 2, y[i], color=colors[i],
#                            alpha=alpha, label=legend_labels[i])
#
#         else:  # connect lines to the paired scatter points in the list
#             if len(xrange_ls) > 0:
#                 for i in xrange_ls:
#                     # plot points  # dont scatter location of points if plotting paired lines
#                     ax.scatter([xrange_ls[i] * w * 2.5] * len(y[i]), y[i], color=colors[i], alpha=0.5,
#                                label=legend_labels[i], zorder=3)
#                 for i in xrange_ls[:-1]:
#                     for point_idx in range(len(y[i])):  # draw the lines connecting pairs of data
#                         ax.plot([xrange_ls[i] * w * 2.5 + 0.058, xrange_ls[i + 1] * w * 2.5 - 0.048],
#                                 [y[i][point_idx], y[i + 1][point_idx]], color='black', zorder=2, alpha=alpha)
#
#                 # for point_idx in range(len(y[i])):  # slight design difference, with straight line going straight through the scatter points
#                 #     ax.plot([x * w * 2.5 for x in x],
#                 #             [y[i][point_idx] for i in x], color='black', zorder=0, alpha=alpha)
#
#             else:
#                 AttributeError('cannot do paired scatter plotting with only one data category')
#
#     if ylims:
#         ax.set_ylim(ylims)
#     elif len(xrange_ls) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
#         ylims = [0, 2 * max(data[0])]
#         ax.set_ylim(ylims)
#
#     # Hide the right and top spines
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#
#     ax.tick_params(axis='both', which='both', length=5)
#
#     # Only show ticks on the left and bottom spines
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#
#     ax.set_xlabel(x_label, fontsize=10 * shrink_text)
#     ax.set_ylabel(y_label, fontsize=10 * shrink_text)
#     if savepath:
#         plt.savefig(savepath)
#
#     # add text to the figure if given:
#     if text_list:
#         assert len(xrange_ls) == len(text_list), 'please provide text_list of same len() as data'
#         if 'text_shift' in kwargs.keys():
#             text_shift = kwargs['text_shift']
#         else:
#             text_shift = 0.8
#         if 'text_y_pos' in kwargs.keys():
#             text_y_pos = kwargs['text_y_pos']
#         else:
#             text_y_pos = max([np.percentile(y[i], 95) for i in xrange_ls])
#         for i in xrange_ls:
#             ax.text(xrange_ls[i] * w * 2.5 - text_shift * w / 2, text_y_pos, text_list[i]),
#
#     if len(legend_labels) > 1:
#         if show_legend:
#             ax.legend(bbox_to_anchor=(1.01, 0.90), fontsize=8 * shrink_text)
#
#     # add title
#     if 'fig' not in kwargs.keys():
#         ax.set_title(title, horizontalalignment='center', pad=title_pad,
#                      fontsize=11 * shrink_text, wrap=True)
#     else:
#         ax.set_title((title), horizontalalignment='center', pad=title_pad,
#                      fontsize=11 * shrink_text, wrap=True)
#
#     if 'show' in kwargs.keys():
#         if kwargs['show'] is True:
#             # Tweak spacing to prevent clipping of ylabel
#             f.tight_layout(pad=1.4)
#             f.show()
#         else:
#             return f, ax
#     else:
#         # Tweak spacing to prevent clipping of ylabel
#         f.tight_layout(pad=1.4)
#         f.show()


# histogram density plot with gaussian best fit line
def plot_hist_density(data: list, mean_line: bool = False, colors: list = None, fill_color: list = None,
                      legend_labels: list = [None], num_bins=10, best_fit_line='gaussian', **kwargs):
    """

    :param data: list; nested list containing the data; if only one array to plot then provide array enclosed inside list (e.g. [array])
    :param colors:
    :param fill_color:
    :param legend_labels:
    :param num_bins:
    :param kwargs:
    :return:
    """

    fig = kwargs['fig']
    ax = kwargs['ax']

    if colors is None:
        colors = ['black'] * len(data)
    if len(data) == 1 and fill_color is None:
        fill_color = ['steelblue']
    else:
        assert len(data) == len(colors)
        assert len(data) == len(fill_color), print('please provide a fill color for each dataset')

    if legend_labels is [None]:
        legend_labels = [None] * len(data)
    else:
        assert len(legend_labels) == len(data), print('please provide a legend label for all your data to be plotted!')

    # set the transparancy for the fill of the plot
    if 'alpha' in kwargs and (type(kwargs['alpha']) is float or kwargs['alpha'] == 1):
        alpha = kwargs['alpha']
    else:
        alpha = 0.3

    # make the primary histogram density plot
    zorder = 2
    for i in range(len(data)):
        # the histogram of the data
        bin_heights, bins, patches = ax.hist(data[i], num_bins, density=1, alpha=0.4, color=fill_color[i],
                                             label=legend_labels[i])  # histogram hidden currently

        # add a 'best fit' line
        if best_fit_line == 'powerlaw':
            from scipy.optimize import curve_fit

            def func_powerlaw(x, m, c, c0):
                return c0 + x ** m * c
            target_func = func_powerlaw

            X = np.linspace(bins[0], bins[-1], num_bins)
            y = bin_heights
            popt, pcov = curve_fit(target_func, X, y, maxfev = 1000000)

            ax.plot(X, target_func(X, *popt), linewidth=2, c=colors[i], zorder=zorder + i)
            ax.fill_between(X, target_func(X, *popt), color=fill_color[i], zorder=zorder + i, alpha=alpha)
            print(bins)
            print('m, c, c0: \n\t', popt)
            title = 'Histogram density: powerlaw fit'

        elif best_fit_line == 'gaussian':
            # fitting a gaussian
            mu = np.mean(data[i])  # mean of distribution
            sigma = np.std(data[i])  # standard deviation of distribution

            x = np.linspace(bins[0], bins[-1], num_bins * 5)
            popt = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                  np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))

            ax.plot(x, popt, linewidth=2, c=colors[i], zorder=zorder + i)
            ax.fill_between(x, popt, color=fill_color[i], zorder=zorder + i, alpha=alpha)

            title = (r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)))
        else:
            title = ''

        if mean_line:
            ax.axvline(x=np.nanmean(data[i]), c=fill_color[i], linewidth=2, zorder=0, linestyle='dashed')

    if 'x_label' in kwargs and kwargs['x_label'] is not None:
        ax.set_xlabel(kwargs['x_label'])
    if 'y_label' in kwargs and kwargs['y_label'] is not None:
        ax.set_ylabel(kwargs['y_label'])
    elif 'y_label' in kwargs and kwargs['y_label'] is None:
        pass
    else:
        ax.set_ylabel('Probability density')

    if 'show_legend' in kwargs and kwargs['show_legend'] is True:
        ax.legend()

    # set x limits
    if 'x_lim' in kwargs:
        ax.set_xlim(kwargs['x_lim'])

    # setting shrinking factor for font size for title
    if 'shrink_text' in kwargs.keys():
        shrink_text = kwargs['shrink_text']
    else:
        shrink_text = 1

    # add title
    if 'title' in kwargs and kwargs['title'] is not None:
        if len(data) == 1:
            ax.set_title(kwargs['title'] + title, wrap=True,
                         fontsize=12 * shrink_text)
        else:
            ax.set_title(kwargs['title'], wrap=True, fontsize=12 * shrink_text)
    else:
        if len(data) == 1:
            ax.set_title(title)
        else:
            ax.set_title('Histogram density plot')


    # if 'show' in kwargs.keys():
    #     if kwargs['show'] is True:
    #         # Tweak spacing to prevent clipping of ylabel
    #         fig.tight_layout()
    #         fig.show()
    #     else:
    #         pass
    # else:
    #     # Tweak spacing to prevent clipping of ylabel
    #     fig.tight_layout()
    #     fig.show()
    #
    # if 'fig' in kwargs.keys():
    #     return fig, ax


# imshow gray plot for a single frame tiff
def plot_single_tiff(tiff_path: str, title: str = None, frame_num: int = 0):
    """
    plots an image of a single tiff frame after reading using tifffile.
    :param tiff_path: path to the tiff file
    :param title: give a string to use as title (optional)
    :return: imshow plot
    """
    stack = tf.imread(tiff_path, key=frame_num)
    plt.imshow(stack, cmap='gray')
    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle('frame num: %s' % frame_num)
    plt.show()
    return stack


############### CALCIUM IMAGING RELATED STUFF ##########################################################################
# paq2py by Llyod Russel
def paq_read(file_path=None, plot=False):
    """
    Read PAQ file (from PackIO) into python
    Lloyd Russell 2015
    Parameters
    ==========
    file_path : str, optional
        full path to file to read in. if none is supplied a load file dialog
        is opened, buggy on mac osx - Tk/matplotlib. Default: None.
    plot : bool, optional
        plot the data after reading? Default: False.
    Returns
    =======
    data : ndarray
        the data as a m-by-n array where m is the number of channels and n is
        the number of datapoints
    chan_names : list of str
        the names of the channels provided in PackIO
    hw_chans : list of str
        the hardware lines corresponding to each channel
    units : list of str
        the units of measurement for each channel
    rate : int
        the acquisition sample rate, in Hz
    """

    # file load gui
    if file_path is None:
        import Tkinter
        import tkFileDialog
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename()
        root.destroy()

    # open file
    fid = open(file_path, 'rb')

    # get sample rate
    rate = int(np.fromfile(fid, dtype='>f', count=1))

    # get number of channels
    num_chans = int(np.fromfile(fid, dtype='>f', count=1))

    # get channel names
    chan_names = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        chan_name = ''
        for j in range(num_chars):
            chan_name = chan_name + chr(np.fromfile(fid, dtype='>f', count=1))
        chan_names.append(chan_name)

    # get channel hardware lines
    hw_chans = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        hw_chan = ''
        for j in range(num_chars):
            hw_chan = hw_chan + chr(np.fromfile(fid, dtype='>f', count=1))
        hw_chans.append(hw_chan)

    # get acquisition units
    units = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        unit = ''
        for j in range(num_chars):
            unit = unit + chr(np.fromfile(fid, dtype='>f', count=1))
        units.append(unit)

    # get data
    temp_data = np.fromfile(fid, dtype='>f', count=-1)
    num_datapoints = int(len(temp_data) / num_chans)
    data = np.reshape(temp_data, [num_datapoints, num_chans]).transpose()

    # close file
    fid.close()

    # plot
    if plot:
        # import matplotlib
        # matplotlib.use('QT4Agg')
        import matplotlib.pylab as plt
        f, axes = plt.subplots(num_chans, 1, sharex=True, figsize=(10, num_chans), frameon=False)
        for idx, ax in enumerate(axes):
            ax.plot(data[idx])
            ax.set_xlim([0, num_datapoints - 1])
            ax.set_ylim([data[idx].min() - 1, data[idx].max() + 1])
            # ax.set_ylabel(units[idx])
            ax.set_title(chan_names[idx])
        plt.tight_layout()
        plt.show()

    return {"data": data,
            "chan_names": chan_names,
            "hw_chans": hw_chans,
            "units": units,
            "rate": rate,
            "num_datapoints": num_datapoints}


# useful for returning indexes when a
def threshold_detect(signal, threshold):
    '''lloyd russell'''
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    frames = np.where(thresh_signal)
    return frames[0]


# normalize dFF for 1dim array
def dff(flu, baseline=None):
    """delta F over F ratio (not % dFF )"""
    if baseline is not None:
        flu_dff = (flu - baseline) / baseline
    else:
        flu_mean = np.mean(flu, 1)
        flu_dff = (flu - flu_mean) / flu_mean

    return flu_dff


# simple ZProfile function for any sized square in the frame (equivalent to ZProfile function in Fiji)
def ZProfile(movie, area_center_coords: tuple = None, area_size: int = -1, plot_trace: bool = True,
             plot_image: bool = True, plot_frame: int = 1, vasc_image: np.array = None, **kwargs):
    """
    from Sarah Armstrong

    Plot a z-profile of a movie, averaged over space inside a square area

    movie = can be np.array of the TIFF stack or a tiff path from which it is read in
    area_center_coords = coordinates of pixel at center of box (x,y)
    area_size = int, length and width of the square in pixels
    plot_frame = which movie frame to take as a reference to plot the area boundaries on
    vasc_image = optionally include a vasculature image tif of the correct dimensions to plot the coordinates on.
    """

    if type(movie) is str:
        movie = tf.imread(movie)
    print('plotting zprofile for TIFF of shape: ', movie.shape)

    # assume 15fps for 1024x1024 movies and 30fps imaging for 512x512 movies
    if movie.shape[1] == 1024:
        img_fps = 15
    elif movie.shape[1] == 512:
        img_fps = 30
    else:
        img_fps = None

    assert area_size <= movie.shape[1] and area_size <= movie.shape[2], "area_size must be smaller than the image"
    if area_size == -1:  # this parameter used to plot whole FOV area
        area_size = movie.shape[1]
        area_center_coords = (movie.shape[1] / 2, movie.shape[2] / 2)
    assert area_size % 2 == 0, "pls give an even area size"

    x = area_center_coords[0]
    y = area_center_coords[1]
    x1 = int(x - 1 / 2 * area_size)
    x2 = int(x + 1 / 2 * area_size)
    y1 = int(y - 1 / 2 * area_size)
    y2 = int(y + 1 / 2 * area_size)
    smol_movie = movie[:, y1:y2, x1:x2]
    smol_mean = np.nanmean(smol_movie, axis=(1, 2))
    print('|- Output shape for z profile: ', smol_mean.shape)

    if plot_image:
        f, ax1 = plt.subplots()
        ref_frame = movie[plot_frame, :, :]
        if vasc_image is not None:
            assert vasc_image.shape == movie.shape[1:], 'vasculature image has incompatible dimensions'
            ax1.imshow(vasc_image, cmap="binary_r")
        else:
            ax1.imshow(ref_frame, cmap="binary_r")

        rect1 = patches.Rectangle(
            (x1, y1), area_size, area_size, linewidth=1.5, edgecolor='r', facecolor="none")

        ax1.add_patch(rect1)
        ax1.set_title("Z-profile area")
        plt.show()

    if plot_trace:
        if 'figsize' in kwargs:
            figsize = kwargs['figsize']
        else:
            figsize = [10, 4]
        fig, ax2 = plt.subplots(figsize=figsize)
        if img_fps is not None:
            ax2.plot(np.arange(smol_mean.shape[0]) / img_fps, smol_mean, linewidth=0.5, color='black')
            ax2.set_xlabel('Time (sec)')
        else:
            ax2.plot(smol_mean, linewidth=0.5, color='black')
            ax2.set_xlabel('frames')
        ax2.set_ylabel('Flu (a.u.)')
        if 'title' in kwargs:
            ax2.set_title(kwargs['title'])
        plt.show()

    return smol_mean


def SaveDownsampledTiff(tiff_path: str = None, stack: np.array = None, group_by: int = 4, save_as: str = None,
                        plot_zprofile: bool = True):
    """
    Create and save a downsampled version of the original tiff file. Original tiff file can be given as a numpy array stack
    or a str path to the tiff.

    :param tiff_path: path to the tiff to downsample
    :param stack: numpy array stack of the tiff file already read in
    :param group_by: specified interval for grouped averaging of the TIFF
    :param save_as: path to save the downsampled tiff to, if none provided it will save to the same directory as the provided tiff_path
    :param plot_zprofile: if True, plot the zaxis profile using the full TIFF stack provided.
    :return: numpy array containing the downsampled TIFF stack
    """
    print('downsampling of tiff stack...')

    if save_as is None:
        assert tiff_path is not None, "please provide a save path to save_as"
        save_as = tiff_path[:-4] + '_downsampled.tif'

    if stack is None:
        # open tiff file
        print('|- working on... %s' % tiff_path)
        stack = tf.imread(tiff_path)

    resolution = stack.shape[1]

    # plot zprofile of full TIFF stack
    if plot_zprofile:
        ZProfile(movie=stack, plot_image=True, title=tiff_path)

    # downsample to 8-bit
    stack8 = np.full_like(stack, fill_value=0)
    for frame in np.arange(stack.shape[0]):
        stack8[frame] = convert_to_8bit(stack[frame], 0, 255)

    # stack8 = stack

    # grouped average by specified interval
    num_frames = stack8.shape[0] // group_by
    # avgd_stack = np.empty((num_frames, resolution, resolution), dtype='uint16')
    avgd_stack = np.empty((num_frames, resolution, resolution), dtype='uint8')
    frame_count = np.arange(0, stack8.shape[0], group_by)
    for i in np.arange(num_frames):
        frame = frame_count[i]
        avgd_stack[i] = np.mean(stack8[frame:frame + group_by], axis=0)

    avgd_stack = avgd_stack.astype(np.uint8)

    # bin down to 512 x 512 resolution if higher resolution
    shape = np.shape(avgd_stack)
    if shape[1] != 512:
        # input_size = avgd_stack.shape[1]
        # output_size = 512
        # bin_size = input_size // output_size
        # final_stack = avgd_stack.reshape((shape[0], output_size, bin_size,
        #                                   output_size, bin_size)).mean(4).mean(2)
        final_stack = avgd_stack
    else:
        final_stack = avgd_stack

    # write output
    print("\nsaving %s to... %s" % (final_stack.shape, save_as))
    tf.imwrite(save_as, final_stack, photometric='minisblack')

    return final_stack


def subselect_tiff(tiff_path: str = None, tiff_stack: np.array = None, select_frames: tuple = (0, 0),
                   save_as: str = None):
    if tiff_stack is None:
        # open tiff file
        print('running subselecting tiffs')
        print('|- working on... %s' % tiff_path)
        tiff_stack = tf.imread(tiff_path)

    stack_cropped = tiff_stack[select_frames[0]:select_frames[1]]

    # stack8 = convert_to_8bit(stack_cropped)

    if save_as is not None:
        tf.imwrite(save_as, stack_cropped, photometric='minisblack')

    return stack_cropped


def make_tiff_stack(sorted_paths: list, save_as: str):
    """
    read in a bunch of tiffs and stack them together, and save the output as the save_as

    :param sorted_paths: list of string paths for tiffs to stack
    :param save_as: .tif file path to where the tif should be saved
    """

    num_tiffs = len(sorted_paths)
    print('working on tifs to stack: ', num_tiffs)
    data_arr = None

    with tf.TiffWriter(save_as, bigtiff=True) as tif:
        for i, tif_ in enumerate(sorted_paths):
            with tf.TiffFile(tif_, multifile=True) as input_tif:
                data = input_tif.asarray()
            msg = ' -- Writing tiff: ' + str(i + 1) + ' out of ' + str(num_tiffs) + f' to {save_as}'
            print(msg, end='\r')
            tif.save(data)
            if data_arr is None:
                data_arr = data
            else:
                data_arr = np.append(data_arr, data, axis=0)

    tf.imwrite(save_as, data_arr, bigtiff=True)

    return data_arr


def convert_to_8bit(img, target_type_min=0, target_type_max=255):
    """
    :param img:
    :param target_type:
    :param target_type_min:
    :param target_type_max:
    :return:
    """
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(np.uint8)
    return new_img

#######


#### UTILS
def dataplot_frame_options():
    import matplotlib as mpl

    mpl.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'legend.frameon': False,
        'figure.subplot.wspace': .01,
        'figure.subplot.hspace': .01,
    })
    sns.set()
    sns.set_style('white')


def lineplot_frame_options(fig, ax, x_label='', y_label='', fs=10):

    sns.set()
    sns.set_style('white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(True)
    ax.margins(0)
    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)




