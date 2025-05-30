# imports
import os
from typing import Union

import numpy as np
import functools
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# global plotting params
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)
sns.set()
sns.set_style('white')


# %% UTILITY FUNCS

# wrapper for piping plots in and out of figures
def plotting_decorator(figsize=(3, 3), nrows=1, ncols=1, apply_image_frame_options=False, apply_heatmap_options=False):
    def plotting_decorator(plotting_func):
        """
        Wrapper to help simplify creating plots from matplotlib.pyplot

        :param plotting_func: plotting convenience function to be wrapped
        :return: fig and ax objects if requested, and/or shows the figure as specified

        Examples:
        ---------
        1) in this example the fig and ax will be taken directly from the kwargs inside the inner wrapper
        >>> @plotting_decorator
        >>> def example_decorated_plot(title='', **kwargs):
        >>>     fig, ax = kwargs['fig'], kwargs['ax']  # need to add atleast this line to each plotting function's definition
        >>>     print(f'|-kwargs inside example_decorated_plot definition: {kwargs}')
        >>>     ax.plot(np.random.rand(10))
        >>>     ax.set_title(title)

        >>> example_decorated_plot()

        """

        @functools.wraps(plotting_func)
        def inner(*args, **kwargs):

            return_fig_obj = False

            # set number of rows, cols and figsize
            if 'nrows' in [*kwargs]:
                nrows_ = kwargs['nrows']
            else:
                nrows_ = nrows

            if 'ncols' in [*kwargs]:
                ncols_ = kwargs['ncols']
            else:
                ncols_ = ncols

            if 'figsize' in [*kwargs]:
                figsize_ = kwargs['figsize']
            else:
                figsize_ = figsize

            # create or retrieve the fig, ax objects --> end up in kwargs to use into the plotting func call below
            if 'fig' in [*kwargs] and 'ax' in [*kwargs]:
                if kwargs['fig'] is None or kwargs['ax'] is None:
                    # print('..creating fig, ax [1]')
                    kwargs['fig'], kwargs['axs'] = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=figsize_)
                else: pass
            elif ncols_ > 1 or nrows_ > 1:
                kwargs['fig'], kwargs['axs'] = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=figsize_)
            else:
                kwargs['fig'], kwargs['ax'] = plt.subplots(figsize=figsize_)


            print(f'.. executing plotting_func')
            res = plotting_func(**kwargs)  # these kwargs are the original kwargs defined at the respective plotting_func call + any additional kwargs defined in inner()

            kwargs['ax'].set_title(kwargs['title'], wrap=True) if 'title' in [*kwargs] else None

            kwargs['fig'].tight_layout(pad=1.8)

            if 'show' in [*kwargs]:
                if kwargs['show'] is True:
                    kwargs['fig'].show()
                    return res
                else:
                    if res is not None:
                        return (kwargs['fig'], kwargs['ax'], res)
                    else:
                        return (kwargs['fig'], kwargs['ax'])
            else:
                kwargs['fig'].show()
                return res

        return inner
    return plotting_decorator

def save_figure(fig, save_path_full: str = None):
    print(f'\nsaving figure to: {save_path_full}')
    os.makedirs(save_path_full)
    fig.savefig(save_path_full)

# custom colorbar for heatmaps
from matplotlib.colors import LinearSegmentedColormap
def _make_colormap(seq):
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


# generate an array of random colors
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
    Generates a list of random colors for an input number of colors required.

    :param n_colors: # of colors required
    :return: list of colors in RGB
    """
    colors = []
    for i in range(0, n_colors):
        colors.append(_generate_new_color(colors, pastel_factor=0.2))
    return colors


image_frame_ops = {
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.subplot.wspace': .01,
        'figure.subplot.hspace': .01,
        'figure.figsize': (18, 13),
        'ytick.major.left': False,
        'xtick.major.bottom': False}

# Figure Style settings for notebook.
def image_frame_options(fig, ax):

    # mpl.pyplot.rcdefaults()

    # mpl.rcParams.update({
    #     'axes.spines.left': False,
    #     'axes.spines.bottom': False,
    #     'axes.spines.top': False,
    #     'axes.spines.right': False,
    #     'legend.frameon': False,
    #     'figure.subplot.wspace': .01,
    #     'figure.subplot.hspace': .01,
    #     'figure.figsize': (18, 13),
    #     'ytick.major.left': False,
    #     'xtick.major.bottom': False
    #
    # })


    #
    # mpl.rcParams.update({
    #     'axes.spines.left': False,
    #     'axes.spines.bottom': False,
    #     'axes.spines.top': False,
    #     'axes.spines.right': False,
    #     'legend.frameon': False,
    #     'figure.subplot.wspace': .01,
    #     'figure.subplot.hspace': .01,
    #     'ytick.major.left': False,
    #     'xtick.major.bottom': False
    # })

    ax.spines.left = False
    ax.spines.bottom = False
    ax.spines.top = False
    ax.spines.right = False
    fig.subplots_adjust(hspace=0.01)
    fig.subplots_adjust(wspace=0.01)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.xaxis.set_major_locator([])
    # ax.yaxis.set_major_locator([])
    # ax.xaxis.set_minor_locator([])
    # ax.yaxis.set_minor_locator([])


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


def dataplot_ax_options(ax, data_length: int, **kwargs):
    """
    :param
        **kwargs:

    """
    if ax:
        ax.margins(0.1)

        # set x_axis label
        # change x-axis to time (secs) if time is requested
        if 'x_axis' in [*kwargs]:
            x_axis = kwargs['x_axis']
            if ('time' in x_axis or 'Time' in x_axis) and 'trialobj' in [*kwargs]:
                trialobj = kwargs['trialobj']

                # change x axis ticks to every 30 seconds
                labels = list(range(0, int(data_length // trialobj.imparams.fps), 30))
                ax.set_xticks(ticks=[(label * trialobj.imparams.fps) for label in labels])

                ax.set_xticklabels(labels)
                ax.set_xlabel('Time (secs)')
            else:
                ax.set_xlabel(x_axis)

        # set y_axis label
        ax.set_ylabel(kwargs['y_axis']) if 'y_axis' in [*kwargs] else None

        # set x and y axis limits
        if 'xlims' in [*kwargs]: ax.set_xlim(kwargs['xlims'])
        if 'ylims' in [*kwargs]: ax.set_ylim(kwargs['ylims'])

    else:
        pass


def heatmap_options():
    # import matplotlib as mpl

    jet = mpl.cm.get_cmap('jet')
    jet.set_bad(color='k')

# %% GENERAL PLOTTING FUNCS
### plot the location of provided coordinates
@plotting_decorator(figsize=(5, 5))
def plot_coordinates(coords: list,  frame_x: int, frame_y: int, background: np.ndarray = None, fig=None, ax=None, **kwargs):
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

    if 'edgecolors' in [*kwargs]:
        edgecolors = kwargs['edgecolors']
    else:
        edgecolors = 'yellowgreen'
    for (x, y) in coords:
        ax.scatter(x=x, y=y, edgecolors=edgecolors, facecolors='none', linewidths=2.0)

    ax.margins(0)
    fig.tight_layout()

    if 'title' in [*kwargs]:
        if kwargs['title'] is not None:
            ax.set_title(kwargs['title'])
        else:
            pass
