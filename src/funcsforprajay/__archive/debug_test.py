# very temp script for testing and debugging various bits of code

import matplotlib.pyplot as plt
import numpy as np
from funcsforprajay.funcs import make_random_color_array #, make_general_plot
import functools

# from funcsforprajay.wrappers import plot_piping_decorator, print_start_end_plot

# %%
def print_start_end_plot(plotting_func):
    @functools.wraps(plotting_func)
    def inner(*args, **kwargs):
        print(f"\n {'.' * 5} plotting function \ start \n")
        print(f"** args during print_start_end_plot {args}")
        print(f"** kwargs during print_start_end_plot {kwargs}\n")
        res = plotting_func(*args, **kwargs)
        print(f"** res during print_start_end_plot {res}")
        print(f"\n {'.' * 5} plotting function \ end \n")
        return res
    return inner


def plot_piping_decorator(figsize=(5,5)):
    def plot_piping_decorator_(plotting_func):
        @functools.wraps(plotting_func)
        def inner(*args, **kwargs):
            print(f'perform fig, ax creation')
            print(f'|-original kwargs {kwargs}')
            return_fig_obj = False

            # set number of rows, cols and figsize
            if 'nrows' in kwargs.keys():
                nrows = kwargs['nrows']
            else:
                nrows = 1

            if 'ncols' in kwargs.keys():
                ncols = kwargs['ncols']
            else:
                ncols = 1

            if 'figsize' in kwargs.keys():
                figsize_ = kwargs['figsize']
            else:
                figsize_ = figsize

            # create or retrieve the fig, ax objects --> end up in kwargs to use into the plotting func call below
            if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
                if kwargs['fig'] is None or kwargs['ax'] is None:
                    print('\-creating fig, ax [1]')
                    kwargs['fig'], kwargs['ax'] = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize_)
                else:
                    return_fig_obj = True
            else:
                print('\-creating fig, ax [2]')
                kwargs['fig'], kwargs['ax'] = plt.subplots(figsize=figsize)


            print(f"\nnew kwargs {kwargs}")

            print(f'\nexecute plotting_func')
            # var = plotting_func(**kwargs)   # these kwargs are the original kwargs defined at the respective plotting_func call + any additional kwargs defined in inner()
            res = plotting_func(*args, **kwargs)   # these kwargs are the original kwargs defined at the respective plotting_func call + any additional kwargs defined in inner()

            # print(f"\n*returned from plotting func {var}")

            print(f'\nreturn fig, ax or show figure as called for')
            kwargs['fig'].suptitle('this title was decorated')
            if 'show' in kwargs.keys():
                if kwargs['show'] is True:
                    print(f'\-showing fig...[3]')
                    kwargs['fig'].show()
                else:
                    print(f"\-not showing, but returning fig_obj [4]")
                    return kwargs['fig'], kwargs['ax'], res
                    # var = [kwargs['fig'], kwargs['ax']]
                    # return var

            else:
                kwargs['fig'].show()

            print(f"|-value of return_fig_obj is {return_fig_obj} [5]")
            if return_fig_obj:
                # var = zip(kwargs['fig'], kwargs['ax'], var)
                return (kwargs['fig'], kwargs['ax'], res)
            # else:
            #     print(f"\n*returning from plotting func {var}")
            #     return var

        return inner
    return plot_piping_decorator_

@print_start_end_plot
@plot_piping_decorator(figsize=(3,5))
def make_general_plot(data_arr, x_range=None, twin_x: bool = False, plot_avg: bool = True, plot_std: bool = True,
                      fig=None, ax=None, **kwargs):
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
        v_span: tuple, vertical span fill - will be same for each axis
        suptitle: str, used for suptitle of fig
    """

    f, axs = fig, ax
    # prepare for plotting over multiple axes if called for
    if type(axs) is np.array:
        num_axes = len(axs)
    else:
        num_axes = 1
        axs = np.array([axs])

    # create data arrays in the correct format for plotting
    if type(data_arr) is list:
        num_traces = len(data_arr)
    if type(data_arr) is np.ndarray:
        num_traces = data_arr.shape[0]

    # check if plotting multi-traces on 1 axis (but not twinx style!):
    if num_traces > num_axes and num_axes == 1:
        alpha = 0.3
    elif num_axes > 1:
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
        colors = make_random_color_array(num_traces)
    else:
        assert type(kwargs['line_colors']) is list, print('|- AssertionError: provide line_colors argument in list form')
        assert len(kwargs['line_colors']) == num_traces, print(
            '|- AssertionError: provide enough line_colors as number of traces to plot')
        colors = kwargs['line_colors']

    # check integrity of function call arguments
    if 'y_labels' in kwargs.keys():
        assert len(kwargs['y_labels']) == num_traces
    if 'x_labels' in kwargs.keys():
        assert len(kwargs['x_labels']) == num_traces
    if 'ax_titles' in kwargs.keys():
        assert len(kwargs['ax_titles']) == num_traces

    # make the plot using each provided data trace
    ax_counter = 0

    if 'v_span' in kwargs.keys() and type(kwargs['v_span']) is tuple:
        axs[ax_counter].axvspan(kwargs['v_span'][0], kwargs['v_span'][1], color='indianred', zorder=1)

    if plot_std is False:  # only plot individual lines if plot_std is inactive
        print(f'\- plotting {num_traces} individual traces on {num_axes} axes')
        for i in range(num_traces):
            axs[ax_counter].plot(x_range[i], data_arr[i], color=colors[i], alpha=alpha)
            axs[ax_counter].set_ylabel(kwargs['y_labels'][i]) if 'y_labels' in kwargs.keys() else None
            axs[ax_counter].set_xlabel(kwargs['x_labels'][i]) if 'x_labels' in kwargs.keys() else None
            if num_axes > 1:
                ax_counter += 1
    if num_axes == 1 and twin_x is False:
        if plot_avg:
            print(f'\- plotting average trace of {data_arr.shape[0]} traces on 1 axis')
            axs[ax_counter].plot(x_range[0], np.mean(data_arr, axis=0), color='black', alpha=1,
                                 zorder=data_arr.shape[0] + 1)
        if plot_std:
            print(f'\- plotting std trace of {data_arr.shape[0]} traces on 1 axis')
            std_low = np.mean(data_arr, axis=0) - np.std(data_arr, axis=0)
            std_high = np.mean(data_arr, axis=0) + np.std(data_arr, axis=0)
            axs[ax_counter].fill_between(x_range[0], std_low, std_high, color='gray', alpha=0.5, zorder=0)
        axs[ax_counter].set_title(f"{num_traces} traces")

    return data_arr[0]
    # f.suptitle(kwargs['suptitle'], wrap=True) if 'suptitle' in kwargs.keys() else None
    # f.show()


data_arr = np.asarray([np.random.rand(11), np.random.rand(6)[::2]])
t = make_general_plot(data_arr=data_arr, xrange=[range(11), range(6)[::2]], colors=['green', 'blue'], plot_std=False,
                      plot_avg=False, suptitle='a new title', show=True)

# fig, ax = plt.subplots(figsize=(3,3))
# t = make_general_plot(data_arr=data_arr, xrange=[range(11), range(6)[::2]], line_colors=['green', 'blue'], plot_std=False,
#                       plot_avg=False, suptitle='a new title', fig=fig, ax=ax, show=True)



