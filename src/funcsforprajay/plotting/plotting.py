import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats

from funcsforprajay.funcs import flattenOnce
from funcsforprajay.wrappers import plot_piping_decorator


# plotting settings
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')


############### PLOTTING FUNCTIONS #####################################################################################

def make_random_plot():
    plt.scatter(range(100), np.random.random(100), s=50, color='green')
    plt.show()


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


# plotting function for plotting a bar graph with the individual data points shown as well
@plot_piping_decorator(verbose=False, figsize=(3, 3))
def plot_bar_with_points(data, title='', x_tick_labels=None, legend_labels: list = [], points: bool = True,
                         bar: bool = True, colors: list = ['black'], ylims=None, xlims=True, text_list=None,
                         x_label=None, y_label=None, alpha=0.2, savepath=None, fontsize: int = 8,
                         show_legend=False, paired=False, sig_compare_lines: dict = None, fig=None, ax=None,
                         lw=1, points_lw=0.5, **kwargs):
    """
    Plots Mean +/- SEM of each data group that is provided.
    all purpose function for plotting a bar graph of multiple categories with the option of individual datapoints shown
    as well. The individual datapoints are drawn by adding a scatter plot with the datapoints randomly jittered around the central
    x location of the bar graph. The individual points can also be paired in which case they will be centered. The bar can also be turned off.

    :param sig_compare_lines: e.g. of line between 0th and 1st data point with '*' --> {'*': [0,1]}
    :param data: list; provide data from each category as a list and then group all into one list
    :param title: str; title of the graph
    :param x_tick_labels: labels to use for categories on x axis
    :param legend_labels:
    :param points: bool; if True plot individual data points for each category in data using scatter function
    :param bar: bool, if True plot the bar, if False plot only the mean line
    :param colors: line_colors (by category) to use for each x group
    :param ylims: tuple; y axis limits
    :param xlims: the x axis is used to position the bars, so use this to move the position of the bars left and right
    :param x_label: x axis label
    :param y_label: y axis label
    :param text_list: list of text to add to each category of data on the plot
    :param text_shift: float; number between 0.5 to 1 used to adjust precise positioning of the text in text_list
    :param alpha: transparency of the individual points when plotted in the scatter
    :param lw: linewidth of bar edges and errorbars
    :param points_lw: linewidth of edges of points
    :param savepath: .svg file path; if given, the plot will be saved to the provided file path
    :param expand_size_x: factor to use for expanding figure size
    :param expand_size_y: factor to use for expanding figure size
    :param paired: bool, if True then draw lines between data points of the same index location in each respective list in the data
    **kwargs:
        :param fontsize: fontsize of text in plot
        :param y_labelpad: labelpad of the y-axis label
        :param x_labelpad: labelpad of the x-axis label
        :param capsize: size of the errorbar cap
    :return: matplotlib plot
    """

    for key in kwargs:
        if 'y' in key and 'lim' in key:
            ylims = kwargs[key]
        if 'x' in key and 'lim' in key:
            xlims = kwargs[key]
        if 'fs' == key:
            fontsize = kwargs[key]


    # collect some info about data to plot
    w = 1.0  # mean bar width
    # xrange_ls = list(range(len(data)))
    xrange_ls = [x + 1.6 for x in range(len(data))]
    y = data
    if len(colors) != len(xrange_ls):
        colors = colors * len(xrange_ls)

    s = 10 if 's' not in kwargs else kwargs['s']

    bar_alpha = 1 if not points else 0.4
    bar_alpha = kwargs['bar_alpha'] if 'bar_alpha' in kwargs else bar_alpha

    ### ADD BAR AND ERROR BARS TO THE PLOT
    if bar:
        edgecolor = 'black' if 'edgecolor' not in kwargs else kwargs['edgecolor']
        capsize = 5 if 'capsize' not in kwargs else kwargs['capsize']
        # plot bar graph
        ax.errorbar([x * w * 2.3 for x in xrange_ls], [np.mean(yi) for yi in y], fmt='none', zorder=10,
                    yerr=np.asarray([np.asarray([stats.sem(yi, ddof=1), stats.sem(yi, ddof=1)]) for yi in y]).T, ecolor='black',
                    capsize=capsize, elinewidth=lw*1.3, markeredgewidth=lw*1.3, alpha=bar_alpha)
        ax.bar([x * w * 2.3 for x in xrange_ls],
               height=[np.mean(yi) for yi in y],
               # yerr=np.asarray([np.asarray([0, np.std(yi, ddof=1)]) for yi in y]).T,  # error bars
               capsize=4.5,  # error bar cap width in points
               width=1.9,  # bar width
               linewidth=lw*1.5,  # line thickness of the bar edges
               edgecolor=edgecolor,
               # color=(0, 0, 0, 0),  # facecolor transparent
               color=colors,  # facecolor transparent
               alpha=bar_alpha,
               zorder=0)
        if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
            ax.axhline(0, color='black', lw=1, zorder=0)

    else:
        for idx, x_val in enumerate(xrange_ls):
            ## plot the mean line
            ax.plot(np.linspace(x_val * w * 2.3 - w / 2, x_val * w * 2.3 + w / 2, 3), [np.mean(y[idx])] * 3, color='black', zorder=10, lw=lw)
        capsize = 5 if 'capsize' not in kwargs else kwargs['capsize']
        # since no bar being shown on plot (lw = 0 from above) then use it to plot the error bars
        ax.errorbar([x * w * 2.3 for x in xrange_ls], [np.mean(yi) for yi in y], fmt='none',
                    yerr=np.asarray([np.asarray([stats.sem(yi, ddof=1), stats.sem(yi, ddof=1)]) for yi in y]).T,
                    ecolor='black', capsize=capsize, zorder=10, elinewidth=lw, markeredgewidth=lw)
        points_lw = 0 if alpha < 1 else points_lw

    ax.set_xticks([x * w * 2.3 for x in xrange_ls])
    # x_tick_labels = [round(x * w * 2.3, 2) for x in xrange_ls]  # use for debugging placement of plotted data
    x_tick_labels = (ax.get_xticks() if legend_labels is None else legend_labels) if x_tick_labels is None else x_tick_labels
    assert len(xrange_ls) == len(x_tick_labels), f'not enough x_tick_labels provided. {x_tick_labels}, need {len(xrange_ls)}'
    if len(xrange_ls) > 1:
        ax.set_xticklabels(x_tick_labels, fontsize=fontsize, rotation=45)
    else:
        ax.set_xticklabels(x_tick_labels, fontsize=fontsize)

    if sig_compare_lines:
        if not points: print('CANNOT MAKE SIG. LINES, NEEDS POINTS.')
        for i, sig_compare_line in sig_compare_lines.items():
            groups = ax.get_xticks()
            assert len(sig_compare_line) > 1
            top_line = [np.max(flattenOnce([data[sig_compare_line[0]], data[sig_compare_line[1]]])) * 1.5] * 2
            ax.plot([groups[sig_compare_line[0]], groups[sig_compare_line[1]]], top_line, color='black', lw=lw)

            gaps_ = [np.max(data[sig_compare_line[0]]) * 1.25 - np.max(data[sig_compare_line[0]])]
            gaps_.append(np.max(data[sig_compare_line[1]]) * 1.25 - np.max(data[sig_compare_line[1]]))
            gap = np.max(gaps_)

            ax.plot([groups[sig_compare_line[0]]] * 2, [np.max(data[sig_compare_line[0]]) + gap, np.max(
                flattenOnce([data[sig_compare_line[0]], data[sig_compare_line[1]]])) * 1.5], color='black', lw=lw)
            ax.plot([groups[sig_compare_line[1]]] * 2, [np.max(data[sig_compare_line[1]]) + gap, np.max(
                flattenOnce([data[sig_compare_line[0]], data[sig_compare_line[1]]])) * 1.5], color='black', lw=lw)

            xy = (np.mean([groups[sig_compare_line[0]], groups[sig_compare_line[1]]]), top_line[0] + 0.25 / fig.dpi)

            ax.text(x=xy[0], y=xy[1], s=i, fontsize=fontsize*1.5, ha='center')

    if xlims:
        ax.set_xlim([(xrange_ls[0] * w * 2) - w * 1.20, (xrange_ls[-1] * w * 2.4) + w * 1.20])
    elif len(xrange_ls) == 1:  # set the x_lims for single bar case so that the bar isn't autoscaled
        xlims_ = [-1.5, 1.5]
        ax.set_xlim(xlims_)

    if len(legend_labels) == 0:
        if len(x_tick_labels) == 0:
            x_tick_labels = [None] * len(xrange_ls)
        legend_labels = x_tick_labels

    ### PLOT DATA POINTS
    if points:
        # points_lw = 0 if alpha < 1 else points_lw
        if not paired:
            for i, _ in enumerate(xrange_ls):
                # distribute scatter randomly across whole width of bar
                ax.scatter(xrange_ls[i] * w * 2.3 + np.random.random(len(y[i])) * w * 1.4 - w / 1.4, y[i],
                           facecolor=colors[i], edgecolor='black', lw=points_lw,
                           alpha=alpha, label=legend_labels[i], zorder=2, s=s)

        else:  # connect lines to the paired scatter points in the list
            assert len(xrange_ls) > 1
            if len(xrange_ls) > 0:
                for i, _ in enumerate(xrange_ls):
                    # plot points  # dont scatter location of points if plotting paired lines
                    ax.scatter([xrange_ls[i] * w * 2.3] * len(y[i]), y[i], color=colors[i], alpha=alpha,
                               label=legend_labels[i], zorder=3, edgecolor='black', lw=points_lw, s=s)
                for i, _ in enumerate(xrange_ls[:-1]):
                    for point_idx in range(len(y[i])):  # draw the lines connecting pairs of data
                        ax.plot([xrange_ls[i] * w * 2.3 + 0.058, xrange_ls[i + 1] * w * 2.3 - 0.048],
                                [y[i][point_idx], y[i + 1][point_idx]], color='black', zorder=1, alpha=0.3)

            else:
                AttributeError('cannot do paired scatter plotting with only one data category')

    if ylims:
        ax.set_ylim(ylims)
    elif len(xrange_ls) == 1:  # set the y_lims for single bar case so that the bar isn't autoscaled
        ylims = [0, 2 * max(data[0])]
        ax.set_ylim(ylims)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=5, labelsize=fontsize)

    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')

    # set yticks and yticklabels
    y_ticklabels_ = [] if 'y_ticklabels' not in kwargs else kwargs['y_ticklabels']
    ax.set_yticks([float(x) for x in y_ticklabels_]) if len(y_ticklabels_) > 0 else None
    ax.set_yticklabels([str(x) for x in y_ticklabels_]) if len(y_ticklabels_) > 0 else None

    ### set axis labels:
    ax.set_xlabel(x_label, fontsize=fontsize, labelpad=kwargs['x_labelpad']) if 'x_labelpad' in kwargs else ax.set_xlabel(x_label, fontsize=fontsize, labelpad=4.0)
    ax.set_ylabel(y_label, fontsize=fontsize, labelpad=kwargs['y_labelpad']) if 'y_labelpad' in kwargs else ax.set_ylabel(y_label, fontsize=fontsize, labelpad=4.0)  # note matplotlib default labelpad is 4.0

    # add text to the figure if given:
    if text_list:
        assert len(xrange_ls) == len(text_list), 'please provide text_list of same len() as data'
        if 'text_shift' in kwargs.keys():
            text_shift = kwargs['text_shift']
        else:
            text_shift = 0.8
        if 'text_y_pos' in kwargs.keys():
            text_y_pos = kwargs['text_y_pos']
        else:
            text_y_pos = max([np.percentile(y[i], 95) for i in xrange_ls])
        for i in xrange_ls:
            ax.text(xrange_ls[i] * w * 2.5 - text_shift * w / 2, text_y_pos, text_list[i]),

    if len(legend_labels) > 1:
        if show_legend:
            ax.legend(bbox_to_anchor=(1.01, 0.90), fontsize=fontsize)

    if title: ax.set_title(title, fontsize=fontsize)

    if savepath:
        plt.savefig(savepath)




@plot_piping_decorator(verbose=False)
def make_general_scatter(x_list: list, y_data: list, fig=None, ax=None,
                         **kwargs):  ## TODO remove the double plotting, just give option to plot all individual as subplots or together!
    """
    General function for quick, simple plotting of data lists as scatters. NOTE: THIS FUNC MAKES TWO SEPARATE PLOTS if given >1 dataset to plot.

    :param x_list: list of x_points for plots, must match one to one to y_data
    :param y_data: list of y_data for plots, must match one to one to x_list
    :param kwargs: (optional)
        s: int, size of points
        alpha: transparency
        facecolors: list, line_colors to use to plot >1 data sets
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

    # ## func arguments
    # kwargs = {}
    #
    # y_data = [[]]
    # x_list = [[]]
    # line_colors = [[]]

    ##
    assert type(x_list) is list
    assert type(y_data) is list
    assert len(y_data) == len(x_list), 'y_data length does not match x_list length'

    num_plots = len(x_list)

    if 'facecolors' not in kwargs:
        colors = make_random_color_array(num_plots)
    else:
        assert type(kwargs['facecolors']) is list and len(kwargs['facecolors']) == len(
            x_list), 'provide line_colors argument in list form matching number of traces to plot'
        colors = kwargs['facecolors']

    edgecolors = colors if 'edgecolors' not in [*kwargs] else kwargs['edgecolors']

    # set plotting properties
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    else:
        alpha = 0.8
    if 's' in kwargs:
        size = kwargs['s']
    else:
        size = 50
    lw = None if 'lw' not in [*kwargs] else kwargs['lw']

    # check integrity of function call arguments
    if 'y_labels' in kwargs and type(kwargs['y_labels']) is list: assert len(kwargs['y_labels']) == num_plots
    if 'x_labels' in kwargs and type(kwargs['x_labels']) is list: assert len(kwargs['x_labels']) == num_plots
    if 'ax_titles' in kwargs and type(kwargs['ax_titles']) is list: assert len(kwargs['ax_titles']) == num_plots

    if 'legend_labels' in kwargs and type(kwargs['legend_labels']) is list:
        assert len(kwargs[
                       'legend_labels']) == num_plots, 'legend_labels len does not match number of plots to make (len of x_list)'
        label = kwargs['legend_labels']
    else:
        label = ['' for i in range(num_plots)]

    if num_plots > 1:
        ncols = 4
        nrows = len(x_list) // ncols
        if len(x_list) % ncols > 0:
            nrows += 1

        fig2, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[(ncols * 4), (nrows * 3)])
        counter = 0

        axs[0, 0].set_xlabel(kwargs['x_labels'][0]) if 'x_labels' in kwargs.keys() else None
        axs[0, 0].set_ylabel(kwargs['y_labels'][0]) if 'y_labels' in kwargs.keys() else None

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

    ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1]) if 'xlim' in kwargs.keys() else None
    ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1]) if 'ylim' in kwargs.keys() else None
    ax.set_xlabel(kwargs['x_labels'][0]) if 'x_labels' in kwargs.keys() else None
    ax.set_ylabel(kwargs['y_labels'][0]) if 'y_labels' in kwargs.keys() else None

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
@plot_piping_decorator(verbose=False)
def make_general_plot(data_arr, x_range=None, twin_x: bool = False, plot_avg: bool = False, plot_std: bool = False,
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
        grid: bool, show major and minor grid lines of the plot
    :return None
    """

    f, axs = kwargs['fig'], [kwargs['ax']]
    # prepare for plotting over multiple axes if called for
    if type(kwargs['ax']):
        axs = kwargs['ax']
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
    if 'alpha' in [*kwargs]:
        alpha = kwargs['alpha']
    elif num_traces > num_axes and num_axes == 1:
        alpha = 0.3
    else:
        alpha = 1
        plot_avg = False  # turn off plotting of average trace
        plot_std = False  # turn off plotting of std trace from data

    # add twin x if called for:
    if num_traces == 2 and twin_x is True:
        ax = axs[0]
        ax2 = ax.twinx()
        axs = np.array([ax, ax2], dtype=object)
        num_axes = 2
    if num_traces > 1 and twin_x is False:
        if num_axes != num_traces:
            raise ValueError(f"need to provide enough axis objects to fit graphs.")

    print(f'\nPlotting {num_traces} data traces across {num_axes} axes') if not twin_x else print(
        f'\nPlotting {num_traces} data traces across 1 axes (with twin_x)')

    # create x_range to use for plotting
    if x_range is not None:
        if type(x_range) is list:
            x_range = np.asarray(x_range)
        assert x_range.shape == data_arr.shape, '|- AssertionError: mismatch between data to plot and x_range provided for this data'
    else:
        x_range = []
        for i in range(num_traces):
            x_range.append(range(len(data_arr[i])))

    # make random line_colors for plotting
    if 'line_colors' not in kwargs.keys():
        colors = make_random_color_array(num_traces) if num_traces > 1 else ['black']
    else:
        assert type(kwargs['line_colors']) is list, '|- AssertionError: provide line_colors argument in list form'
        assert len(kwargs[
                       'line_colors']) == num_traces, '|- AssertionError: provide enough line_colors as number of traces to plot'
        colors = kwargs['line_colors']

    # check integrity of function call arguments
    if 'y_labels' in [*kwargs] and len(kwargs['y_labels']) > 1: assert len(kwargs['y_labels']) == num_traces
    if 'x_labels' in [*kwargs] and len(kwargs['x_labels']) > 1: assert len(kwargs['x_labels']) == num_traces
    if 'ax_titles' in [*kwargs] and not twin_x: assert len(kwargs['ax_titles']) == num_traces

    # shrink or enlarge the fontsize option:
    fontsize = kwargs['fontsize'] if 'fontsize' in [*kwargs] else 10
    lw = kwargs['lw'] if 'lw' in [*kwargs] else 1

    # make the plot using each provided data trace
    ax_counter = 0

    if 'v_span' in kwargs.keys() and type(kwargs['v_span']) is tuple:
        axs[ax_counter].axvspan(kwargs['v_span'][0], kwargs['v_span'][1], color='indianred', zorder=1)


    if plot_std is False or num_traces == 1:  # only plot individual lines if plot_std is inactive
        print(f'.. plotting {num_traces} individual traces on {num_axes} axes')
        for i in range(num_traces):
            print(f"\tplotting: {kwargs['ax_titles'][i]}")
            ### show grid lines
            if 'grid' in kwargs and kwargs['grid']:
                axs[ax_counter].grid(visible=True, which='both')

            # make plot
            axs[ax_counter].plot(x_range[i], data_arr[i], color=colors[i], alpha=alpha, linewidth=lw)
            axs[ax_counter].set_title(kwargs['ax_titles'][i], fontsize=fontsize) if 'ax_titles' in [*kwargs] else None
            if not twin_x:
                axs[ax_counter].set_title(kwargs['ax_titles'][i], fontsize=fontsize) if 'ax_titles' in [
                    *kwargs] else None
                axs[ax_counter].set_xlabel(kwargs['x_labels'][i], fontsize=fontsize) if 'x_labels' in [
                    *kwargs] else None
                axs[ax_counter].set_ylabel(kwargs['y_labels'][i], fontsize=fontsize) if 'y_labels' in [
                    *kwargs] else None
                ax_counter += 1
            else:
                axs[0].set_title(kwargs['ax_titles'][0], fontsize=fontsize) if 'ax_titles' in [*kwargs] else None
                axs[0].set_xlabel(kwargs['x_labels'][0], fontsize=fontsize) if 'x_labels' in [*kwargs] else None
                axs[i].set_ylabel(kwargs['y_labels'][i], fontsize=fontsize, color=colors[i]) if 'y_labels' in [
                    *kwargs] else None
                ax_counter += 1

    elif num_axes == 1 and twin_x is False and num_traces > 1:
        ### show grid lines
        if 'grid' in kwargs and kwargs['grid']:
            axs[ax_counter].grid(visible=True, which='both')

        if plot_avg:
            print(f'.. plotting average trace of {data_arr.shape[0]} traces on 1 axis')
            axs[ax_counter].plot(x_range[0], np.mean(data_arr, axis=0), color='black', alpha=1,
                                 zorder=data_arr.shape[0] + 1, lw=lw)
            axs[ax_counter].set_title(kwargs['ax_titles'][ax_counter], fontsize=fontsize) if 'ax_titles' in [
                *kwargs] else None

        if plot_std:
            print(f'.. plotting std trace of {data_arr.shape[0]} traces on 1 axis')
            std_low = np.mean(data_arr, axis=0) - np.std(data_arr, axis=0)
            std_high = np.mean(data_arr, axis=0) + np.std(data_arr, axis=0)
            axs[ax_counter].fill_between(x_range[0], std_low, std_high, color='gray', alpha=0.5, zorder=0)

        axs[ax_counter].set_title(kwargs['ax_titles'][ax_counter], fontsize=fontsize * 1.1,
                                  wrap=True) if 'ax_titles' in [*kwargs] else axs[ax_counter].set_title(
            f"{num_traces} traces")
        axs[ax_counter].set_ylabel(kwargs['y_label'][ax_counter], fontsize=fontsize) if 'y_label' in [*kwargs] else None
        axs[ax_counter].set_xlabel(kwargs['x_label'][ax_counter], fontsize=fontsize) if 'x_label' in [*kwargs] else None
        axs[ax_counter].set_ylabel(kwargs['y_labels'][ax_counter], fontsize=fontsize) if 'y_labels' in [
            *kwargs] else None
        axs[ax_counter].set_xlabel(kwargs['x_labels'][ax_counter], fontsize=fontsize) if 'x_labels' in [
            *kwargs] else None

    # elif num_axes == 1 and twin_x is True and num_traces > 1:
    #     for

    # change x axis units if x axis units specified
    if 'x_axis_ratio' in [*kwargs]:
        if num_axes == 1 or twin_x:
            x_ticks = axs[0].get_xticks()
            if 'x_tick_labels' in [*kwargs]:
                new_ticks_labels = kwargs['x_tick_labels']
                x_ticks = [float(i) * kwargs['x_axis_ratio'] for i in new_ticks_labels]
            else:
                # new_ticks = list(range(int(x_ticks[0] // kwargs['x_axis_ratio']), int(x_ticks[-1] // kwargs['x_axis_ratio']), (int(x_ticks[-1] // kwargs['x_axis_ratio']) - int(x_ticks[0]))//len(x_ticks)))
                new_ticks_labels = [tick // kwargs['x_axis_ratio'] for tick in x_ticks]
            axs[0].set_xticks(x_ticks)
            axs[0].set_xticklabels(new_ticks_labels)
        elif 'same_x' in [*kwargs] and kwargs['same_x']:
            for ax in axs:
                x_ticks = ax.get_xticks()
                if 'x_tick_labels' in [*kwargs]:
                    new_ticks_labels = kwargs['x_tick_labels']
                    x_ticks = [float(i) * kwargs['x_axis_ratio'] for i in new_ticks_labels]
                else:
                    # new_ticks = list(range(int(x_ticks[0] // kwargs['x_axis_ratio']), int(x_ticks[-1] // kwargs['x_axis_ratio']), (int(x_ticks[-1] // kwargs['x_axis_ratio']) - int(x_ticks[0]))//len(x_ticks)))
                    new_ticks_labels = [tick // kwargs['x_axis_ratio'] for tick in x_ticks]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(new_ticks_labels)


    # set x_lim if requested
    if 'xlim' in kwargs:
        if num_axes == 1 or twin_x or ('same_x' in kwargs and kwargs['same_x']):
            if 'x_axis_ratio' in kwargs:
                ratio = kwargs['x_axis_ratio']
            else:
                ratio = 1
            xlim_low = kwargs['xlim'][0] * ratio
            xlim_high = kwargs['xlim'][1] * ratio
            for ax in axs:
                ax.set_xlim(xlim_low, xlim_high)


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

    ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1]) if 'ylim' in kwargs.keys() else None
    ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1]) if 'xlim' in kwargs.keys() else None

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


# histogram density plot with gaussian best fit line
@plot_piping_decorator(figsize=(5, 5), verbose=False)
def plot_hist_density(data: list, mean_line: bool = False, colors: list = None, fill_color: list = None,
                      legend_labels: list = [None], num_bins=11, best_fit_line='gaussian', density=True, **kwargs):
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

    if not type(data[0]) == list: raise ValueError('need to provide `data` as nested list of list(s) for plotting.')

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
    if 'fill_alpha' in kwargs and (type(kwargs['fill_alpha']) is float or kwargs['fill_alpha'] == 1):
        alpha1 = kwargs['fill_alpha']
    else: alpha1 = 0.3

    if 'bar_alpha' in kwargs and (type(kwargs['bar_alpha']) is float or kwargs['bar_alpha'] == 1):
        alpha2 = kwargs['bar_alpha']
    else: alpha2 = 0.4

    lw = kwargs['lw'] if 'lw' in kwargs else 2
    # make the primary histogram density plot
    zorder = 2
    for i in range(len(data)):
        # the histogram of the data
        bin_heights, bins, patches = ax.hist(data[i], num_bins, density=density, alpha=alpha2, color=fill_color[i],
                                             label=legend_labels[i])  # histogram hidden currently

        # add a 'best fit' line
        if best_fit_line == 'powerlaw':
            from scipy.optimize import curve_fit

            def func_powerlaw(x, m, c, c0):
                return c0 + x ** m * c

            target_func = func_powerlaw

            X = np.linspace(bins[0], bins[-1], num_bins)
            y = bin_heights
            popt, pcov = curve_fit(target_func, X, y, maxfev=1000000)

            ax.plot(X, target_func(X, *popt), linewidth=2, c=colors[i], zorder=zorder + i)
            ax.fill_between(X, target_func(X, *popt), color=fill_color[i], zorder=zorder + i, alpha=alpha1)
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

            ax.plot(x, popt, linewidth=lw, c=colors[i], zorder=zorder + i)
            ax.fill_between(x, popt, color=fill_color[i], zorder=zorder + i, alpha=alpha1)

            title = (r': $\mu=%s$, $\sigma=%s$' % (round(mu, 2), round(sigma, 2)))
        else:
            title = ''

        if mean_line:
            ax.axvline(x=np.nanmean(data[i]), c=fill_color[i], linewidth=1.5, zorder=0, linestyle='dashed')

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
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])

    # setting shrinking factor for font size for title
    fontsize = kwargs['fontsize'] if 'fontsize' in kwargs else 10

    # add title
    if 'title' in kwargs and kwargs['title'] is not None:
        if len(data) == 1:
            ax.set_title(kwargs['title'] + title, wrap=True,
                         fontsize=fontsize)
        else:
            ax.set_title(kwargs['title'], wrap=True, fontsize=fontsize)
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
    import tifffile as tf
    stack = tf.imread(tiff_path, key=frame_num)
    plt.imshow(stack, cmap='gray')
    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle('frame num: %s' % frame_num)
    plt.show()
    return stack

# plot an image numpy array in grayscale
@plot_piping_decorator(figsize=(5, 5), verbose=False)
def plotImg(img: np.ndarray, **kwargs):
    """Plot image in grayscale.

    :param img: input image to show
    :param kwargs:
        :trialobj: ImagingTrial or SingleImage; object associated with input image.
        :scalebar_um: int; size of scalebar to plot on image (in um); must provide trialobj parameter.
    """
    assert img.ndim == 2, 'img to plot must only have 2 dimensions.'
    fig, ax = kwargs['fig'], kwargs['ax']
    ax.imshow(img, cmap='gray')

if __name__ == '__main__':
    data = [np.random.random(20) for i in range(3)]
    plot_bar_with_points(data=data, bar=False, x_tick_labels=['baseline', 'interictal', 'ictal'],
            colors=['blue', 'green', 'purple'], lw=1, alpha=1, shrink_text=1, points=True,
            title='Average s2p ROIs spk rate', y_label='spikes rate (Hz)', figsize=(2, 2))

