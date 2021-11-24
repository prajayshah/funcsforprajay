# this script contains wrapper that are to be used elsewhere`

import functools
import matplotlib.pyplot as plt


# PLOTTING WRAPPERS
# works
def print_start_end_plot(plotting_func):
    """wrapper to print start and end of the plotting func call, use at the top of nested decorators"""
    def inner(*args, **kwargs):
        print(f"\n {'.' * 5} plotting function \ start \n")
        res = plotting_func(*args, **kwargs)
        print(f"** res during print_start_end_plot {res}")
        print(f"\n {'.' * 5} plotting function \ end \n")
        return res
    return inner


# works
def plot_piping_decorator(plotting_func):
    """
    Wrapper to help simplify creating plots from matplotlib.pyplot

    :param plotting_func: function to be wrapper
    :return: fig+ax objects or shows the figure as specified

    Examples:
    ---------
    @plot_piping_decorator
    def example_decorated_plot(title='', **kwargs):
        fig, ax = kwargs['fig'], kwargs['ax']  # need to add atleast this line to each plotting function's definition
        print(f'|-kwargs inside example_decorated_plot definition: {kwargs}')
        ax.plot(np.random.rand(10))
        ax.set_title(title)

    def example_decorated_plot(fig=None, ax=None, title='', **kwargs):
        # in this example the fig and ax will be taken directly from the kwargs inside the inner wrapper
        print(f'|-kwargs inside example_decorated_plot definition: {kwargs}')
        ax.plot(np.random.rand(10))
        ax.set_title(title)


    """

    @functools.wraps(plotting_func)
    def inner(**kwargs):
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
            figsize = kwargs['figsize']
        else:
            figsize = (5, 5)

        # create or retrieve the fig, ax objects --> end up in kwargs to use into the plotting func call below
        if 'fig' in kwargs.keys() and 'ax' in kwargs.keys():
            if kwargs['fig'] is None or kwargs['ax'] is None:
                print('\-creating fig, ax [1]')
                kwargs['fig'], kwargs['ax'] = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            else:
                return_fig_obj = True
        else:
            print('\-creating fig, ax [2]')
            kwargs['fig'], kwargs['ax'] = plt.subplots(figsize=figsize)

        print(f"\nnew kwargs {kwargs}")

        print(f'\nexecute plotting_func')
        plotting_func(
            **kwargs)  # these kwargs are the original kwargs defined at the respective plotting_func call + any additional kwargs defined in inner()

        print(f'\nreturn fig, ax or show figure as called for')
        kwargs['fig'].suptitle(kwargs['suptitle'], wrap=True) if 'suptitle' in kwargs.keys() else None
        if 'show' in kwargs.keys():
            if kwargs['show'] is True:
                print(f'\-showing fig...[3]')
                kwargs['fig'].show()
            else:
                print(f"\-not showing, but returning fig_obj [4]")
                return (kwargs['fig'], kwargs['ax'])
        else:
            kwargs['fig'].show()

        print(f"|-value of return_fig_obj is {return_fig_obj} [5]")
        return (kwargs['fig'], kwargs['ax']) if return_fig_obj else None

    return inner
