# this script contains wrapper that are to be used elsewhere

import functools
import matplotlib.pyplot as plt

## works
def plot_piping_decorator(plotting_func):
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
        plotting_func(**kwargs)   # these kwargs are the original kwargs defined at the respective plotting_func call + any additional kwargs defined in inner()

        print(f'\nreturn fig, ax or show figure as called for')
        kwargs['fig'].suptitle('this title was decorated')
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