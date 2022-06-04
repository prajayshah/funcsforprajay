import matplotlib.pyplot as plt
import numpy as np


def test_plot_hist_density():
    data = np.random.power(1/5, 1000)
    from funcsforprajay.funcs import plot_hist_density
    fig, ax = plt.subplots(figsize=(3,3))
    plot_hist_density(data=[data], fill_color=['blue'], alpha=0.0, num_bins=50, best_fit_line='powerlaw',
                      colors=['black'], x_label='firing rate (Hz)', fig=fig, ax=ax, show=False,
                      title='Inverse Power Law distribution')
    fig.show()

def test_plot_bar_with_points():
    data = [np.random.random(100) for i in range(3)]
    from funcsforprajay.plotting.plotting import plot_bar_with_points
    plot_bar_with_points(data=data, bar=True, x_tick_labels=['baseline', 'interictal', 'ictal'],
            colors=['blue', 'green', 'purple'], lw=1, alpha=0.6, shrink_text=1, points=False,
            title='Average s2p ROIs spk rate', y_label='spikes rate (Hz)')

def test_plot_bar_with_points_single_set():
    data = [np.random.random(50) for i in range(1)]
    from funcsforprajay.plotting.plotting import plot_bar_with_points
    plot_bar_with_points(data=data, bar=False, x_tick_labels=['baseline'],
            colors=['blue'], lw=1.3, expand_size_x = 2,
            title='Average ', y_label='rate (Hz)')


def test_plot_bar_with_points_paired_set():
    data = [np.random.random(50) + i / 10 for i in range(2)]
    from funcsforprajay.plotting.plotting import plot_bar_with_points
    plot_bar_with_points(data=data, bar=False, x_tick_labels=['baseline'], paired=True,
            colors=['blue', 'green'], lw=1.3,
            title='Average', y_label='rate (Hz)')

def test_make_general_scatter():
    from funcsforprajay.plotting.plotting import make_general_scatter
    from funcsforprajay.funcs import flattenOnce
    import numpy as np
    make_general_scatter([flattenOnce([np.random.random(100) for i in range(3)])],
                               [flattenOnce([np.random.random(100) for i in range(3)])],
                               x_labels=['Avg. Photostim Response (% dFF)'], y_labels=['Variance (dFF^2)'],
                               ax_titles=['Targets: avg. response vs. variability - baseline'],
                               facecolors=['gray'], lw=1.3, figsize=(4, 4),
                               alpha=0.1)



def test_flattenOnce():
    from funcsforprajay.funcs import flattenOnce
    print(flattenOnce([list(range(10)) for i in range(3)]))



