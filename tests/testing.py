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
            colors=['blue', 'green', 'purple'], lw=1.3,
            title='Average s2p ROIs spk rate', y_label='spikes rate (Hz)')
