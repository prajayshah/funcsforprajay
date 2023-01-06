import numpy as np

def test_plot_bar_with_points():
    data = [np.random.random(20) for i in range(3)]
    from funcsforprajay.plotting.plotting import plot_bar_with_points
    plot_bar_with_points(data=data, bar=False, x_tick_labels=['baseline', 'interictal', 'ictal'],
            colors=['blue', 'green', 'purple'], lw=1, alpha=1, shrink_text=1, points=True,
            title='Average s2p ROIs spk rate', y_label='spikes rate (Hz)', figsize=(3,3))

test_plot_bar_with_points()

