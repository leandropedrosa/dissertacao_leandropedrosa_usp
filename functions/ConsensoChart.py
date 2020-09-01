import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def print_consenso(stats, algoritimo, descritores):
    # Transpose stats
    stats_t = stats.T
    stats_t = stats_t.reset_index()
    stats_t = stats_t.rename(columns={'index': 'Stats'})

    # Make plot
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(9,5), dpi=130)

    stats_t.plot(kind='bar', ax=ax1, width=0.9)
    ax1.set_xticklabels(labels=stats_t['Stats'].tolist(), fontsize=8, rotation=0)
    ax1.axhline(y=.6, color='indianred', ls='dashed')
    ax1.legend_.remove()
    plt.title('Características estatísticas do QSAR '+algoritimo+' '+descritores, fontsize=12)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.tick_params(labelsize=8)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=9,
                loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=7)
    fig.tight_layout()

    plt.savefig('figures/statistics-consenso-5f'+descritores+'.png', bbox_inches='tight', transparent=False, format='png', dpi=300)
    plt.show();