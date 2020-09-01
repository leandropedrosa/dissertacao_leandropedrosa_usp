import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def print_stats(morgan_stats, algoritimo, descritores):
    # Estatísticas de transposição
    morgan_stats_t = morgan_stats.T
    morgan_stats_t = morgan_stats_t.reset_index()
    morgan_stats_t = morgan_stats_t.rename(columns={'index': 'Stats'})

    # Fazer enredo
    plt.style.use('seaborn-colorblind')
    fig, ax1 = plt.subplots(figsize=(10,6))

    morgan_stats_t.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_xticklabels(labels=morgan_stats_t['Stats'].tolist(), fontsize=14, rotation=0)
    ax1.axhline(y=.6, color='indianred', ls='dashed')# xmin=0.25, xmax=0.75)
    ax1.legend_.remove()
    plt.title('Características estatísticas - '+descritores+'X'+algoritimo, fontsize=16)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.tick_params(labelsize=12)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=16,
                loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True,
                shadow=True, ncol=2)
    fig.tight_layout()

    plt.savefig('figures/'+descritores+'X'+algoritimo+'statistics-morgan.png', bbox_inches='tight',
                transparent=False, format='png', dpi=300)
    plt.show();