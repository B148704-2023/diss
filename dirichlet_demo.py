import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Sample frequencies (should add up to 1)
def sample_frequencies(n, alpha):
    alpha_values = np.linspace(alpha, 1, n)
    return np.random.dirichlet(alpha_values, size=1).flatten()


# Run over num_lineages 2 to 5 and dirichlet alpha 0.1 to 10
# at alpha=1 the proportions will be balanced
alphas = np.linspace(0.1, 10, 50)
num_samples = 100
num_lineages_list = range(2, 6)
data = []
for n_lineages in num_lineages_list:
    for alpha in alphas:
        samples = np.array([sample_frequencies(n_lineages, alpha) for _ in range(num_samples)])
        mean_proportions = samples.mean(axis=0)
        for i in range(n_lineages):
            data.append({
                'Alpha': alpha,
                'Proportion': mean_proportions[i],
                'Lineage': f'Lineage {i+1}',
                'Num Lineages': n_lineages
            })
df = pd.DataFrame(data)

# plot
g = sns.FacetGrid(df, col="Num Lineages", hue="Lineage", col_wrap=2,
                  sharey=False, height=4)
g.map(sns.lineplot, 'Alpha', 'Proportion').add_legend()
g.set_axis_labels('Alpha Parameter', 'Average Proportion')
g.set_titles('Number of Lineages: {col_name}')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Lineage Frequencies Across Different Alpha Parameters')
plt.show()