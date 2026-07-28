# Country Socio-Economic Clustering: KMeans and DBSCAN Comparison

## Description

This project applies unsupervised machine learning to segment 167 countries into groups based on socio-economic and health indicators. The objective is to identify which countries are in the greatest need of humanitarian and development aid, without relying on any pre-existing labels or classifications.

The analysis follows a complete clustering workflow: data acquisition, standardization, exploratory correlation analysis, dimensionality reduction via Principal Component Analysis (PCA), model selection through quantitative evaluation metrics, and a comparative assessment of two clustering algorithms with fundamentally different assumptions.

The primary use case is aid prioritization. An organization with limited resources cannot evaluate 167 countries individually; clustering reduces the problem to a small number of interpretable tiers, each of which can be assessed as a single unit. A secondary and equally important use case is methodological: the notebook documents why the highest-scoring model is not always the correct model, and why algorithm selection must be driven by the underlying structure of the data rather than by a single evaluation metric.

## Dataset

The dataset is sourced from Kaggle (`rohan0301/unsupervised-learning-on-country-data`) and is downloaded programmatically at runtime via `kagglehub`. It contains 167 records and 10 columns.

| Column | Description |
|--------|-------------|
| `country` | Country name (identifier, excluded from modeling) |
| `child_mort` | Deaths of children under 5 years of age per 1,000 live births |
| `exports` | Exports of goods and services as a percentage of GDP per capita |
| `health` | Total health spending as a percentage of GDP per capita |
| `imports` | Imports of goods and services as a percentage of GDP per capita |
| `income` | Net income per person |
| `inflation` | Annual growth rate of the total GDP |
| `life_expec` | Average life expectancy at birth |
| `total_fer` | Total fertility rate (children per woman) |
| `gdpp` | GDP per capita |

Note that the Kaggle download directory contains a data dictionary file in addition to the primary dataset. The loading logic explicitly filters this file out rather than relying on the ordering returned by `glob`.

## Features

- **Automated dataset acquisition.** The dataset is retrieved directly from Kaggle at runtime, with explicit filtering to select the correct CSV file from the download directory.
- **Feature standardization.** All numeric features are scaled using `StandardScaler`, a prerequisite for distance-based clustering algorithms given that the raw features span several orders of magnitude.
- **Correlation analysis.** A correlation heatmap identifies which indicators move together and which features most strongly drive separation between countries.
- **Principal Component Analysis.** The nine-dimensional feature space is reduced to two components, retaining approximately 63.1 percent of the total variance. PCA is used both for two-dimensional visualization and as an alternative input space for clustering.
- **Systematic selection of k.** The optimal number of clusters is evaluated across a range of values using both the elbow method (inertia) and silhouette score, computed on both the full standardized feature space and the PCA-reduced space.
- **Cluster interpretation and labeling.** Clusters are profiled by their mean values across key indicators and assigned descriptive names, converting numeric cluster identifiers into meaningful development tiers.
- **Centroid projection.** KMeans centroids are projected into PCA space and inverse-transformed back to the original feature scale, allowing them to be overlaid on both PCA plots and raw-feature scatter plots.
- **Manual grid search for DBSCAN.** Because `GridSearchCV` requires target labels and is therefore unsuitable for unsupervised estimators, a custom exhaustive search over `eps` and `min_samples` is implemented, with guard conditions that skip degenerate parameter combinations before scoring.
- **Comparative algorithm evaluation.** KMeans and DBSCAN results are compared directly, with explicit reasoning documented for why the parameter set with the highest silhouette score was rejected in favor of a lower-scoring but substantially more useful configuration.

## Prerequisites and Installation

### Requirements

- Python 3.11 or later
- A Kaggle account with API credentials configured, as required by `kagglehub` for dataset download

### Dependencies

| Package | Purpose |
|---------|---------|
| `kagglehub` | Programmatic dataset download from Kaggle |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations and parameter grid generation |
| `scikit-learn` | Standardization, PCA, KMeans, DBSCAN, silhouette scoring |
| `matplotlib` | Plotting |
| `seaborn` | Correlation heatmap and pair plots |
| `jupyter` | Notebook execution environment |

### Installation

Install the required packages using pip:

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn jupyter
```

Alternatively, create an isolated conda environment:

```bash
conda create -n country-clustering python=3.11
conda activate country-clustering
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Kaggle Credentials

`kagglehub` requires valid Kaggle API credentials. Download `kaggle.json` from your Kaggle account settings and place it in the appropriate directory for your operating system:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Running the Notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook countries_clustring.ipynb
```

Execute all cells sequentially from top to bottom. Several cells depend on variables defined in earlier cells (`scaled`, `pca_result`, `kmeans`, `df['cluster']`), so running cells out of order will produce `NameError` or `KeyError` exceptions.

### Core Workflow

**1. Load and inspect the data.**

```python
import kagglehub, glob, os
import pandas as pd
import numpy as np

path = kagglehub.dataset_download("rohan0301/unsupervised-learning-on-country-data")
csv_files = glob.glob(os.path.join(path, '*.csv'))

# Filter out the data dictionary file explicitly
data_path = [f for f in csv_files if 'dictionary' not in f.lower()][0]
df = pd.read_csv(data_path)
print(df.shape)
```

**2. Standardize the feature matrix.**

```python
from sklearn.preprocessing import StandardScaler

num_cols = df.drop(columns='country')
scaler = StandardScaler()
scaled = scaler.fit_transform(num_cols)
```

**3. Reduce dimensionality with PCA.**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
```

**4. Select the number of clusters.**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_range = range(2, 16)
inertias, sil_scores = [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled, labels))
```

**5. Fit the final KMeans model and profile the clusters.**

```python
best_k = 6
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(scaled)

cluster_profile = df.groupby('cluster')[['child_mort', 'income', 'gdpp', 'life_expec']].mean()
print(cluster_profile)
```

**6. Assign interpretable cluster names.**

Cluster identifiers assigned by KMeans are arbitrary and are not stable across runs with different random seeds or modified input data. The cluster profile table must be inspected before mapping names, and the mapping must be revalidated whenever the model is refit.

```python
cluster_names = {
    0: 'Developing',
    1: 'Developed',
    2: 'Needs Aid',
    3: 'Critical Need',
    4: 'Ultra High Income'
}

df['cluster_name'] = df['cluster'].map(cluster_names)
```

**7. Tune DBSCAN via manual grid search.**

```python
from sklearn.cluster import DBSCAN
import itertools

eps_values = np.arange(0.3, 2.0, 0.1)
min_samples_values = range(3, 10)

results = []
for eps, min_samples in itertools.product(eps_values, min_samples_values):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pca_result)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Silhouette score requires at least two distinct clusters
    if n_clusters < 2:
        continue

    # Silhouette score requires at least two non-noise observations
    mask = labels != -1
    if mask.sum() < 2:
        continue

    score = silhouette_score(pca_result[mask], labels[mask])
    results.append((eps, min_samples, n_clusters, n_noise, score))
```

**8. Fit the final DBSCAN model.**

```python
best_db = DBSCAN(eps=0.6, min_samples=8)
df['cluster_dbscan'] = best_db.fit_predict(pca_result)
```

## Methodology and Findings

### Principal Component Analysis

The first two principal components account for 45.95 percent and 17.18 percent of the total variance respectively, for a combined 63.13 percent. Approximately 37 percent of the original variance is therefore discarded in the two-dimensional projection. This is acceptable for visualization but should be considered when interpreting cluster separation observed in PCA plots, as distinctions present only in the discarded components will not be visible.

### KMeans Behavior with Outliers

KMeans assigns every observation to a cluster by construction; it has no mechanism for identifying observations that do not belong to any group. At higher values of k, this causes the algorithm to isolate extreme outliers into single-member or near-single-member clusters rather than discovering additional meaningful structure. In this dataset, Nigeria and a small set of high-income micro-economies (Luxembourg, Malta, Singapore) are repeatedly separated in this manner. This behavior is documented in the notebook and is the motivation for evaluating DBSCAN as an alternative.

### DBSCAN Parameter Selection

The manual grid search returned `eps=0.3, min_samples=6` as the configuration with the highest silhouette score (0.6898). This configuration was rejected. Silhouette score for DBSCAN is computed only over non-noise observations, so a configuration that discards the majority of the dataset can achieve an inflated score while providing no practical value. The selected configuration labeled 146 of 167 countries as noise, retaining only 21 countries across three clusters.

The configuration `eps=0.6, min_samples=8` was selected instead, producing three clusters with 41 noise points (approximately 25 percent of the dataset) at a silhouette score of 0.4197.

| eps | min_samples | Clusters | Noise Points | Silhouette |
|-----|-------------|----------|--------------|------------|
| 0.3 | 6 | 3 | 146 | 0.6898 |
| 0.4 | 8 | 5 | 113 | 0.5990 |
| 0.6 | 9 | 3 | 44 | 0.4237 |
| 0.6 | 8 | 3 | 41 | 0.4197 |

### Algorithm Comparison

Across the full parameter grid, DBSCAN's silhouette score rose monotonically with the proportion of observations classified as noise. This indicates that the dataset does not exhibit the dense, well-separated regions that DBSCAN is designed to detect. Instead, the transition between development tiers is largely continuous, with countries distributed along a gradient rather than clustered into discrete high-density groups.

This finding explains why a centroid-based method such as KMeans produces more complete and more interpretable partitions on this dataset than a density-based method, and illustrates that algorithm selection in unsupervised learning must account for the geometric structure of the data rather than being driven by a single scalar metric.

## Project Structure

```
.
├── countries_clustring.ipynb    Primary analysis notebook
└── README.md                    Project documentation
```

## Contributing

Contributions are welcome. Please observe the following guidelines:

1. Fork the repository and create a feature branch from `main`.
2. Ensure the notebook executes cleanly from a fresh kernel using "Restart Kernel and Run All Cells" before submitting.
3. Clear all cell outputs prior to committing to minimize diff noise and repository size.
4. Set an explicit `random_state` on any stochastic estimator to preserve reproducibility.
5. Document methodological decisions in markdown cells adjacent to the relevant code, particularly where a non-obvious choice has been made.
6. Open a pull request with a clear description of the change and its rationale.

For substantial methodological changes, please open an issue for discussion before beginning work.

## License

This project is released under the MIT License. See the `LICENSE` file for the full text.

The underlying dataset is distributed through Kaggle and remains subject to its original terms of use. Please consult the dataset page for licensing details before redistributing the data itself.
