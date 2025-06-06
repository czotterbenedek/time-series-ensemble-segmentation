# Ensemble Segmentation Methods for Time Series

This project focuses on time series segmentation using ensemble models. Throughout the research, I studied various state-of-the-art time series segmentation algorithms in depth, such as:

- **PELT**
- **CLaSP**
- **BinSeg**
- **Bottom-Up**
- **Window-based approach**

Additionally, I built a **Temporal Convolutional Network (TCN)-based autoencoder** for unsupervised segmentation tasks.

---

## Ensemble-Based Approaches

I developed and compared two different ensemble-based approaches:

---

### 1. Change Point Detection with Hierarchical Clustering

In this approach, I aimed to adapt **majority voting**, a common ensemble technique in supervised classification problems, to an **unsupervised time series segmentation** task.

#### Summary of the Method:

- Each segmentation model outputs a list of predicted change points (timestamps).
- I ran multiple algorithms independently, each providing a set of predicted change points.
- I concatenated all predicted timestamps and then applied **hierarchical clustering** to group similar change points together.

#### Clustering Method:

- `scipy.cluster.hierarchy.linkage()` with the `'single'` method  
  *(more suitable for one-dimensional time values)*

#### Threshold Selection:

I determined the optimal clustering threshold using **inconsistency statistics**, selecting the level with the **maximum variation in inconsistency scores**.

##### Pseudocode:

function find_optimal_threshold(Z, max_depth):
    max_variation ← 0
    best_threshold ← None
    for depth from 1 to max_depth:
        inconsistency ← inconsistent(Z, depth)
        values ← inconsistency[:, 3]
        threshold ← mean(values) + std(values)
        variation ← std(values)
        if variation > max_variation:
            max_variation ← variation
            best_threshold ← threshold
    return best_threshold

## Final Output

Each cluster of change points is represented by its **mean timestamp**, reducing redundancy among models.

### Tested Combinations

- **Similar complexity:** e.g., PELT, BinSeg, Bottom-Up
- **Diverse approaches:** e.g., CLaSP, BinSeg, Window

---

## 2. Cluster-Based Prediction Selection

In the second approach, I aimed to match each time series with the algorithm that performs best on similar series, based on their statistical features.

### Method Overview

I extracted statistical features from time series:

- Length
- Trend
- Autocorrelation
- Coefficient of Variation (standard deviation / mean)
- Moving window standard deviation (window size = 10)

- Used KMeans clustering to group similar time series.
- Determined the optimal number of clusters using the elbow method → **4 clusters**.

---

### Dataset Setup

#### Time Series Segmentation Benchmark (TSSB)

- 75 time series
- Split: 60% train / 40% test
- 45 train / 30 test

#### Human Activity Segmentation Challenge (HAS)

- 250 time series
- Split: 60% train / 40% test
- 150 train / 100 test

---

### Process

1. Extracted statistical features from training time series.
2. Clustered using KMeans.
3. Evaluated the performance of each algorithm on each cluster:
   - CLaSP
   - BinSeg
   - Bottom-Up
   - Window
   - TCN Autoencoder
4. For each cluster, identified the best-performing algorithm.
5. For test series:
   - Extracted features.
   - Assigned each series to a cluster.
   - Applied the best algorithm for that cluster.

---

### Visual Insights

- Clusters showed clear separation based on length and autocorrelation.
- Some series were distinguished by extreme coefficient of variation, visible in cluster plots.

---


