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
## 2.3 Model Evaluation

To evaluate the different models, I examined two types of metrics. The first is the F1-score, which I further weighted by the length of the time series, since the longer a time series is, the higher the chance of error. This weighted F1 value is calculated based on how well the predictions of each algorithm match the actual change points within an allowed margin of error. The second metric I used is a so-called covering score, which indicates how well the segments returned by the model cover the actual segments. Both evaluation methods are available in the `claspy` library, with the difference that the F1-score in this library is not weighted by the length of the time series by default — I considered this a useful addition. To achieve this, I multiplied the F1 score obtained on each time series by its length, summed these products, and then divided by the total length of all time series.

For evaluation and training, I used the two datasets previously mentioned: TSSB [5] and HAS [6]. Both contain time series of varying lengths and characteristics. The TSSB dataset contains 75 pre-segmented time series, while HAS has 250, which makes it easy to measure the accuracy of model predictions. For models involving a training phase, I trained on 60% of the dataset and evaluated on the remaining 40%. To maintain consistency, algorithms without a training phase were also evaluated only on 40%, ensuring comparability of results.

| Algorithm                | F1 Score (HAS) | Weighted F1 (HAS) | Covering (HAS) | F1 Score (TSSB) | Weighted F1 (TSSB) | Covering (TSSB) |
|--------------------------|----------------|-------------------|----------------|-----------------|--------------------|-----------------|
| CLaSP                    | 0.7402         | 0.7121            | 0.6820         | 0.8923          | 0.7066             | 0.6664          |
| Window                   | 0.7235         | 0.7462            | 0.7385         | 0.8890          | 0.7705             | 0.6560          |
| BinSeg                   | 0.6459         | 0.6100            | 0.6412         | 0.8385          | 0.5962             | 0.3663          |
| BottomUp                 | 0.7121         | 0.7462            | 0.6820         | 0.6029          | 0.6940             | 0.6175          |
| PELT                     | 0.7402         | 0.7235            | 0.6459         | 0.5772          | 0.6376             | 0.5225          |
| TCN Autoencoder          | 0.4425         | 0.3173            | 0.4592         | 0.4876          | 0.4233             | 0.4472          |
| CLaSP_Window_BinSeg      | 0.7287         | 0.7835            | 0.5759         | 0.6483          | 0.6401             | 0.4215          |
| PELT_BinSeg_BottomUp     | 0.5053         | 0.5538            | 0.6164         | 0.4158          | 0.4395             | 0.6764          |
| Window_BinSeg_BottomUp   | 0.4505         | 0.5335            | 0.5364         | 0.3951          | 0.4260             | 0.4670          |
| Pre-clustering the series| 0.7426         | 0.7504            | 0.6368         | 0.8923          | 0.8890             | 0.8385          |

As can be seen from Table 2, different algorithms achieved varying results across datasets and metrics. On both datasets, the CLaSP algorithm and my own developed algorithm reach the highest scores, except for the weighted F1 on the HAS dataset, where PELT performed best. This is partly because PELT is a very strong time series segmentation algorithm with linear computational complexity relative to the series length. Also, during optimization, I set the maximum number of segments to 7 for PELT — the lowest among the algorithms. This reduces the number of false positives, improving F1 scores. Since PELT was not used in my pre-clustering approach, its F1 scores are lower there.

The reason CLaSP and my pre-clustering algorithm produce identical results on the TSSB data is that the test set contained only 30 series, and the training set 45 series, too few to reveal differences in cluster performance. For the HAS dataset, with more data, differences among algorithms on clusters emerge clearly in the weighted F1 scores. In terms of covering score, CLaSP outperformed all other algorithms consistently.

It is also worth noting the TCN autoencoder’s relatively poor performance, likely due to the need for better optimization of its encoder-decoder architecture and parameters — an area for future improvement.

![Figure 5 - Weighted F1 scores of various algorithms on clusters from the HAS training set](./)

Finally, it is informative to analyze which algorithm performs best on which cluster and the cluster's characteristics. Figure 5 shows that three clusters can be distinctly separated based on series length and autocorrelation, and a fourth cluster is distinguishable by its coefficient of variation. On short time series with high autocorrelation, CLaSP performs best in both weighted F1 and covering. On long series with high autocorrelation, BottomUp is most accurate, though several algorithms perform similarly, likely due to noise. On shorter series with low autocorrelation, Window performs best in weighted F1, while CLaSP leads in covering. On series with high variation coefficient, CLaSP again excels on both metrics.

These results align with expectations: CLaSP uses a classifier-based approach, excelling at pattern-based segmentation, especially on short, repetitive data segments. BottomUp gradually builds segmentations well suited to long series’ global structures. However, on long series, CLaSP tends to generate more false positives, lowering weighted F1. High variation coefficient indicates high dispersion relative to mean, favoring CLaSP's classification logic. Other algorithms may over-segment due to difficulty distinguishing significant change points. Low autocorrelation indicates noisy, chaotic data, where Window methods excel at detecting local changes. CLaSP’s better covering suggests it detects relevant segments well, but with some false positives affecting weighted F1.

---

## 2.4 Summary and Outlook

As demonstrated, time series segmentation is a complex and innovative field, remaining a focus of ongoing research due to its broad applications, including healthcare, finance, and behavior monitoring.

My research centered on joint learning for time series segmentation, which I intend to refine further. In particular, the hierarchical clustering process requires further optimization and tuning of the appropriate number of clusters, as results show. Additionally, further work on the TCN autoencoder’s architecture and optimization is warranted, given its potential as an innovative deep neural network model.

I also aim to develop a method that does not require retraining and reclustering for every new dataset. Instead, I plan to create a pre-trained model trained on a large collection of pre-segmented time series. During training, cluster characteristics and parameters will be learned. This approach will enable automatic selection of the most appropriate segmentation algorithm for new datasets, based on previously learned knowledge, without retraining.

This requires further investigation into time series clustering and defining parameters that identify strengths and weaknesses of individual models.

Overall, I succeeded in developing a method that outperforms state-of-the-art segmentation solutions by leveraging the strengths of individual models. This method relies heavily on proper data preprocessing and deep knowledge of the datasets, highlighting the critical importance of thorough data exploration in time series analysis problems.


