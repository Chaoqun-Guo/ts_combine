<div align="center">
<h1>TS_Combine</h1>

A time series analysis integration toolkit in Python

![](static/images/ts-combine.png)
</div>
<div align="center">

| Section | Description |
|:-:|:-:|
| [Installation](#installation) | Installing the dependencies and ts_combine |
| [Getting started](#getting-started) | A quick introduction on how to use ts_combine |
| [Available features](#available-features) | An extensive overview of ts_combine's functionalities |
| [Documentation](#documentation) | A link to our API reference and a gallery of examples |
| [Contributing](#contributing) | A guide for heroes willing to contribute |
| [Acknowledgements](#acknowledgements) | Acknowledgements |
| [License](#license) | License |
</div>

<hr>
<div align="center">

### **\*Notice, ts_combine integrates the following projects.\***
| [darts](https://github.com/unit8co/darts) | [Merlion](https://github.com/salesforce/Merlion) | [prophet](https://github.com/facebook/prophet) | [tslearn](https://github.com/tslearn-team/tslearn/) | [tsfresh](https://github.com/blue-yonder/tsfresh) | [sktime](https://github.com/alan-turing-institute/sktime) |
</div>

<hr>

## Installation

- Git:
  ```
  git clone https://github.com/Chaoqun-Guo/ts_combine.git
  cd ts_combine
  
  # update submodules for source codes, you can also ignore this step for just using the python package insteade of installing these packages through source codes.

  git submodule update --init repos/darts
  git submodule update --init repos/Merlion
  git submodule update --init repos/Prophet
  git submodule update --init repos/sktime
  git submodule update --init repos/tsfresh
  git submodule update --init repos/tslearn

  conda create -n ts_combine #using conda to create a new python environment.
  conda activate ts_combine && pip install -r requirements.txt # install dependencies for ts_combine
  ```
In order for the installation to be successful, the required dependencies must be installed. For a more detailed guide on how to install ts_combine, please see the [Documention](./docs/installation.md).
## Getting started
### 1. Getting the data in the right format
ts_combine expects a time series dataset to be formatted as a 3D `numpy` array. The 3D correspond to the number of time series, the number of measurements per time series and the number of dimensions respectively (`n_ts,max_sz,d`). In order to get the data in the right format, please see the [Documention](./docs/getting_the_data_in_the_right_format.md).

It should further be noted that tslearn included in ts_combine supports variable-length timeseries.

### 2. Spend less time on feature engineering
ts_combine contains the `tsfresh` python package, the package provides systematic time-series feature extraction by combining established algorithms from statistics, time-series analysis, signal processing, and nonlinear dynamics with a robust feature selection algorithm. In this context, the term time-series is interpreted in the broadest possible sense, such that any types of sampled data or even event sequences can be characterised.

Data Scientists often spend most of their time either cleaning data or building features. While we cannot change the first thing, the second can be automated. `tsfresh` frees your time spent on building features by extracting them automatically. Hence, you have more time to study the newest deep learning paper, read hacker news or build better models.

`tsfresh` automatically extracts 100s of features from time series. Those features describe basic characteristics of the time series such as the number of peaks, the average or maximal value or more complex features such as the time reversal symmetry statis

<!-- <div align="center">

![tsfresh](./static/images/introduction_ts_exa_features.png)
</div> -->

### 3. Data preprocessing and transformations
Optionally, `tslearn` included in ts_combine has several utilities to preprocess the data. In order to facilitate the convergence of different algorithms, you can scale time series. Alternatively, in order to speed up training times, one can resample the data or apply a piece-wise transformation. 

### 4. Training a model
After getting the data in the right format, a model can be trained. Depending on the use case, ts_combine supports different tasks: classification, clustering and regression. For an extensive overview of possibilities, check out our gallery of examples.
<!-- 
We begin by importing Merlion’s TimeSeries class and the data loader for the Numenta Anomaly Benchmark NAB. We can then divide a specific time series from this dataset into training and testing splits.
```
from merlion.utils import TimeSeries
from ts_datasets.anomaly import NAB

# Data loader returns pandas DataFrames, which we convert to Merlion TimeSeries
time_series, metadata = NAB(subset="realKnownCause")[3]
train_data = TimeSeries.from_pd(time_series[metadata.trainval])
test_data = TimeSeries.from_pd(time_series[~metadata.trainval])
test_labels = TimeSeries.from_pd(metadata.anomaly[~metadata.trainval])
```
We can then initialize and train Merlion’s DefaultDetector, which is an anomaly detection model that balances performance with efficiency. We also obtain its predictions on the test split.
```
from merlion.models.defaults import DefaultDetectorConfig, DefaultDetector
model = DefaultDetector(DefaultDetectorConfig())
model.train(train_data=train_data)
test_pred = model.get_anomaly_label(time_series=test_data)
```
Next, we visualize the model's predictions.
```
from merlion.plot import plot_anoms
import matplotlib.pyplot as plt
fig, ax = model.plot_anomaly(time_series=test_data)
plot_anoms(ax=ax, anomaly_labels=test_labels)
plt.show()
```
Finally, we can quantitatively evaluate the model. The precision and recall come from the fact that the model fired 3 alarms, with 2 true positives, 1 false negative, and 1 false positive. We also evaluate the mean time the model took to detect each anomaly that it correctly detected. -->

<!-- ```
from merlion.evaluate.anomaly import TSADMetric
p = TSADMetric.Precision.value(ground_truth=test_labels, predict=test_pred)
r = TSADMetric.Recall.value(ground_truth=test_labels, predict=test_pred)
f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=test_pred)
mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=test_pred)
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n"
      f"Mean Time To Detect: {mttd}
```
```
Precision: 0.6667, Recall: 0.6667, F1: 0.6667
Mean Time To Detect: 1 days 10:30
``` -->
<!-- <div align="center">

![anom](./static/images/anom_example.png)
</div> -->

## Available features
### 1. Detection、Forecasting、Classification
The table below provides a visual overview of how ts_combine key features for time series anomaly detection and/or forecastin and classification.

| Feature | Supported | Examples |
| :---: | :---: | :--: |
| Detection | ✅ | [Anomaly Detection](./examples/merlion/anomaly/)|
| Forecasting | ✅ | [Multivariate Forecasting](./examples/merlion/forecast/) |
| Classification | ✅ | [SVM](./examples/tslearn/classification/) |
| Clustering | ✅ | [KMEANS](./examples/tslearn/clustering/) |
| Benchmarking | ✅ | ... |
| Visualization | ✅ | ... |

### 2. Models
Here's a breakdown of the forecasting models currently implemented in ts_combine. We are constantly working on bringing more models and features.

| Model | Univariate | Multivariate | Probabilistic | Multiple-series training | Past-observed covariates support | Future-known covariates | Static covariates support | Reference |
|:---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
[ARIMA](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/arima.py) | ✅ | | ✅ | | | ✅ | |
[VARIMA](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/varima.py) | ✅ | ✅ | | | | ✅ | |
[AutoARIMA](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/auto_arima.py) | ✅ | | | | | ✅ | |
[StatsForecastAutoARIMA](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/sf_auto_arima.py)| ✅ | | ✅ | | | ✅ | | [statsforecast](https://github.com/Nixtla/statsforecast)
[ExponentialSmoothing](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/exponential_smoothing.py) | ✅ | | ✅ | | | | |
[BATS+TBATS](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/tbats.py) | ✅ | | ✅ | | | | | [TBATS paper](https://robjhyndman.com/papers/ComplexSeasonality.pdf)
[Theta+FourTheta](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/theta.py)| ✅ | | | | | | | [Theta](https://robjhyndman.com/papers/Theta.pdf) & [4 Theta](https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R)
[Prophet](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/prophet_model.py) | ✅ | | ✅ | | | ✅ | | [Prophet repo](https://github.com/facebook/prophet)
[FFT](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/fft.py) | ✅ | | | | | | |
[KalmanForecaster](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/kalman_forecaster.py) | ✅ | ✅ | ✅ | | | ✅ | | [N4SID paper](https://people.duke.edu/~hpgavin/SystemID/References/VanOverschee-Automatica-1994.pdf)
[Croston](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/croston.py)| ✅ | | | | | | |
[RegressionModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/regression_model.py)| ✅ | ✅ | | ✅ | ✅ | ✅ | |
[RandomForest](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/random_forest.py) | ✅ | ✅ | | ✅ | ✅ | ✅ | |
[LinearRegressionModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/linear_regression_model.py) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
[LightGBMModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/gradient_boosted_model.py) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
[CatBoostModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/catboost_model.py) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
[RNNModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/rnn_model.py)| ✅ | ✅ | ✅ | ✅ | | ✅ | | [DeepAR paper](https://arxiv.org/abs/1704.04110)
[BlockRNNModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/block_rnn_model.py)| ✅ | ✅ | ✅ | ✅ | ✅ | | |
[NBEATSModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/nbeats.py) | ✅ | ✅ | ✅ | ✅ | ✅ | | | [N-BEATS paper](https://arxiv.org/abs/1905.10437)
[NHiTSModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/nhits.py) | ✅ | ✅ | ✅ | ✅ | ✅ | | | [N-HiTS paper](https://arxiv.org/abs/2201.12886)
[TCNModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/tcn_model.py) | ✅ | ✅ | ✅ | ✅ | ✅ | | | [TCN paper](https://arxiv.org/abs/1803.01271), [DeepTCN paper](https://arxiv.org/abs/1906.04397), [blog post](https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4)
[TransformerModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/transformer_model.py) | ✅ | ✅ | ✅ | ✅ | ✅ | | |
[TFTModel](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/tft_model.py)| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | [TFT paper](https://arxiv.org/pdf/1912.09363.pdf), [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/models.html)
[Naive Baselines](https://github.com/unit8co/darts/tree/20b3a298b5f3da5fb9b94961862da012af6cf7ff/darts/models/forecasting/forecasting_model.py) | ✅ | | | | | | |


* **Forecasting Models:** A large collection of forecasting models; from statistical models (such as ARIMA) to deep learning models (such as N-BEATS).
* **Multivariate Support:** `TimeSeries` can be multivariate - i.e., contain multiple time-varying dimensions instead of a single scalar value. Many models can consume and produce multivariate series.
* **Multiple series training:** All machine learning based models (incl. all neural networks) support being trained on multiple (potentially multivariate) series. This can scale to large datasets.
* **Probabilistic Support:** `TimeSeries` objects can (optionally) represent stochastic time series; this can for instance be used to get confidence intervals, and many models support different flavours of probabilistic forecasting (such as estimating parametric distributions or quantiles).
* **Past and Future Covariates support:** Many models in Darts support past-observed and/or future-known covariate (external data) time series as inputs for producing forecasts.
* **Static Covariates support:** In addition to time-dependent data, `TimeSeries` can also contain static data for each dimension, which can be exploited by some models.
* **Hierarchical Reconciliation:** Darts offers transformers to perform reconciliation. These can make the forecasts add up in a way that respects the underlying hierarchy.
* **Regression Models:** It is possible to plug-in any scikit-learn compatible model to obtain forecasts as functions of lagged values of the target series and covariates.
* **Data processing:** Tools to easily apply (and revert) common transformations on time series data (scaling, filling missing values, boxcox, ...)
* **Metrics:** A variety of metrics for evaluating time series' goodness of fit; from R2-scores to Mean Absolute Scaled Error.
* **Backtesting:** Utilities for simulating historical forecasts, using moving time windows.
* **PyTorch Lightning Support:** All deep learning models are implemented using PyTorch Lightning, supporting among other things custom callbacks, GPUs/TPUs training and custom trainers.
* **Filtering Models:** Darts offers three filtering models: `KalmanFilter`, `GaussianProcessFilter`, and `MovingAverage`, which allow to filter time series, and in some cases obtain probabilistic inferences of the underlying states/values.
* **Datasets** The `darts.datasets` submodule contains some popular time series datasets for rapid experimentation.
### For examples:
#### Forecasting in sktime
```
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
mean_absolute_percentage_error(y_test, y_pred)
>>> 0.08661467738190656
```
#### Classification in sktime
```
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_arrow_head()
X_train, X_test, y_train, y_test = train_test_split(X, y)
classifier = TimeSeriesForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
>>> 0.8679245283018868
```
## Documentation
For example code and an introduction to ts_combine, see the Jupyter notebooks in examples, and the guided walkthrough here. You may find detailed API documentation (including the example code) here. The technical report outlines ts_combine's overall architecture and presents experimental results on time series anomaly detection & forecasting for both univariate and multivariate time series.
## Contributing
The development is ongoing, and we welcome suggestions, pull requests and issues on GitHub. All contributors will be acknowledged on the [change log page](./docs/change_log_page.md).
## Acknowledgements
ts_combine integrates the following projects, you can click the link for more details. [darts](https://github.com/unit8co/darts), [Merlion](https://github.com/salesforce/Merlion), [prophet](https://github.com/facebook/prophet), [tslearn](https://github.com/tslearn-team/tslearn/), [tsfresh](https://github.com/blue-yonder/tsfresh), [sktime](https://github.com/alan-turing-institute/sktime). 

Thanks for these excellent [open source projects](./repos/)
## License
ts_combine is licensed under the [MIT license](./LICENSE).
