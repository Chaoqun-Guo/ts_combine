"""
Models
------
"""

from ts_combine.utils.logging import get_logger

logger = get_logger(__name__)

# Forecasting
from ts_combine.models.forecasting.arima import ARIMA
from ts_combine.models.forecasting.auto_arima import AutoARIMA
from ts_combine.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal
from ts_combine.models.forecasting.catboost_model import CatBoostModel
from ts_combine.models.forecasting.croston import Croston
from ts_combine.models.forecasting.exponential_smoothing import ExponentialSmoothing
from ts_combine.models.forecasting.fft import FFT
from ts_combine.models.forecasting.kalman_forecaster import KalmanForecaster
from ts_combine.models.forecasting.linear_regression_model import LinearRegressionModel
from ts_combine.models.forecasting.prophet_model import Prophet
from ts_combine.models.forecasting.random_forest import RandomForest
from ts_combine.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
from ts_combine.models.forecasting.regression_model import RegressionModel
from ts_combine.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
from ts_combine.models.forecasting.tbats import BATS, TBATS
from ts_combine.models.forecasting.theta import FourTheta, Theta
from ts_combine.models.forecasting.varima import VARIMA

try:
    from ts_combine.models.forecasting.block_rnn_model import BlockRNNModel
    from ts_combine.models.forecasting.nbeats import NBEATSModel
    from ts_combine.models.forecasting.nhits import NHiTSModel
    from ts_combine.models.forecasting.rnn_model import RNNModel
    from ts_combine.models.forecasting.tcn_model import TCNModel
    from ts_combine.models.forecasting.tft_model import TFTModel
    from ts_combine.models.forecasting.transformer_model import TransformerModel

except ModuleNotFoundError:
    logger.warning(
        "Support for Torch based models not available. "
        'To enable them, install "ts_combine", "u8darts[torch]" or "u8darts[all]" (with pip); '
        'or "u8darts-torch" or "u8darts-all" (with conda).'
    )

try:
    from ts_combine.models.forecasting.gradient_boosted_model import LightGBMModel
except ModuleNotFoundError:
    logger.warning(
        "Support for LightGBM not available."
        "To enable LightGBM support in Darts, follow the detailed "
        "install instructions for LightGBM in the README: "
        "https://github.com/unit8co/darts/blob/master/README.md"
    )

from ts_combine.models.filtering.gaussian_process_filter import GaussianProcessFilter
from ts_combine.models.filtering.kalman_filter import KalmanFilter

# Filtering
from ts_combine.models.filtering.moving_average import MovingAverage
from ts_combine.models.forecasting.baselines import NaiveEnsembleModel

# Ensembling
from ts_combine.models.forecasting.ensemble_model import EnsembleModel
