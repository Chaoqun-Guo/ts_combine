import logging
import unittest

from ts_combine import TimeSeries
from ts_combine.dataprocessing.transformers import BaseDataTransformer
from ts_combine.utils.timeseries_generation import constant_timeseries


class BaseDataTransformerTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class DataTransformerMock(BaseDataTransformer):
        def __init__(self):
            super().__init__(name="DataTransformerMock")
            self.transform_called = False

        @staticmethod
        def ts_transform(series: TimeSeries) -> TimeSeries:
            return series + 10

    def test_input_transformed(self):
        # given
        test_input = constant_timeseries(value=1, length=10)
        mock = self.DataTransformerMock()

        # when
        transformed = mock.transform(test_input)

        expected = constant_timeseries(value=11, length=10)
        self.assertEqual(transformed, expected)
