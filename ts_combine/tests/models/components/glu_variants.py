from ts_combine.logging import get_logger

logger = get_logger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Loss tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    from ts_combine.models.components import glu_variants
    from ts_combine.models.components.glu_variants import GLU_FFN
    from ts_combine.tests.base_test_class import DartsBaseTestClass

    class FFNTestCase(DartsBaseTestClass):
        def test_ffn(self):
            for FeedForward_network in GLU_FFN:
                self.feed_forward_block = getattr(glu_variants, FeedForward_network)(
                    d_model=4, d_ff=16, dropout=0.1
                )

                inputs = torch.zeros(1, 4, 4)
                self.feed_forward_block(x=inputs)
