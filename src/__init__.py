import warnings
import logging

# Suppress torchao PlainLayout deprecation warning.
warnings.filterwarnings(
	"ignore",
	message=r"Deprecation: PlainLayout is deprecated and will be removed in a future release of torchao.*",
	category=UserWarning,
	module=r"torchao\.dtypes\.utils",
)

# Suppress PyTorch lr_scheduler warning when using DeepSpeed
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="torch.optim.lr_scheduler"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="accelerate.scheduler"
)

# Suppress DeepSpeed logging to reduce noise
logging.getLogger("DeepSpeed").setLevel(logging.ERROR)
class SuppressDeepSpeedCacheWarning(logging.Filter):
    def filter(self, record):
        if record.getMessage() is None:
            return True
        if "pytorch allocator cache flushes" in record.getMessage():
            return False
        return True
logging.getLogger().addFilter(SuppressDeepSpeedCacheWarning())
logging.getLogger("DeepSpeed").addFilter(SuppressDeepSpeedCacheWarning())
