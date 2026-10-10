import sys

from .config import LGPSModelConfig, LGPSRuntimeConfig, LGPSTrainConfig
from . import aux_tasks as _aux_tasks

# Checkpoints created before the Windows-safe module rename pickle aux classes
# as lgps.aux.*. Keep that import path resolvable while source imports use the
# Windows-safe filename aux_tasks.py.
sys.modules.setdefault(f'{__name__}.aux', _aux_tasks)

__all__ = ('LGPSModelConfig', 'LGPSRuntimeConfig', 'LGPSTrainConfig')
