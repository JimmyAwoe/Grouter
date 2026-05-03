"""
Megatron integration module for grouter EP optimization
"""

from .dataset_adapter import (
    MegatronDatasetProcessor,
    create_megatron_processor
)

from .data_processor import (
    TokenizerFactory,
    create_tokenizer,
)

from .add_argument import _add_grouter_args

from .grouter_distillation_hooks import (
    create_grouter_distillation_hook,
    register_grouter_hooks,
    clear_grouter_hooks
)

from .grouter_distillation_trainer import GrouterDistillationTrainer

__all__ = [
    'MegatronDatasetProcessor',
    'create_megatron_processor',
    'TokenizerFactory',
    'create_tokenizer',
]
