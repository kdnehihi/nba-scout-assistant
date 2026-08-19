from __future__ import annotations

# Compatibility shim for older imports. New code should import from
# src.dataset.long_term_modeling.
from src.dataset.long_term_modeling import (  # noqa: F401
    infer_long_term_feature_columns,
    is_long_term_model_feature,
    prepare_long_term_modeling_data,
    prepare_long_term_training,
    validate_long_term_columns,
    validate_long_term_split,
)
