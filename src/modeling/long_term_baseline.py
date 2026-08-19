from __future__ import annotations

# Compatibility shim for older joblib artifacts pickled with this module path.
# New code should import from src.modeling.long_term_tabular.
from src.modeling.long_term_tabular import (  # noqa: F401
    attach_long_term_preprocessor,
    build_long_term_hist_gradient_boosting_regressor,
    build_long_term_logistic_baseline,
    build_long_term_preprocessor,
    build_long_term_random_forest_classifier,
    build_long_term_random_forest_regressor,
    build_long_term_ridge_baseline,
    get_one_hot_encoder,
    normalize_categorical_missing_values,
)
