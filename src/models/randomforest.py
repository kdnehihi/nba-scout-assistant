from __future__ import annotations

# Compatibility shim for older imports. New code should import from
# src.models.random_forest.
from src.models.random_forest import (  # noqa: F401
    build_random_forest_classifier,
    build_random_forest_regressor,
)
