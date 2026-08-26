"""
====================================================
QuantAI Professional
Purged K-Fold Cross-Validation
====================================================

PurgedKFold implementation based on López de Prado
"Advances in Financial Machine Learning" (Ch. 7).

Prevents data leakage in time-series ML by:
1. Purging: removing training samples that overlap with test period
2. Embargo: adding a gap between train and test sets

For financial data with future_bars=5 prediction horizon,
embargo should be >= 5 to prevent autocorrelation leakage.
====================================================
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validator for time-series data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    embargo_pct : float, default=0.01
        Embargo percentage of total samples. Gap between train and test.
        For 15m data with future_bars=5, embargo_pct=0.01 on 10000 samples = 100 bars gap.
    purge_pct : float, default=0.0
        Additional purge percentage. Removes training samples that
        have labels overlapping with test period.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0.0, 1.0)")
        if not 0.0 <= purge_pct < 1.0:
            raise ValueError("purge_pct must be in [0.0, 1.0)")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target variable.
        groups : array-like of shape (n_samples,), optional
            Group labels for the samples.

        Yields
        ------
        train_idx : ndarray
            Training set indices.
        test_idx : ndarray
            Test set indices.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # Calculate embargo and purge sizes in samples
        embargo = int(n_samples * self.embargo_pct)
        purge = int(n_samples * self.purge_pct)

        # Standard KFold split points
        fold_size = n_samples // self.n_splits
        test_starts = [i * fold_size for i in range(self.n_splits)]
        test_ends = [(i + 1) * fold_size for i in range(self.n_splits)]
        # Last fold takes remaining samples
        test_ends[-1] = n_samples

        for test_start, test_end in zip(test_starts, test_ends):
            # Test indices
            test_idx = indices[test_start:test_end]

            # Embargo: gap after test set
            embargo_start = test_end
            embargo_end = min(test_end + embargo, n_samples)

            # Purge: remove training samples whose labels overlap with test
            # For financial ML, labels are typically based on future returns
            # So we purge samples from (test_start - purge) to test_end
            purge_start = max(0, test_start - purge)
            purge_end = test_end

            # Training indices: everything except test + embargo + purge
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            train_mask[embargo_start:embargo_end] = False
            train_mask[purge_start:purge_end] = False

            train_idx = indices[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx


class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged K-Fold (CPCV) for more robust validation.

    Creates multiple test sets per fold by combining folds,
    providing more test paths while maintaining purging/embargo.

    Parameters
    ----------
    n_splits : int, default=5
        Number of base folds.
    n_test_folds : int, default=2
        Number of folds to combine for each test set.
    embargo_pct : float, default=0.01
        Embargo percentage.
    purge_pct : float, default=0.0
        Purge percentage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_folds: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 1 <= n_test_folds < n_splits:
            raise ValueError("n_test_folds must be in [1, n_splits)")

        self.n_splits = n_splits
        self.n_test_folds = n_test_folds
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

        self._base_cv = PurgedKFold(
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            purge_pct=purge_pct,
        )

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        # Number of combinations: C(n_splits, n_test_folds)
        from math import comb
        return comb(self.n_splits, self.n_test_folds)

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial splits with purging/embargo.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # Get base fold boundaries
        fold_size = n_samples // self.n_splits
        fold_starts = [i * fold_size for i in range(self.n_splits)]
        fold_ends = [(i + 1) * fold_size for i in range(self.n_splits)]
        fold_ends[-1] = n_samples

        from itertools import combinations

        embargo = int(n_samples * self.embargo_pct)
        purge = int(n_samples * self.purge_pct)

        for test_fold_indices in combinations(range(self.n_splits), self.n_test_folds):
            # Combine selected folds for test set
            test_idx_list = []
            for fi in test_fold_indices:
                test_idx_list.append(indices[fold_starts[fi]:fold_ends[fi]])
            test_idx = np.concatenate(test_idx_list)

            # Embargo after each test fold
            embargo_mask = np.zeros(n_samples, dtype=bool)
            for fi in test_fold_indices:
                emb_start = fold_ends[fi]
                emb_end = min(fold_ends[fi] + embargo, n_samples)
                embargo_mask[emb_start:emb_end] = True

            # Purge before each test fold
            purge_mask = np.zeros(n_samples, dtype=bool)
            for fi in test_fold_indices:
                pur_start = max(0, fold_starts[fi] - purge)
                pur_end = fold_ends[fi]
                purge_mask[pur_start:pur_end] = True

            # Training mask
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            train_mask[embargo_mask] = False
            train_mask[purge_mask] = False

            train_idx = indices[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx


def get_purged_cv(
    cv_type: str = "purged",
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.0,
    n_test_folds: int = 2,
) -> BaseCrossValidator:
    """
    Factory function to get purged cross-validator.

    Parameters
    ----------
    cv_type : str, default="purged"
        "purged" for PurgedKFold, "combinatorial" for CPCV.
    n_splits : int, default=5
        Number of folds.
    embargo_pct : float, default=0.01
        Embargo percentage (gap between train/test).
    purge_pct : float, default=0.0
        Purge percentage (remove overlapping labels).
    n_test_folds : int, default=2
        For CPCV: number of folds per test set.

    Returns
    -------
    BaseCrossValidator
        Configured cross-validator.
    """
    if cv_type == "purged":
        return PurgedKFold(
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            purge_pct=purge_pct,
        )
    elif cv_type == "combinatorial":
        return CombinatorialPurgedKFold(
            n_splits=n_splits,
            n_test_folds=n_test_folds,
            embargo_pct=embargo_pct,
            purge_pct=purge_pct,
        )
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}. Use 'purged' or 'combinatorial'")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    "get_purged_cv",
]