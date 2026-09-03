"""Candidate equivalence classes under a fixed emission model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EquivalenceResult:
    """Deterministic candidate groups whose emission rows are indistinguishable."""

    groups: tuple[tuple[int, ...], ...]
    labels: NDArray[np.int64]

    @property
    def all_groups_singleton(self) -> bool:
        """Whether this exact duplicate-row check found only singleton groups."""

        return all(len(group) == 1 for group in self.groups)

    @property
    def identifiable(self) -> bool:
        """Compatibility alias; singleton rows do not prove full identifiability."""

        return self.all_groups_singleton


def find_equivalence_classes(
    emission_rows: Any, *, atol: float = 0.0, rtol: float = 0.0
) -> EquivalenceResult:
    """Group candidate emission rows that are equal (or, with a tolerance, close).

    With the default ``atol = rtol = 0`` this is exact row equality -- a true
    equivalence relation and the basis of :func:`find_observable_groups`.

    .. warning::
       With a nonzero ``atol``/``rtol``, "closeness" is not transitive, so the
       grouping is greedy (first-match-wins) and *order-dependent*: each row is
       assigned to the first existing representative it is close to, which is
       single-linkage-like and need not be a valid equivalence partition.  Use a
       tolerance only for a rough scan, never for identifiability decisions.

    Examples
    --------
    >>> import numpy as np
    >>> # exact grouping (the observable-group use): rows 0 and 1 are identical
    >>> find_equivalence_classes([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]).groups
    ((0, 1), (2,))
    >>> # with a tolerance, closeness is not transitive: 0.0~0.6 and 0.6~1.2 are
    >>> # each within 0.7, yet candidates 1 and 2 land in different groups
    >>> rows = np.array([[0.0], [0.6], [1.2]])
    >>> find_equivalence_classes(rows, atol=0.7).groups
    ((0, 1), (2,))
    >>> bool(np.allclose(rows[1], rows[2], atol=0.7))
    True
    """

    matrix = np.asarray(emission_rows)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("emission_rows must be a non-empty candidate-by-probe matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("emission_rows must be numeric")
    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("emission_rows must be finite")
    if atol < 0 or rtol < 0 or not np.isfinite(atol) or not np.isfinite(rtol):
        raise ValueError("atol and rtol must be finite and non-negative")

    groups: list[list[int]] = []
    representatives: list[int] = []
    labels = np.empty(matrix.shape[0], dtype=np.int64)
    for candidate, row in enumerate(matrix):
        assigned = False
        for label, representative in enumerate(representatives):
            if np.allclose(row, matrix[representative], atol=atol, rtol=rtol):
                groups[label].append(candidate)
                labels[candidate] = label
                assigned = True
                break
        if not assigned:
            labels[candidate] = len(groups)
            groups.append([candidate])
            representatives.append(candidate)

    labels.setflags(write=False)
    return EquivalenceResult(
        groups=tuple(tuple(group) for group in groups),
        labels=labels,
    )


def find_observable_groups(emission_rows: Any) -> EquivalenceResult:
    """Group origins with exactly identical fixed emission distributions.

    Origins with the same emission row produce the same trace distribution, so
    only their combined abundance is identifiable; the fitter reports these
    groups and their pooled weights.  Grouping uses *exact* equality only:
    approximate similarity is deliberately excluded because tolerance-based
    pairwise closeness is not a transitive equivalence relation (see the warning
    on :func:`find_equivalence_classes`).  Near-identifiability should be
    assessed with a separate distance or conditioning diagnostic.

    Examples
    --------
    >>> # origins 0 and 1 share an emission row -> one observable group
    >>> find_observable_groups([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]).groups
    ((0, 1), (2,))
    """

    return find_equivalence_classes(emission_rows, atol=0.0, rtol=0.0)
