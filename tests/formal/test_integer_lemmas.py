import pytest
import z3

from tests.formal.proof import (
    InconclusiveProof,
    ProofFailure,
    VacuousProof,
    prove_broadcast_tile_axis,
    prove_for_all,
    prove_full_slice_axis,
)


def test_full_slice_update_axis_for_all_positive_dimensions_and_indices():
    prove_full_slice_axis()


def test_broadcast_tile_axis_for_all_positive_dimensions_reps_and_indices():
    prove_broadcast_tile_axis()


def test_slice_lemma_rejects_using_the_old_buffer():
    dim, index = z3.Ints("bad_slice_dim bad_slice_index")
    buffer = z3.Function("bad_slice_buffer", z3.IntSort(), z3.BitVecSort(32))
    update = z3.Function("bad_slice_update", z3.IntSort(), z3.BitVecSort(32))
    with pytest.raises(ProofFailure, match="counterexample"):
        prove_for_all(
            [dim > 0, index >= 0, index < dim],
            buffer(index) == update(index),
            theorem="deliberately wrong slice replacement",
        )


def test_proof_rejects_inconsistent_assumptions():
    value = z3.Int("vacuous_value")
    with pytest.raises(VacuousProof, match="inconsistent"):
        prove_for_all([value > 0, value < 0], value == value, theorem="vacuous theorem")


class _UnknownSolver:
    def set(self, **options):
        pass

    def add(self, *expressions):
        pass

    def check(self):
        return z3.unknown

    def reason_unknown(self):
        return "deliberate test result"


def test_proof_rejects_unknown_solver_results():
    with pytest.raises(InconclusiveProof, match="unknown"):
        prove_for_all([], z3.BoolVal(True), theorem="inconclusive theorem", solver_factory=_UnknownSolver)
