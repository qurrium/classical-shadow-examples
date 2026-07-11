"""custom_rho.py — Extend qurry's classical shadow pipeline with a new rho cell function.

Usage:
    from custom_rho import custom_mean_rho, MY_METHOD_KEY

    result = custom_mean_rho(
        shots=shots,
        counts=counts,
        random_basis_array=random_basis_array,
        selected_classical_registers=selected_classical_registers,
        rho_method=MY_METHOD_KEY,
        shadow_basis=shadow_basis,
    )
    # result is a ClassicalShadowBasic TypedDict, same as mean_rho() returns.

To plug in your own algorithm, edit my_rho_m_cell() below.
All other layers (custom_rho_m_core_py, custom_rho_core, custom_mean_rho) are
wiring — they do not need to change unless you also want single-shot or
vectorized variants of your method.
"""

from typing import Literal
from collections.abc import Iterable
import time
import functools as ft

import numpy as np
import numpy.typing as npt
import tqdm

# --- qurry internal imports (read-only, package is not modified) ---
from qurry.process.classical_shadow.rho_process.unitary_set import (
    ShadowRandomBasis,
    ShadowBasisMethod,
    ShadowBasisType,
    DEFAULT_SHADOW_BASIS,
)
from qurry.process.classical_shadow.rho_process.rho_m_cell import (
    rho_m_cell_precomputed,
    rho_m_cell_vectorized,
    RhoMCellMethod,
)
from qurry.process.classical_shadow.rho_process import (
    rho_core as _orig_rho_core,
    mean_rho_core,
    DEFAULT_RHO_METHOD,
    RhoMethodType,
)
from qurry.process.classical_shadow.all_observable.container_kind import ClassicalShadowBasic
from qurry.process.classical_shadow.utils import check_random_basis_array
from qurry.process.utils import (
    counts_list_recount_pyrust,
    shot_counts_selected_clreg_checker_pyrust,
    rho_m_flatten_counts_list_vectorize_pyrust,
)
from qurry.process.classical_shadow.utils import spreadout

# ---------------------------------------------------------------------------
# 1. The new method key
# ---------------------------------------------------------------------------

MY_METHOD_KEY = "my_custom"
"""String token that selects my_rho_m_cell in the dispatch."""


# ---------------------------------------------------------------------------
# 2. New cell function — same signature as rho_m_cell_precomputed
#    Edit this function to implement your own algorithm.
# ---------------------------------------------------------------------------

def my_rho_m_cell(
    single_counts: dict[str, int],
    single_random_basis: list[int],
    selected_clregs_sorted: list[int],
    random_basis_obj: ShadowRandomBasis,
) -> npt.NDArray[np.complex128]:
    r"""Compute the snapshot density matrix with a custom algorithm.

    Implements the same interface as rho_m_cell_precomputed:

        rho_mk^i  =  3 U†_mi |b_k><b_k| U_mi  -  I          (per qubit)
        rho_mk    =  rho_mk^1 ⊗ rho_mk^2 ⊗ ... ⊗ rho_mk^Nq  (Kronecker product)
        rho_m     =  weighted average over shots

    Replace the body below with your new algorithm.

    Args:
        single_counts: Measurement outcome counts for one snapshot circuit.
        single_random_basis: Shadow basis index per qubit for this snapshot.
        selected_clregs_sorted: Classical register indices in reverse-sorted order.
        random_basis_obj: Provides cached per-qubit rho matrices.

    Returns:
        2D complex128 array of shape (2**n_qubits, 2**n_qubits).
    """
    n_qubits = len(selected_clregs_sorted)
    matrix_dim = 2 ** n_qubits

    if not single_counts:
        return np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)

    # -----------------------------------------------------------------------
    # TODO: replace the block below with your new algorithm.
    # This placeholder replicates rho_m_cell_precomputed exactly so the
    # pipeline runs end-to-end before you add your own math.
    # -----------------------------------------------------------------------
    bitstrings = list(single_counts.keys())
    counts_nums = list(single_counts.values())

    single_matrices = np.empty((len(bitstrings), n_qubits), dtype=object)
    for i, bitstring in enumerate(bitstrings):
        for j, (c_i, s_b) in enumerate(zip(selected_clregs_sorted, bitstring)):
            single_matrices[i, j] = (random_basis_obj.cached_precomputed_rho_m_k_i(
                single_random_basis[c_i], s_b
            ) + np.eye(2)) / 3

    all_rho_mk = np.empty((len(bitstrings), matrix_dim, matrix_dim), dtype=np.complex128)
    for i in range(len(bitstrings)):
        all_rho_mk[i] = ft.reduce(np.kron, single_matrices[i, :])

    return np.average(all_rho_mk, axis=0, weights=counts_nums)


# ---------------------------------------------------------------------------
# 3. Extended rho_m_core_py — adds MY_METHOD_KEY branch before the
#    original if/else, then falls through to existing logic.
# ---------------------------------------------------------------------------

def custom_rho_m_core_py(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_array: list[list[int]],
    selected_classical_registers: Iterable[int] | None = None,
    convert_to_single_shot: bool = False,
    rho_method: RhoMCellMethod = "numpy",
    shadow_basis: ShadowBasisType = DEFAULT_SHADOW_BASIS,
) -> tuple[list[npt.NDArray[np.complex128]], list[int], ShadowRandomBasis, float]:
    """Extended version of rho_m_core_py that handles MY_METHOD_KEY."""

    shadow_basis_obj = ShadowBasisMethod.get_shadow_basis(shadow_basis)

    total_system_size, selected_classical_registers = shot_counts_selected_clreg_checker_pyrust(
        shots=shots,
        counts=counts,
        selected_classical_registers=selected_classical_registers,
    )

    if convert_to_single_shot:
        shots, counts, random_unitary_array = spreadout(shots, counts, random_unitary_array)

    begin = time.time()

    selected_clregs_sorted = sorted(selected_classical_registers, reverse=True)
    counts_under_degree_list = counts_list_recount_pyrust(
        counts,
        num_classical_register=total_system_size,
        selected_classical_registers=selected_clregs_sorted,
    )

    if rho_method == MY_METHOD_KEY:
        rho_m_list = [
            my_rho_m_cell(
                single_counts,
                random_unitary_array[idx],
                selected_clregs_sorted,
                shadow_basis_obj,
            )
            for idx, single_counts in enumerate(counts_under_degree_list)
        ]
    elif rho_method == "numpy_vectorized":
        flatten_recount_list_vectorized = rho_m_flatten_counts_list_vectorize_pyrust(
            counts_under_degree_list, random_unitary_array, selected_clregs_sorted
        )
        rho_m_list = [
            rho_m_cell_vectorized(bits_array, count_num, shadow_basis_obj)
            for bits_array, count_num in flatten_recount_list_vectorized
        ]
    else:
        rho_m_list = [
            rho_m_cell_precomputed(
                single_counts,
                random_unitary_array[idx],
                selected_clregs_sorted,
                shadow_basis_obj,
            )
            for idx, single_counts in enumerate(counts_under_degree_list)
        ]

    taken = time.time() - begin

    return rho_m_list, selected_clregs_sorted, shadow_basis_obj, taken


# ---------------------------------------------------------------------------
# 4. Extended rho_core — routes MY_METHOD_KEY to custom_rho_m_core_py,
#    delegates everything else to the original package function.
# ---------------------------------------------------------------------------

def custom_rho_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_array: list[list[Literal[0, 1, 2] | int]],
    selected_classical_registers: Iterable[int] | None = None,
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    shadow_basis: ShadowBasisType = DEFAULT_SHADOW_BASIS,
) -> tuple[list[npt.NDArray[np.complex128]], list[int], ShadowRandomBasis, float]:
    """Drop-in replacement for rho_core that adds MY_METHOD_KEY support."""

    if rho_method == MY_METHOD_KEY:
        return custom_rho_m_core_py(
            shots=shots,
            counts=counts,
            random_unitary_array=random_unitary_array,
            selected_classical_registers=selected_classical_registers,
            convert_to_single_shot=False,
            rho_method=MY_METHOD_KEY,
            shadow_basis=shadow_basis,
        )

    return _orig_rho_core(
        shots=shots,
        counts=counts,
        random_unitary_array=random_unitary_array,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
        shadow_basis=shadow_basis,
    )


# ---------------------------------------------------------------------------
# 5. custom_mean_rho — drop-in replacement for mean_rho.
#    Accepts MY_METHOD_KEY as rho_method in addition to all existing values.
# ---------------------------------------------------------------------------

def custom_mean_rho(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[Literal[0, 1, 2] | int]],
    selected_classical_registers: Iterable[int] | None = None,
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    shadow_basis: ShadowBasisType = DEFAULT_SHADOW_BASIS,
    pbar: tqdm.tqdm | None = None,
) -> ClassicalShadowBasic:
    """Compute the mean rho using any existing method or MY_METHOD_KEY.

    This is a drop-in replacement for
    qurry.process.classical_shadow.all_observable.mean_rho.
    Pass rho_method=MY_METHOD_KEY to use my_rho_m_cell; all other
    rho_method values are forwarded to the original package implementation.

    Returns:
        ClassicalShadowBasic TypedDict — same structure as mean_rho().
    """
    check_random_basis_array(random_basis_array, len(counts), len(next(iter(counts[0].keys()))))

    rho_m_list, selected_clregs_sorted, shadow_basis_obj, taken = custom_rho_core(
        shots=shots,
        counts=counts,
        random_unitary_array=random_basis_array,
        selected_classical_registers=selected_classical_registers,
        rho_method=rho_method,
        shadow_basis=shadow_basis,
    )

    if pbar is not None:
        pbar.set_description(f"| taking time of all rho_m: {taken:.4f} sec")

    expect_rho = mean_rho_core(
        rho_m_list=rho_m_list,
        selected_classical_registers_sorted=selected_clregs_sorted,
    )

    return ClassicalShadowBasic(
        average_snapshots_rho_list=rho_m_list,
        classical_registers_actually=selected_clregs_sorted,
        taking_time=taken,
        rho_method=rho_method,
        random_basis_data=shadow_basis_obj.export(),
        mean_of_rho=expect_rho,
    )
