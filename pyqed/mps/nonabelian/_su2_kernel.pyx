# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True
# distutils: language = c++
"""Optional Cython entry point for SU(2) block2-style local actions."""

import hashlib
import numpy as np
import mmap as _mmap
cimport numpy as cnp
from cython.operator cimport dereference as deref
from cpython.ref cimport PyObject
from libc.stdint cimport int32_t, int64_t, uint64_t
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.stdlib cimport free, realloc
from libcpp cimport bool as cpp_bool
from libcpp.complex cimport complex as cpp_complex
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef class _SU2RouteBuffers:
    """Lifetime owner for the direct NumPy route-construction buffers."""

    cdef void* routes
    cdef void* outputs

    def __cinit__(self):
        self.routes = NULL
        self.outputs = NULL

    def __dealloc__(self):
        if self.routes != NULL:
            free(self.routes)
            self.routes = NULL
        if self.outputs != NULL:
            free(self.outputs)
            self.outputs = NULL


cdef cpp_bool _execute_half_sweep_bond(
    void* context,
    int64_t bond,
) noexcept with gil:
    cdef object state = <object><PyObject*>context
    try:
        state["callback"](int(bond))
        return True
    except BaseException as exc:
        state["error"] = exc
        state["traceback"] = exc.__traceback__
        return False


def _cpp_array_revision(*values):
    """Return a stable value-based revision for C++ topology/value caches."""

    digest = hashlib.blake2b(digest_size=8)
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return int.from_bytes(digest.digest(), "little") or 1


def pack_normal_complementary_boundary_routes(
    object parent_ket_sector_ids,
    object parent_entry_offsets,
    object parent_out_sector_ids,
    object parent_channel_offsets,
    object parent_channel_ids,
    object transition_offsets,
    object transition_ids,
    object transition_operator_ids,
    object transition_output_channels,
    object primitive_nonzero,
    object bra_entries,
    object ket_entries,
    long long n_next_sectors,
    long long n_output_channels,
):
    """Pack NC boundary routes and unique output blocks in compiled code."""

    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_sector_arr = (
        np.ascontiguousarray(parent_ket_sector_ids, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] entry_offset_arr = (
        np.ascontiguousarray(parent_entry_offsets, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] out_sector_arr = (
        np.ascontiguousarray(parent_out_sector_ids, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] channel_offset_arr = (
        np.ascontiguousarray(parent_channel_offsets, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] channel_id_arr = (
        np.ascontiguousarray(parent_channel_ids, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] transition_offset_arr = (
        np.ascontiguousarray(transition_offsets, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] transition_id_arr = (
        np.ascontiguousarray(transition_ids, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] operator_id_arr = (
        np.ascontiguousarray(transition_operator_ids, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_channel_arr = (
        np.ascontiguousarray(transition_output_channels, dtype=np.int64).reshape(-1)
    )
    cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] primitive_arr = (
        np.ascontiguousarray(primitive_nonzero, dtype=np.uint8)
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] bra_arr = (
        np.ascontiguousarray(bra_entries, dtype=np.int64).reshape((-1, 8))
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] ket_arr = (
        np.ascontiguousarray(ket_entries, dtype=np.int64).reshape((-1, 8))
    )
    cdef _SU2RouteBuffers buffer_owner = _SU2RouteBuffers.__new__(
        _SU2RouteBuffers
    )
    cdef size_t route_size = 0
    cdef size_t route_capacity = 0
    cdef size_t output_size = 0
    cdef size_t output_capacity = 0
    cdef size_t new_capacity
    cdef void* resized
    cdef int32_t* route_data
    cdef int64_t* output_data
    cdef cnp.npy_intp route_shape[2]
    cdef cnp.npy_intp output_shape[2]
    cdef object route_result
    cdef object output_result
    cdef cpp_map[uint64_t, int64_t] output_index
    cdef cpp_map[uint64_t, int64_t].iterator found
    cdef Py_ssize_t row, entry, bra, ket, channel_pos, transition_pos
    cdef int64_t q_ket, q_bra, parent_channel, transition, operator_id
    cdef int64_t output_channel, output_id
    cdef uint64_t output_key
    cdef int64_t physical_out_charge, physical_in_charge

    if (
        n_next_sectors <= 0
        or n_output_channels <= 0
        or entry_offset_arr.size != ket_sector_arr.size + 1
        or channel_offset_arr.size != out_sector_arr.size + 1
    ):
        raise ValueError("Invalid NC boundary route topology.")

    for row in range(ket_sector_arr.size):
        q_ket = ket_sector_arr[row]
        for entry in range(entry_offset_arr[row], entry_offset_arr[row + 1]):
            q_bra = out_sector_arr[entry]
            for bra in range(bra_arr.shape[0]):
                if bra_arr[bra, 0] != q_bra:
                    continue
                physical_out_charge = bra_arr[bra, 2]
                for ket in range(ket_arr.shape[0]):
                    if ket_arr[ket, 0] != q_ket:
                        continue
                    physical_in_charge = ket_arr[ket, 2]
                    for channel_pos in range(
                        channel_offset_arr[entry],
                        channel_offset_arr[entry + 1],
                    ):
                        parent_channel = channel_id_arr[channel_pos]
                        if (
                            parent_channel < 0
                            or parent_channel + 1 >= transition_offset_arr.size
                        ):
                            continue
                        for transition_pos in range(
                            transition_offset_arr[parent_channel],
                            transition_offset_arr[parent_channel + 1],
                        ):
                            transition = transition_id_arr[transition_pos]
                            operator_id = operator_id_arr[transition]
                            if (
                                physical_out_charge < 0
                                or physical_out_charge >= primitive_arr.shape[0]
                                or physical_in_charge < 0
                                or physical_in_charge >= primitive_arr.shape[1]
                                or operator_id < 0
                                or operator_id >= primitive_arr.shape[2]
                                or primitive_arr[
                                    physical_out_charge,
                                    physical_in_charge,
                                    operator_id,
                                ] == 0
                            ):
                                continue
                            output_channel = output_channel_arr[transition]
                            output_key = <uint64_t>(
                                (
                                    bra_arr[bra, 1] * n_next_sectors
                                    + ket_arr[ket, 1]
                                ) * n_output_channels
                                + output_channel
                            )
                            found = output_index.find(output_key)
                            if found == output_index.end():
                                output_id = <int64_t>(output_size // 6)
                                output_index[output_key] = output_id
                                if output_size + 6 > output_capacity:
                                    new_capacity = (
                                        1024
                                        if output_capacity == 0
                                        else 2 * output_capacity
                                    )
                                    while new_capacity < output_size + 6:
                                        new_capacity *= 2
                                    resized = realloc(
                                        buffer_owner.outputs,
                                        new_capacity * sizeof(int64_t),
                                    )
                                    if resized == NULL:
                                        raise MemoryError(
                                            "Unable to grow NC output topology."
                                        )
                                    buffer_owner.outputs = resized
                                    output_capacity = new_capacity
                                output_data = <int64_t*>buffer_owner.outputs
                                output_data[output_size] = bra_arr[bra, 1]
                                output_data[output_size + 1] = ket_arr[ket, 1]
                                output_data[output_size + 2] = output_channel
                                output_data[output_size + 3] = bra_arr[bra, 6]
                                output_data[output_size + 4] = ket_arr[ket, 6]
                                output_data[output_size + 5] = output_id
                                output_size += 6
                            else:
                                output_id = deref(found).second
                            if route_size + 13 > route_capacity:
                                new_capacity = (
                                    4096
                                    if route_capacity == 0
                                    else 2 * route_capacity
                                )
                                while new_capacity < route_size + 13:
                                    new_capacity *= 2
                                resized = realloc(
                                    buffer_owner.routes,
                                    new_capacity * sizeof(int32_t),
                                )
                                if resized == NULL:
                                    raise MemoryError(
                                        "Unable to grow NC boundary routes."
                                    )
                                buffer_owner.routes = resized
                                route_capacity = new_capacity
                            route_data = <int32_t*>buffer_owner.routes
                            route_data[route_size] = <int32_t>channel_pos
                            route_data[route_size + 1] = <int32_t>bra_arr[bra, 7]
                            route_data[route_size + 2] = <int32_t>ket_arr[ket, 7]
                            route_data[route_size + 3] = <int32_t>transition
                            route_data[route_size + 4] = (
                                <int32_t>physical_out_charge
                            )
                            route_data[route_size + 5] = (
                                <int32_t>physical_in_charge
                            )
                            route_data[route_size + 6] = <int32_t>output_id
                            route_data[route_size + 7] = <int32_t>bra_arr[bra, 4]
                            route_data[route_size + 8] = <int32_t>ket_arr[ket, 4]
                            route_data[route_size + 9] = <int32_t>bra_arr[bra, 3]
                            route_data[route_size + 10] = <int32_t>ket_arr[ket, 3]
                            route_data[route_size + 11] = <int32_t>bra_arr[bra, 5]
                            route_data[route_size + 12] = <int32_t>ket_arr[ket, 5]
                            route_size += 13

    if route_size == 0:
        resized = realloc(buffer_owner.routes, sizeof(int32_t))
        if resized == NULL:
            raise MemoryError("Unable to allocate an empty NC route array.")
        buffer_owner.routes = resized
    elif route_capacity != route_size:
        resized = realloc(
            buffer_owner.routes,
            route_size * sizeof(int32_t),
        )
        if resized != NULL:
            buffer_owner.routes = resized
    if output_size == 0:
        resized = realloc(buffer_owner.outputs, sizeof(int64_t))
        if resized == NULL:
            raise MemoryError("Unable to allocate an empty NC output array.")
        buffer_owner.outputs = resized
    elif output_capacity != output_size:
        resized = realloc(
            buffer_owner.outputs,
            output_size * sizeof(int64_t),
        )
        if resized != NULL:
            buffer_owner.outputs = resized
    route_shape[0] = <cnp.npy_intp>(route_size // 13)
    route_shape[1] = 13
    output_shape[0] = <cnp.npy_intp>(output_size // 6)
    output_shape[1] = 6
    route_result = cnp.PyArray_SimpleNewFromData(
        2,
        route_shape,
        cnp.NPY_INT32,
        buffer_owner.routes,
    )
    output_result = cnp.PyArray_SimpleNewFromData(
        2,
        output_shape,
        cnp.NPY_INT64,
        buffer_owner.outputs,
    )
    cnp.set_array_base(route_result, buffer_owner)
    cnp.set_array_base(output_result, buffer_owner)
    return route_result, output_result


cdef extern from "su2_dmrg_engine.hpp" namespace "pyqed::su2":
    ctypedef cpp_bool (*HalfSweepBondExecutor)(void*, int64_t)
    double reduced_environment_recoupling_coefficient(
        cpp_bool left,
        int64_t physical_output_charge,
        int64_t physical_input_charge,
        int32_t boundary_bra_two_j,
        int32_t boundary_ket_two_j,
        int32_t physical_bra_two_j,
        int32_t physical_ket_two_j,
        int32_t next_bra_two_j,
        int32_t next_ket_two_j,
        int32_t left_channel_two_j,
        int32_t right_channel_two_j,
        int32_t operator_two_j,
        int32_t left_channel_two_m,
        int32_t right_channel_two_m,
        int32_t operator_two_m,
    ) noexcept

    cdef cppclass CppBoundaryBufferHandle "pyqed::su2::BoundaryBufferHandle":
        pass

    const double* boundary_buffer_data(
        const CppBoundaryBufferHandle* handle,
    ) noexcept
    size_t boundary_buffer_size(
        const CppBoundaryBufferHandle* handle,
    ) noexcept
    void release_boundary_buffer(
        CppBoundaryBufferHandle* handle,
    ) noexcept

    cdef cppclass CppFamilyData "pyqed::su2::FamilyData":
        int rank
        vector[int64_t] indices
        vector[double] values

    cdef cppclass CppPackedSiteTensor "pyqed::su2::PackedSiteTensor":
        vector[double] values
        vector[int64_t] offsets
        vector[int64_t] labels
        vector[int64_t] shape_offsets
        vector[int64_t] shapes

    cdef enum CppNCOperator "pyqed::su2::NCOperator":
        pass

    cdef cppclass CppNormalComplementaryTransition "pyqed::su2::NormalComplementaryTransition":
        int source
        int target
        CppNCOperator local_operator
        int64_t first_index
        int64_t second_index
        double coefficient
        uint64_t family_mask

    cdef cppclass CppNormalComplementaryChannel "pyqed::su2::NormalComplementaryChannel":
        int64_t charge
        int two_j
        int64_t point_group

    cdef cppclass CppNormalComplementaryComponentAction "pyqed::su2::NormalComplementaryComponentAction":
        int transition
        int source_component
        int target_component
        int local_two_m
        double coefficient

    cdef cppclass CppNormalComplementaryPlan "pyqed::su2::NormalComplementaryPlan":
        int64_t site
        int64_t left_channels
        int64_t right_channels
        vector[CppNormalComplementaryChannel] left_channel_quantum_numbers
        vector[CppNormalComplementaryChannel] right_channel_quantum_numbers
        vector[int64_t] left_component_offsets
        vector[int64_t] right_component_offsets
        vector[CppNormalComplementaryTransition] transitions
        vector[CppNormalComplementaryComponentAction] component_actions
        uint64_t family_transition_counts[6]

    cdef cppclass CppSystem "pyqed::su2::System":
        CppSystem(
            const double* h1,
            const double* eri,
            size_t n_sites,
            double ecore,
            int64_t n_elec,
            int64_t two_s,
            const int64_t* orb_sym,
            double cutoff,
            cpp_bool include_half,
        ) except +
        size_t n_sites() noexcept
        int64_t n_elec() noexcept
        int64_t two_s() noexcept
        double ecore() noexcept
        double cutoff() noexcept
        cpp_bool include_half() noexcept
        uint64_t revision() noexcept
        size_t stored_integral_elements() noexcept
        size_t stored_family_terms() noexcept
        const CppNormalComplementaryPlan& normal_complementary_plan(
            int64_t site,
        ) except +
        size_t normal_complementary_transition_count() noexcept
        size_t normal_complementary_component_action_count() noexcept
        size_t normal_complementary_memory_bytes() noexcept
        size_t normal_complementary_left_component_count(
            int64_t site,
        ) except +
        size_t normal_complementary_right_component_count(
            int64_t site,
        ) except +
        const double* normal_complementary_primitives(
            int64_t site,
        ) except +
        size_t normal_complementary_primitive_count() noexcept
        size_t normal_complementary_primitive_bytes() noexcept
        void apply_normal_complementary_core(
            int64_t site,
            const double* primitive_components,
            size_t n_primitive_components,
            const double* input,
            size_t n_input,
            size_t batch_size,
            double* output,
            size_t n_output,
        ) except +
        size_t memory_bytes() noexcept
        void family_partition_counts(
            int family_id,
            cpp_bool left,
            int64_t bond,
            size_t* internal,
            size_t* cross,
            size_t* external,
        ) except +
        const CppFamilyData& family(int family_id) except +

    cdef cppclass CppDavidsonResult "pyqed::dmrg::DavidsonResult":
        cpp_bool accepted
        double energy
        vector[cpp_complex[double]] vector
        double residual_norm
        int iterations
        size_t basis_size
        int restarts
        cpp_bool converged
        cpp_bool workspace_reused
        uint64_t matvec_calls
        uint64_t norm_matvec_calls

    cdef cppclass CppBlockDavidsonResult "pyqed::dmrg::BlockDavidsonResult":
        cpp_bool accepted
        vector[double] energies
        vector[vector[cpp_complex[double]]] vectors
        vector[double] residual_norms
        int iterations
        size_t basis_size
        int restarts
        cpp_bool converged
        cpp_bool workspace_reused
        uint64_t matvec_calls

    cdef cppclass CppContextualCoreBlock "pyqed::su2::ContextualCoreBlock":
        int64_t left_channel
        int64_t right_channel
        int64_t rows
        int64_t cols
        vector[double] values

    cdef cppclass CppBlockSVDResult "pyqed::su2::BlockSVDResult":
        vector[cpp_complex[double]] left_values
        vector[double] singular_values
        vector[cpp_complex[double]] right_values
        vector[int64_t] left_offsets
        vector[int64_t] singular_offsets
        vector[int64_t] right_offsets
        vector[int64_t] kept_offsets
        vector[int64_t] kept_indices
        double truncation_error
        double full_squared_norm
        double kept_squared_norm

    cdef cppclass CppCanonicalProjectionInfo "pyqed::su2::CanonicalProjectionInfo":
        cpp_bool compatible
        cpp_bool reused
        string projection_key
        size_t parent_dimension
        size_t orthonormal_dimension
        size_t components
        size_t max_component_dimension
        size_t transform_elements
        double whitening_residual
        double build_seconds

    cdef cppclass CppActiveBondCanonicalSolveResult "pyqed::su2::ActiveBondCanonicalSolveResult":
        CppCanonicalProjectionInfo projection
        CppDavidsonResult davidson
        int complementary_action_status
        size_t metric_routes
        size_t requested_restart_dimension
        size_t workspace_restart_dimension
        size_t estimated_workspace_bytes
        double solve_seconds

    cdef cppclass CppActiveBondStateAverageSolveResult "pyqed::su2::ActiveBondStateAverageSolveResult":
        CppCanonicalProjectionInfo projection
        CppBlockDavidsonResult davidson
        int complementary_action_status
        size_t metric_routes
        size_t requested_restart_dimension
        size_t workspace_restart_dimension
        size_t estimated_workspace_bytes
        double solve_seconds

    cdef cppclass CppActiveBondSplitResult "pyqed::su2::ActiveBondSplitResult":
        CppPackedSiteTensor left
        CppPackedSiteTensor right
        vector[int64_t] bond_labels
        vector[int64_t] bond_dims
        vector[double] singular_values
        vector[int64_t] singular_offsets
        double truncation_error
        double full_squared_norm
        double kept_squared_norm
        uint64_t kept_states
        uint64_t left_topology_revision
        uint64_t left_numeric_revision
        uint64_t right_topology_revision
        uint64_t right_numeric_revision

    cdef cppclass CppOwnedHalfSweepSplitSummary "pyqed::su2::OwnedHalfSweepSplitSummary":
        double truncation_error
        double full_squared_norm
        double kept_squared_norm
        uint64_t kept_states
        uint64_t left_topology_revision
        uint64_t left_numeric_revision
        uint64_t right_topology_revision
        uint64_t right_numeric_revision

    cdef cppclass CppOwnedHalfSweepBondResult "pyqed::su2::OwnedHalfSweepBondResult":
        int64_t bond
        CppActiveBondCanonicalSolveResult solve
        CppOwnedHalfSweepSplitSummary split
        vector[double] state_energies
        vector[double] state_residual_norms
        cpp_bool state_average

    cdef cppclass CppOwnedSplitSiteExport "pyqed::su2::OwnedSplitSiteExport":
        int64_t site
        CppPackedSiteTensor tensor
        vector[int64_t] leg_sector_offsets
        vector[int64_t] leg_sector_labels
        vector[int64_t] leg_sector_dims
        uint64_t topology_revision
        uint64_t numeric_revision

    cdef cppclass CppMovingEnvironment "pyqed::su2::MovingEnvironment":
        CppMovingEnvironment(const CppSystem* system) except +
        cpp_bool install_boundary(
            const string& side,
            int64_t bond,
            const double* values,
            size_t n_values,
            const int64_t* offsets,
            size_t n_offsets,
            const int64_t* labels,
            size_t n_labels,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) except +
        cpp_bool install_metric_boundary(
            const string& side,
            int64_t bond,
            const double* values,
            size_t n_values,
            const int64_t* offsets,
            size_t n_offsets,
            const int64_t* labels,
            size_t n_labels,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) except +
        cpp_bool release_boundary(const string& side, int64_t bond) except +
        void clear_boundaries()
        cpp_bool boundary_installed(
            const string& side,
            int64_t bond,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) noexcept
        cpp_bool metric_boundary_installed(
            const string& side,
            int64_t bond,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) noexcept
        cpp_bool advance_boundary(
            const string& side,
            int64_t parent_bond,
            int64_t child_bond,
            cpp_bool left,
            const int64_t* routes,
            size_t n_routes,
            const double* bra_values,
            size_t n_bra_values,
            const int64_t* bra_offsets,
            size_t n_bra_offsets,
            const int64_t* bra_shape_offsets,
            size_t n_bra_shape_offsets,
            const int64_t* bra_shapes,
            size_t n_bra_shapes,
            const double* ket_values,
            size_t n_ket_values,
            const int64_t* ket_offsets,
            size_t n_ket_offsets,
            const int64_t* ket_shape_offsets,
            size_t n_ket_shape_offsets,
            const int64_t* ket_shapes,
            size_t n_ket_shapes,
            const double* mpo_values,
            size_t n_mpo_values,
            const int64_t* mpo_offsets,
            size_t n_mpo_offsets,
            const int64_t* mpo_shape_offsets,
            size_t n_mpo_shape_offsets,
            const int64_t* mpo_shapes,
            size_t n_mpo_shapes,
            const int64_t* output_offsets,
            size_t n_output_offsets,
            const int64_t* output_shape_offsets,
            size_t n_output_shape_offsets,
            const int64_t* output_shapes,
            size_t n_output_shapes,
            const int64_t* output_labels,
            size_t n_output_labels,
            uint64_t topology_revision,
            uint64_t numeric_revision,
            const double* route_coefficients,
            cpp_bool metric_boundary,
        ) except +
        cpp_bool advance_normal_complementary_boundary(
            const string& side,
            int64_t parent_bond,
            int64_t child_bond,
            cpp_bool left,
            int64_t site,
            cpp_bool reduced_physical,
            cpp_bool dual_right_boundary,
            const int32_t* routes,
            size_t n_routes,
            const double* bra_values,
            size_t n_bra_values,
            const int64_t* bra_offsets,
            size_t n_bra_offsets,
            const int64_t* bra_shape_offsets,
            size_t n_bra_shape_offsets,
            const int64_t* bra_shapes,
            size_t n_bra_shapes,
            const double* ket_values,
            size_t n_ket_values,
            const int64_t* ket_offsets,
            size_t n_ket_offsets,
            const int64_t* ket_shape_offsets,
            size_t n_ket_shape_offsets,
            const int64_t* ket_shapes,
            size_t n_ket_shapes,
            const int64_t* output_offsets,
            size_t n_output_offsets,
            const int64_t* output_shape_offsets,
            size_t n_output_shape_offsets,
            const int64_t* output_shapes,
            size_t n_output_shapes,
            const int64_t* output_labels,
            size_t n_output_labels,
            uint64_t topology_revision,
            uint64_t numeric_revision,
            uint64_t route_topology_revision,
        ) except +
        cpp_bool install_split_site(
            int64_t site,
            vector[double]& values,
            vector[int64_t]& offsets,
            vector[int64_t]& labels,
            vector[int64_t]& shape_offsets,
            vector[int64_t]& shapes,
            vector[int64_t]& leg_sector_offsets,
            vector[int64_t]& leg_sector_labels,
            vector[int64_t]& leg_sector_dims,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) except +
        void configure_state_average(
            const double* weights,
            size_t nroots,
            int64_t center_site,
        ) except +
        void install_state_average_center(
            size_t root,
            const double* values,
            size_t n_values,
        ) except +
        cpp_bool state_average_installed() noexcept
        size_t state_average_roots() noexcept
        int64_t state_average_center_site() noexcept
        const vector[vector[double]]& state_average_center_values() noexcept
        cpp_bool split_site_installed(
            int64_t site,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) noexcept
        void merge_active_bond() except +
        const CppPackedSiteTensor& merged_site() noexcept
        const CppPackedSiteTensor& merged_channel_site() noexcept
        cpp_bool advance_normal_complementary_boundary_from_split_site(
            const string& side,
            int64_t parent_bond,
            int64_t child_bond,
            cpp_bool left,
            int64_t site,
            cpp_bool reduced_physical,
            cpp_bool dual_right_boundary,
            const int32_t* routes,
            size_t n_routes,
            const int64_t* output_offsets,
            size_t n_output_offsets,
            const int64_t* output_shape_offsets,
            size_t n_output_shape_offsets,
            const int64_t* output_shapes,
            size_t n_output_shapes,
            const int64_t* output_labels,
            size_t n_output_labels,
            uint64_t topology_revision,
            uint64_t numeric_revision,
            uint64_t route_topology_revision,
        ) except +
        cpp_bool cached_normal_complementary_boundary_ready(
            const string& side,
            int64_t child_bond,
            int64_t site,
            uint64_t output_topology_revision,
            uint64_t route_topology_revision,
        ) noexcept
        cpp_bool replay_normal_complementary_boundary_from_split_site(
            const string& side,
            int64_t child_bond,
            uint64_t numeric_revision,
        ) except +
        cpp_bool advance_metric_boundary_from_split_site(
            const string& side,
            int64_t parent_bond,
            int64_t child_bond,
            cpp_bool left,
            int64_t site,
            const int64_t* routes,
            size_t n_routes,
            const double* route_coefficients,
            size_t n_route_coefficients,
            const double* mpo_values,
            size_t n_mpo_values,
            const int64_t* mpo_offsets,
            size_t n_mpo_offsets,
            const int64_t* mpo_shape_offsets,
            size_t n_mpo_shape_offsets,
            const int64_t* mpo_shapes,
            size_t n_mpo_shapes,
            const int64_t* output_offsets,
            size_t n_output_offsets,
            const int64_t* output_shape_offsets,
            size_t n_output_shape_offsets,
            const int64_t* output_shapes,
            size_t n_output_shapes,
            const int64_t* output_labels,
            size_t n_output_labels,
            uint64_t topology_revision,
            uint64_t numeric_revision,
            uint64_t route_topology_revision,
        ) except +
        cpp_bool replay_metric_boundary_from_split_site(
            const string& side,
            int64_t child_bond,
            uint64_t numeric_revision,
        ) except +
        vector[CppContextualCoreBlock] contextual_core(
            int64_t site,
            int64_t physical_output_charge,
            int64_t physical_input_charge,
            int32_t boundary_bra_two_j,
            int32_t boundary_ket_two_j,
            int32_t physical_bra_two_j,
            int32_t physical_ket_two_j,
            int32_t next_bra_two_j,
            int32_t next_ket_two_j,
            cpp_bool left,
            cpp_bool dual_right_basis,
        ) except +
        size_t boundary_value_count(
            const string& side,
            int64_t bond,
        ) except +
        void copy_boundary_values(
            const string& side,
            int64_t bond,
            double* output,
            size_t n_values,
        ) except +
        size_t metric_boundary_value_count(
            const string& side,
            int64_t bond,
        ) except +
        void copy_metric_boundary_values(
            const string& side,
            int64_t bond,
            double* output,
            size_t n_values,
        ) except +
        CppBoundaryBufferHandle* retain_boundary_buffer(
            const string& side,
            int64_t bond,
        ) except +
        CppBoundaryBufferHandle* retain_metric_boundary_buffer(
            const string& side,
            int64_t bond,
        ) except +
        cpp_bool install_local_operator(
            const string& key,
            const cpp_complex[double]* values,
            size_t n_values,
            const int64_t* value_offsets,
            const int64_t* rows,
            const int64_t* cols,
            const int64_t* input_starts,
            const int64_t* output_starts,
            size_t n_blocks,
            size_t dimension,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) except +
        void clear_local_operator()
        void local_matvec(
            const string& key,
            const cpp_complex[double]* input,
            cpp_complex[double]* output,
            size_t dimension,
        ) except +
        void local_diagonal(
            const string& key,
            cpp_complex[double]* output,
            size_t dimension,
        ) except +
        CppDavidsonResult local_davidson(
            const string& key,
            const cpp_complex[double]* diagonal,
            const cpp_complex[double]* guess,
            size_t dimension,
            double tolerance,
            int max_iterations,
            int restart_dimension,
            cpp_bool accept_unconverged,
        ) except +
        cpp_bool install_factor_routes(
            const string& key,
            const int64_t* in_indices,
            const int64_t* out_indices,
            const int64_t* left_indices,
            const int64_t* right_indices,
            size_t n_routes,
            const int64_t* basis_offsets,
            const int64_t* basis_shapes,
            size_t n_basis,
            const int64_t* left_factor_indices,
            size_t n_left_factors,
            const int64_t* left_offsets,
            const int64_t* left_shape_offsets,
            size_t n_left_pool,
            const int64_t* left_shapes,
            size_t n_left_shapes,
            const double* left_data,
            size_t n_left_data,
            const int64_t* right_factor_indices,
            size_t n_right_factors,
            const int64_t* right_offsets,
            const int64_t* right_shape_offsets,
            size_t n_right_pool,
            const int64_t* right_shapes,
            size_t n_right_shapes,
            const double* right_data,
            size_t n_right_data,
            size_t total_dimension,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) except +
        cpp_bool install_raw_factor_routes(
            const string& key,
            const void* in_indices,
            const void* out_indices,
            const void* left_indices,
            const void* right_indices,
            size_t n_routes,
            size_t route_basis_index_bytes,
            size_t route_factor_index_bytes,
            const int64_t* basis_offsets,
            const int64_t* basis_shapes,
            size_t n_basis,
            const int64_t* left_factor_indices,
            size_t n_left_factors,
            const int64_t* left_boundary_ids,
            const int64_t* left_w_ids,
            size_t n_left_raw_factors,
            const int64_t* left_boundary_offsets,
            const int64_t* left_boundary_shape_offsets,
            size_t n_left_boundary_arrays,
            const int64_t* left_boundary_shapes,
            size_t n_left_boundary_shape_values,
            const double* left_boundary_data,
            size_t n_left_boundary_data,
            const int64_t* left_w_offsets,
            const int64_t* left_w_shape_offsets,
            size_t n_left_w_arrays,
            const int64_t* left_w_shapes,
            size_t n_left_w_shape_values,
            const double* left_w_data,
            size_t n_left_w_data,
            const int64_t* right_factor_indices,
            size_t n_right_factors,
            const int64_t* right_boundary_ids,
            const int64_t* right_w_ids,
            size_t n_right_raw_factors,
            const int64_t* right_boundary_offsets,
            const int64_t* right_boundary_shape_offsets,
            size_t n_right_boundary_arrays,
            const int64_t* right_boundary_shapes,
            size_t n_right_boundary_shape_values,
            const double* right_boundary_data,
            size_t n_right_boundary_data,
            const int64_t* right_w_offsets,
            const int64_t* right_w_shape_offsets,
            size_t n_right_w_arrays,
            const int64_t* right_w_shapes,
            size_t n_right_w_shape_values,
            const double* right_w_data,
            size_t n_right_w_data,
            const void* left_family_masks,
            const void* right_family_masks,
            size_t family_mask_bytes,
            size_t total_dimension,
            uint64_t topology_revision,
            uint64_t numeric_revision,
            cpp_bool direct_actions,
            cpp_bool sources_preconfigured,
        ) except +
        cpp_bool install_contextual_factor_routes(
            const string& key,
            int64_t bond,
            int64_t left_boundary_bond,
            int64_t right_boundary_bond,
            const int64_t* basis_offsets,
            const int64_t* basis_shapes,
            const int64_t* basis_quantum_numbers,
            const int64_t* left_sector_ids,
            const int64_t* right_sector_ids,
            size_t n_basis,
            size_t total_dimension,
            uint64_t topology_revision,
            cpp_bool dual_right_basis,
        ) except +
        int prepare_active_bond_complementary_actions(
            int64_t left_boundary_bond,
            int64_t right_boundary_bond,
            size_t expected_basis,
            size_t expected_dimension,
            cpp_bool dual_right_basis,
        ) except +
        const string& factor_route_key() noexcept
        size_t prepare_active_bond_metric_routes(
            int64_t left_boundary_bond,
            int64_t right_boundary_bond,
        ) except +
        const string& metric_key() noexcept
        void clear_factor_routes()
        void set_factor_routes_hermitianized(cpp_bool enabled) noexcept
        void factor_route_matvec(
            const string& key,
            const cpp_complex[double]* input,
            cpp_complex[double]* output,
            size_t dimension,
        ) except +
        void factor_route_real_matvec(
            const string& key,
            const double* input,
            double* output,
            size_t dimension,
        ) except +
        void factor_route_diagonal(
            const string& key,
            double* output,
            size_t dimension,
        ) except +
        CppDavidsonResult factor_route_davidson(
            const string& key,
            const cpp_complex[double]* diagonal,
            const cpp_complex[double]* guess,
            size_t dimension,
            double tolerance,
            int max_iterations,
            int restart_dimension,
            cpp_bool accept_unconverged,
        ) except +
        CppDavidsonResult active_bond_complementary_davidson(
            const string& key,
            const cpp_complex[double]* guess,
            size_t dimension,
            double tolerance,
            int max_iterations,
            int restart_dimension,
            cpp_bool accept_unconverged,
        ) except +
        CppActiveBondCanonicalSolveResult solve_active_bond_canonical(
            const string& metric_key,
            double projection_tolerance,
            size_t max_component_elements,
            size_t max_transform_elements,
            double davidson_tolerance,
            int max_iterations,
            int requested_restart_dimension,
            size_t workspace_budget_bytes,
            size_t workspace_basis_arrays,
            cpp_bool accept_unconverged,
        ) except +
        CppActiveBondCanonicalSolveResult prepare_and_solve_active_bond_canonical(
            int64_t left_boundary_bond,
            int64_t right_boundary_bond,
            cpp_bool dual_right_basis,
            double projection_tolerance,
            size_t max_component_elements,
            size_t max_transform_elements,
            double davidson_tolerance,
            int max_iterations,
            int requested_restart_dimension,
            size_t workspace_budget_bytes,
            size_t workspace_basis_arrays,
            cpp_bool accept_unconverged,
        ) except +
        CppActiveBondStateAverageSolveResult prepare_and_solve_active_bond_state_average(
            int64_t left_boundary_bond,
            int64_t right_boundary_bond,
            cpp_bool dual_right_basis,
            double projection_tolerance,
            size_t max_component_elements,
            size_t max_transform_elements,
            double davidson_tolerance,
            int max_iterations,
            int requested_restart_dimension,
            size_t workspace_budget_bytes,
            size_t workspace_basis_arrays,
            cpp_bool accept_unconverged,
        ) except +
        cpp_bool active_bond_complementary_action_ready(
            const string& key,
            size_t dimension,
        ) noexcept
        cpp_bool factor_route_installed(
            const string& key,
            size_t dimension,
        ) noexcept
        cpp_bool begin_factor_route_projection(
            const string& key,
            const string& factor_route_key,
            size_t parent_dimension,
            size_t orthonormal_dimension,
            size_t n_components,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) except +
        void install_factor_route_projection_component(
            size_t component,
            const int64_t* parent_indices,
            size_t n_indices,
            const double* real_transform,
            const cpp_complex[double]* complex_transform,
            size_t transform_columns,
            size_t orthonormal_offset,
        ) except +
        void install_factor_route_projection_indexed_component(
            size_t component,
            const int64_t* parent_indices,
            size_t n_parent_indices,
            const int64_t* orthonormal_indices,
            size_t n_orthonormal_indices,
            const double* real_transform,
            const cpp_complex[double]* complex_transform,
        ) except +
        void install_factor_route_projection_kronecker_component(
            size_t component,
            size_t parent_offset,
            const int64_t* orthonormal_indices,
            size_t n_orthonormal_indices,
            size_t left_dim,
            size_t selected_dim,
            size_t local_dim,
            size_t right_dim,
            const double* real_transform,
            const cpp_complex[double]* complex_transform,
        ) except +
        void finish_factor_route_projection() except +
        void clear_factor_route_projection()
        void factor_route_projected_matvec(
            const string& key,
            const cpp_complex[double]* input,
            cpp_complex[double]* output,
            size_t dimension,
        ) except +
        void factor_route_projected_real_matvec(
            const string& key,
            const double* input,
            double* output,
            size_t dimension,
        ) except +
        CppDavidsonResult factor_route_projected_davidson(
            const string& key,
            const cpp_complex[double]* diagonal,
            const cpp_complex[double]* guess,
            size_t dimension,
            double tolerance,
            int max_iterations,
            int restart_dimension,
            cpp_bool accept_unconverged,
        ) except +
        cpp_bool begin_factorized_metric(
            const string& key,
            size_t dimension,
            size_t n_routes,
            uint64_t topology_revision,
            uint64_t numeric_revision,
        ) except +
        size_t install_contextual_metric_routes(
            const string& key,
            int64_t left_boundary_bond,
            int64_t right_boundary_bond,
            const int64_t* basis_offsets,
            const int64_t* basis_shapes,
            const int64_t* basis_quantum_numbers,
            const int64_t* left_sector_ids,
            const int64_t* right_sector_ids,
            size_t n_basis,
            size_t total_dimension,
            uint64_t topology_revision,
        ) except +
        void install_factorized_metric_route(
            size_t route,
            int64_t input_offset,
            int64_t output_offset,
            const int64_t* input_shape,
            const int64_t* output_shape,
            const double* left_real,
            const cpp_complex[double]* left_complex,
            const double* right_real,
            const cpp_complex[double]* right_complex,
        ) except +
        void finish_factorized_metric() except +
        void clear_factorized_metric()
        void factorized_metric_matvec(
            const string& key,
            const cpp_complex[double]* input,
            cpp_complex[double]* output,
            size_t dimension,
        ) except +
        void factorized_metric_real_matvec(
            const string& key,
            const double* input,
            double* output,
            size_t dimension,
        ) except +
        void factorized_metric_real_diagonal(
            const string& key,
            double* output,
            size_t dimension,
        ) except +
        CppCanonicalProjectionInfo prepare_canonical_reduced_projection(
            const string& metric_key,
            double tolerance,
            size_t max_component_elements,
            size_t max_transform_elements,
        ) except +
        void canonical_reduced_projection_guess(
            const string& projection_key,
            const string& metric_key,
            const cpp_complex[double]* parent_guess,
            cpp_complex[double]* orthonormal_guess,
            size_t parent_dimension,
            size_t orthonormal_dimension,
        ) except +
        void lift_factor_route_projection_vector(
            const string& projection_key,
            const cpp_complex[double]* orthonormal_vector,
            cpp_complex[double]* parent_vector,
            size_t orthonormal_dimension,
            size_t parent_dimension,
        ) except +
        CppDavidsonResult factor_route_generalized_davidson(
            const string& factor_route_key,
            const string& metric_key,
            const cpp_complex[double]* h_diagonal,
            const cpp_complex[double]* n_diagonal,
            const cpp_complex[double]* guess,
            size_t dimension,
            double energy_tolerance,
            double residual_tolerance,
            double linear_dependence_tolerance,
            int max_iterations,
            int restart_dimension,
            cpp_bool accept_unconverged,
        ) except +
        CppDavidsonResult active_bond_complementary_generalized_davidson(
            const string& factor_route_key,
            const string& metric_key,
            const cpp_complex[double]* guess,
            size_t dimension,
            double energy_tolerance,
            double residual_tolerance,
            double linear_dependence_tolerance,
            int max_iterations,
            int restart_dimension,
            cpp_bool accept_unconverged,
        ) except +
        CppDavidsonResult factor_route_projected_generalized_davidson(
            const string& projection_key,
            const string& metric_key,
            const cpp_complex[double]* h_diagonal,
            const cpp_complex[double]* n_diagonal,
            const cpp_complex[double]* guess,
            size_t dimension,
            double energy_tolerance,
            double residual_tolerance,
            double linear_dependence_tolerance,
            int max_iterations,
            int restart_dimension,
            cpp_bool accept_unconverged,
        ) except +
        CppBlockSVDResult blockwise_svd(
            const cpp_complex[double]* values,
            size_t n_values,
            const int64_t* value_offsets,
            const int64_t* rows,
            const int64_t* cols,
            const int64_t* state_weights,
            size_t n_blocks,
            double cutoff,
            int64_t max_bond,
            const string& max_bond_mode,
            cpp_bool retain_sector_topology,
        ) except +
        CppActiveBondSplitResult split_active_bond_solution(
            double cutoff,
            int64_t max_bond,
            const string& max_bond_mode,
            cpp_bool retain_sector_topology,
            const string& absorb,
            cpp_bool install_and_stage,
            cpp_bool retain_result_tensors,
        ) except +
        cpp_bool active_bond_solution_ready() noexcept
        vector[int64_t] sweep_bonds(
            const string& direction,
            int64_t n_sites,
        ) except +
        void begin_half_sweep(const string& direction, int64_t n_sites) except +
        void prepare_owned_half_sweep() except +
        int owned_half_sweep_readiness_code() noexcept
        cpp_bool owned_half_sweep_ready() noexcept
        vector[CppOwnedHalfSweepBondResult] execute_owned_half_sweep(
            double cutoff,
            int64_t max_bond,
            const string& max_bond_mode,
            cpp_bool retain_sector_topology,
            double projection_tolerance,
            size_t max_component_elements,
            size_t max_transform_elements,
            double davidson_tolerance,
            int max_iterations,
            int requested_restart_dimension,
            size_t workspace_budget_bytes,
            size_t workspace_basis_arrays,
            cpp_bool accept_unconverged,
        ) except +
        vector[CppOwnedSplitSiteExport] export_owned_split_sites() except +
        void release_workspaces() except +
        size_t execute_half_sweep(
            HalfSweepBondExecutor executor,
            void* context,
        ) except +
        int64_t claim_next_bond() except +
        void begin_bond(int64_t bond) except +
        void mark_bond_solved() except +
        void mark_bond_split(
            uint64_t kept_states,
            double truncation_seconds,
        ) except +
        void stage_bond_update(
            uint64_t kept_states,
            double truncation_seconds,
        ) except +
        void mark_bond_advanced() except +
        void commit_bond(
            uint64_t matvec_calls,
            uint64_t davidson_iterations,
            double matvec_seconds,
            double davidson_seconds,
            double energy,
        ) except +
        void commit_bond_update(
            uint64_t matvec_calls,
            uint64_t davidson_iterations,
            double matvec_seconds,
            double davidson_seconds,
            double energy,
        ) except +
        void record_bond(
            int64_t bond,
            uint64_t matvec_calls,
            uint64_t davidson_iterations,
            uint64_t kept_states,
            double matvec_seconds,
            double davidson_seconds,
            double truncation_seconds,
        ) except +
        void finish_half_sweep() except +
        void abort_half_sweep() noexcept
        uint64_t system_revision() noexcept
        uint64_t boundary_topology_builds() noexcept
        uint64_t boundary_numeric_refreshes() noexcept
        uint64_t boundary_reallocations() noexcept
        uint64_t boundary_update_topology_builds() noexcept
        uint64_t boundary_update_calls() noexcept
        uint64_t boundary_update_routes() noexcept
        double boundary_update_seconds() noexcept
        size_t normal_complementary_boundary_action_count() noexcept
        size_t normal_complementary_boundary_action_bytes() noexcept
        size_t metric_boundary_count() noexcept
        size_t metric_boundary_action_count() noexcept
        size_t metric_boundary_action_bytes() noexcept
        uint64_t half_sweeps() noexcept
        uint64_t half_sweep_executor_calls() noexcept
        uint64_t half_sweep_executor_bonds() noexcept
        uint64_t half_sweep_python_bond_callbacks() noexcept
        double half_sweep_executor_seconds() noexcept
        uint64_t owned_half_sweep_calls() noexcept
        uint64_t owned_half_sweep_bonds() noexcept
        double owned_half_sweep_seconds() noexcept
        uint64_t aborted_half_sweeps() noexcept
        uint64_t bond_steps() noexcept
        uint64_t bond_prepares() noexcept
        uint64_t bond_solves() noexcept
        uint64_t bond_splits() noexcept
        uint64_t bond_advances() noexcept
        uint64_t staged_bond_updates() noexcept
        uint64_t committed_bond_updates() noexcept
        uint64_t matvec_calls() noexcept
        uint64_t davidson_iterations() noexcept
        uint64_t kept_states() noexcept
        double matvec_seconds() noexcept
        double davidson_seconds() noexcept
        double truncation_seconds() noexcept
        double last_half_sweep_energy() noexcept
        size_t boundary_count() noexcept
        size_t borrowed_boundary_bytes() noexcept
        size_t owned_boundary_bytes() noexcept
        size_t local_operator_blocks() noexcept
        size_t borrowed_local_operator_bytes() noexcept
        size_t factor_route_count() noexcept
        size_t factor_route_table_bytes() noexcept
        size_t borrowed_factor_pool_bytes() noexcept
        size_t factor_route_scratch_bytes() noexcept
        size_t borrowed_raw_factor_source_bytes() noexcept
        size_t raw_factor_cache_bytes() noexcept
        size_t peak_raw_factor_cache_bytes() noexcept
        size_t factor_route_projection_components() noexcept
        size_t factor_route_projection_index_bytes() noexcept
        size_t borrowed_factor_route_transform_bytes() noexcept
        size_t factor_route_projection_scratch_bytes() noexcept
        size_t davidson_workspace_bytes() noexcept
        size_t peak_borrowed_local_operator_bytes() noexcept
        size_t peak_factor_route_table_bytes() noexcept
        size_t peak_borrowed_factor_pool_bytes() noexcept
        size_t peak_factor_route_scratch_bytes() noexcept
        size_t peak_factor_route_projection_index_bytes() noexcept
        size_t peak_borrowed_factor_route_transform_bytes() noexcept
        size_t peak_factor_route_projection_scratch_bytes() noexcept
        uint64_t local_topology_builds() noexcept
        uint64_t local_numeric_refreshes() noexcept
        uint64_t local_matvec_calls() noexcept
        uint64_t local_davidson_calls() noexcept
        uint64_t local_davidson_workspace_reuses() noexcept
        uint64_t factor_route_topology_builds() noexcept
        uint64_t factor_route_numeric_refreshes() noexcept
        uint64_t factor_route_matvec_calls() noexcept
        uint64_t real_factor_route_matvec_calls() noexcept
        uint64_t factor_route_diagonal_calls() noexcept
        uint64_t factor_route_davidson_calls() noexcept
        uint64_t factor_route_scratch_growths() noexcept
        uint64_t contextual_route_plan_builds() noexcept
        uint64_t contextual_route_plan_hits() noexcept
        uint64_t contextual_route_plan_shape_refreshes() noexcept
        uint64_t decomposed_action_plan_builds() noexcept
        uint64_t decomposed_action_plan_hits() noexcept
        uint64_t decomposed_action_plan_rebuilds() noexcept
        size_t complementary_execution_graph_bytes() noexcept
        uint64_t complementary_execution_graph_builds() noexcept
        uint64_t complementary_execution_graph_hits() noexcept
        size_t contextual_route_plan_count() noexcept
        size_t contextual_route_plan_bytes() noexcept
        size_t contextual_route_index_bytes() noexcept
        size_t contextual_route_core_value_bytes() noexcept
        size_t contextual_compiled_schedule_bytes() noexcept
        uint64_t contextual_compiled_schedule_builds() noexcept
        uint64_t contextual_compiled_schedule_hits() noexcept
        double contextual_compiled_schedule_restore_seconds() noexcept
        size_t contextual_route_core_elements() noexcept
        size_t contextual_route_core_nonzero_elements() noexcept
        size_t contextual_core_cache_count() noexcept
        size_t contextual_core_cache_bytes() noexcept
        uint64_t contextual_core_cache_hits() noexcept
        uint64_t contextual_core_reuse_hits() noexcept
        double contextual_route_match_seconds() noexcept
        double contextual_route_activation_seconds() noexcept
        double contextual_core_build_seconds() noexcept
        double contextual_core_reuse_seconds() noexcept
        double raw_route_setup_seconds() noexcept
        double raw_route_group_seconds() noexcept
        double dense_pair_build_seconds() noexcept
        double fused_factor_build_seconds() noexcept
        double raw_execution_build_seconds() noexcept
        double raw_factor_matvec_seconds() noexcept
        double raw_input_pack_seconds() noexcept
        double dense_pair_matvec_seconds() noexcept
        double raw_execution_matvec_seconds() noexcept
        double raw_execution_pack_seconds() noexcept
        double raw_pointer_execution_matvec_seconds() noexcept
        uint64_t raw_pointer_execution_matvec_calls() noexcept
        double direct_complementary_action_seconds() noexcept
        uint64_t direct_complementary_action_calls() noexcept
        uint64_t direct_complementary_actions() noexcept
        double factorized_metric_matvec_seconds() noexcept
        uint64_t contextual_zero_core_cache_hits() noexcept
        size_t contextual_zero_core_cache_count() noexcept
        size_t contextual_zero_core_cache_bytes() noexcept
        cpp_bool raw_factor_routes() noexcept
        cpp_bool factor_routes_hermitianized() noexcept
        uint64_t raw_factor_cache_hits() noexcept
        uint64_t raw_factor_cache_misses() noexcept
        uint64_t raw_factor_gemm_calls() noexcept
        uint64_t raw_output_product_calls() noexcept
        uint64_t direct_source_factor_loads() noexcept
        size_t compact_right_panel_count() noexcept
        size_t compact_right_panel_value_bytes() noexcept
        size_t compact_right_panel_product_count() noexcept
        size_t compact_right_panel_batch_count() noexcept
        size_t compact_right_panel_budget_bytes() noexcept
        uint64_t compact_right_panel_registry_builds() noexcept
        uint64_t compact_right_panel_numeric_refreshes() noexcept
        uint64_t compact_right_panel_matvec_batches() noexcept
        uint64_t compact_right_panel_matvec_products() noexcept
        uint64_t complementary_family_route_count(int family_id) except +
        uint64_t unlabeled_family_route_count() noexcept
        double raw_factor_build_seconds() noexcept
        size_t raw_route_group_count() noexcept
        cpp_bool complementary_local_actions() noexcept
        size_t complementary_local_action_count() noexcept
        size_t complementary_local_term_count() noexcept
        size_t complementary_local_action_bytes() noexcept
        size_t fused_raw_route_group_count() noexcept
        size_t fused_raw_route_count() noexcept
        size_t dense_pair_kernel_count() noexcept
        size_t dense_pair_execution_count() noexcept
        size_t dense_pair_kernel_elements() noexcept
        size_t dense_pair_route_count() noexcept
        size_t dense_factor_pack_bytes() noexcept
        uint64_t dense_factor_pack_builds() noexcept
        uint64_t dense_factor_pack_reuses() noexcept
        size_t raw_execution_group_count() noexcept
        size_t raw_execution_action_count() noexcept
        size_t right_grouped_execution_action_count() noexcept
        size_t peak_right_grouped_execution_action_count() noexcept
        size_t peak_raw_execution_group_count() noexcept
        size_t peak_raw_execution_action_count() noexcept
        size_t raw_input_superchannel_count() noexcept
        size_t raw_input_superchannel_tile_count() noexcept
        size_t raw_input_superchannel_batch_count() noexcept
        size_t peak_raw_input_superchannel_batch_count() noexcept
        size_t peak_raw_channel_unique_left_count() noexcept
        size_t peak_raw_channel_left_occurrence_count() noexcept
        size_t peak_raw_shared_left_panel_count() noexcept
        size_t peak_raw_shared_left_occurrence_count() noexcept
        size_t peak_raw_output_fusion_wave_count() noexcept
        size_t peak_raw_output_fusion_group_count() noexcept
        size_t peak_raw_output_fusion_tile_count() noexcept
        size_t peak_raw_output_fusion_workspace_bytes() noexcept
        uint64_t raw_output_fusion_gemm_calls() noexcept
        uint64_t raw_output_fusion_copied_elements() noexcept
        cpp_bool grouped_output_product_backend() noexcept
        size_t grouped_output_product_group_count() noexcept
        size_t grouped_output_product_binding_count() noexcept
        size_t peak_grouped_output_candidate_binding_count() noexcept
        uint64_t peak_grouped_output_candidate_work_count(int bin) except +
        uint64_t grouped_output_product_batch_calls() noexcept
        uint64_t grouped_output_products() noexcept
        size_t peak_raw_shared_right_panel_count() noexcept
        size_t peak_raw_shared_right_binding_count() noexcept
        size_t peak_raw_shared_right_workspace_bytes() noexcept
        uint64_t raw_shared_right_gemm_calls() noexcept
        uint64_t raw_shared_right_copied_elements() noexcept
        cpp_bool reduced_contextual_routes() noexcept
        size_t reduced_contextual_execution_count() noexcept
        size_t reduced_contextual_matrix_elements() noexcept
        size_t peak_reduced_contextual_execution_count() noexcept
        size_t peak_reduced_contextual_matrix_elements() noexcept
        size_t peak_borrowed_reduced_contextual_right_elements() noexcept
        size_t complementary_execution_slab_bytes() noexcept
        size_t complementary_execution_slab_capacity_bytes() noexcept
        size_t complementary_execution_slab_budget_bytes() noexcept
        size_t complementary_execution_slab_required_bytes() noexcept
        size_t peak_complementary_execution_slab_required_bytes() noexcept
        size_t peak_complementary_left_required_bytes() noexcept
        size_t peak_complementary_right_required_bytes() noexcept
        size_t peak_complementary_left_cached_bytes() noexcept
        size_t peak_complementary_right_cached_bytes() noexcept
        uint64_t complementary_execution_slab_full_prepares() noexcept
        uint64_t complementary_execution_slab_partial_prepares() noexcept
        uint64_t complementary_execution_slab_matvec_repacks() noexcept
        int64_t peak_reduced_contextual_boundary_rank() noexcept
        uint64_t reduced_contextual_fallbacks() noexcept
        int32_t reduced_contextual_fallback_reason() noexcept
        double reduced_contextual_fallback_residual_norm() noexcept
        double reduced_contextual_fallback_boundary_norm() noexcept
        double reduced_contextual_build_seconds() noexcept
        double reduced_contextual_numeric_refresh_seconds() noexcept
        double reduced_contextual_execution_refresh_seconds() noexcept
        double reduced_contextual_diagonal_seconds() noexcept
        double reduced_contextual_matvec_seconds() noexcept
        uint64_t factor_route_projection_topology_builds() noexcept
        uint64_t factor_route_projection_numeric_refreshes() noexcept
        uint64_t factor_route_projected_matvec_calls() noexcept
        uint64_t factor_route_projected_davidson_calls() noexcept
        uint64_t factor_route_generalized_davidson_calls() noexcept
        uint64_t real_generalized_davidson_calls() noexcept
        uint64_t factorized_metric_matvec_calls() noexcept
        uint64_t real_factorized_metric_matvec_calls() noexcept
        uint64_t canonical_projection_builds() noexcept
        uint64_t canonical_projection_reuses() noexcept
        uint64_t canonical_projection_davidson_calls() noexcept
        size_t canonical_projection_transform_elements() noexcept
        size_t canonical_projection_max_component_dimension() noexcept
        size_t canonical_projection_cache_entries() noexcept
        size_t canonical_projection_cache_transform_elements() noexcept
        uint64_t canonical_projection_cache_evictions() noexcept
        double canonical_projection_whitening_residual() noexcept
        double canonical_projection_build_seconds() noexcept
        uint64_t block_svd_calls() noexcept
        uint64_t block_svd_blocks() noexcept
        uint64_t block_svd_workspace_growths() noexcept
        double block_svd_seconds() noexcept
        size_t block_svd_workspace_bytes() noexcept
        uint64_t split_site_owner_revision() noexcept
        uint64_t split_site_installs() noexcept
        uint64_t split_site_topology_builds() noexcept
        uint64_t split_site_boundary_uses() noexcept
        uint64_t cached_boundary_replays() noexcept
        size_t split_site_count() noexcept
        size_t split_site_bytes() noexcept
        uint64_t site_merge_calls() noexcept
        uint64_t site_merge_blocks() noexcept
        double site_merge_seconds() noexcept
        size_t site_merge_bytes() noexcept
        uint64_t active_bond_complementary_prepares() noexcept
        uint64_t active_bond_complementary_fallbacks() noexcept
        int active_bond_complementary_fallback_reason() noexcept
        int64_t active_bond_complementary_fallback_bond() noexcept
        size_t active_bond_complementary_basis() noexcept
        size_t active_bond_complementary_dimension() noexcept
        size_t active_bond_complementary_expected_basis() noexcept
        size_t active_bond_complementary_expected_dimension() noexcept
        uint64_t active_bond_complementary_davidson_calls() noexcept
        uint64_t active_bond_complementary_generalized_davidson_calls() noexcept
        uint64_t active_bond_metric_prepares() noexcept
        uint64_t active_bond_cpp_splits() noexcept
        size_t factorized_metric_route_bytes() noexcept
        size_t factorized_metric_scratch_bytes() noexcept
        size_t memory_bytes() noexcept
        int lifecycle_phase_code() noexcept
        int64_t active_bond() noexcept


def reduced_environment_recoupling(
    object side,
    long long physical_output_charge,
    long long physical_input_charge,
    int boundary_bra_two_j,
    int boundary_ket_two_j,
    int physical_bra_two_j,
    int physical_ket_two_j,
    int next_bra_two_j,
    int next_ket_two_j,
    int left_channel_two_j,
    int right_channel_two_j,
    int operator_two_j,
    int left_channel_two_m,
    int right_channel_two_m,
    int operator_two_m,
):
    """Return one C++ reduced-environment angular recoupling factor."""

    cdef str normalized_side = str(side).lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    return reduced_environment_recoupling_coefficient(
        <cpp_bool>(normalized_side == "left"),
        <int64_t>physical_output_charge,
        <int64_t>physical_input_charge,
        <int32_t>boundary_bra_two_j,
        <int32_t>boundary_ket_two_j,
        <int32_t>physical_bra_two_j,
        <int32_t>physical_ket_two_j,
        <int32_t>next_bra_two_j,
        <int32_t>next_ket_two_j,
        <int32_t>left_channel_two_j,
        <int32_t>right_channel_two_j,
        <int32_t>operator_two_j,
        <int32_t>left_channel_two_m,
        <int32_t>right_channel_two_m,
        <int32_t>operator_two_m,
    )


cdef class _SU2BoundaryBuffer:
    """Lifetime owner for a NumPy view of one C++ reduced boundary arena."""

    cdef CppBoundaryBufferHandle* _handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            release_boundary_buffer(self._handle)
            self._handle = NULL


cdef class SU2MovingEnvironment:
    """Persistent C++ owner for active integrals and packed SU(2) sweep state."""

    cdef CppSystem* _system
    cdef CppMovingEnvironment* _engine
    cdef dict _boundary_value_owners
    cdef object _local_operator_owner
    cdef object _factor_route_owners
    cdef object _factor_route_projection_owners
    cdef object _factorized_metric_owners
    cdef object _factor_route_key
    cdef object _factor_route_revision
    cdef dict _split_site_keys
    cdef dict _split_site_key_indices
    cdef dict _split_site_topologies
    cdef dict _split_site_revisions
    cdef unsigned long long _split_site_topology_clock
    cdef unsigned long long _split_site_numeric_clock

    def __cinit__(
        self,
        object h1,
        object eri,
        long long n_elec,
        long long two_s=0,
        double ecore=0.0,
        object orb_sym=None,
        double cutoff=1.0e-10,
        bint include_half=True,
    ):
        cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] h1_arr
        cdef cnp.ndarray[cnp.double_t, ndim=4, mode="c"] eri_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] orb_arr
        cdef Py_ssize_t n_sites

        self._system = NULL
        self._engine = NULL
        self._boundary_value_owners = {}
        self._local_operator_owner = None
        self._factor_route_owners = None
        self._factor_route_projection_owners = None
        self._factorized_metric_owners = None
        self._factor_route_key = None
        self._factor_route_revision = None
        self._split_site_keys = {}
        self._split_site_key_indices = {}
        self._split_site_topologies = {}
        self._split_site_revisions = {}
        self._split_site_topology_clock = 0
        self._split_site_numeric_clock = 0
        if np.iscomplexobj(h1) or np.iscomplexobj(eri):
            raise TypeError("SU2MovingEnvironment currently requires real active integrals.")
        h1_arr = np.ascontiguousarray(h1, dtype=np.float64)
        eri_arr = np.ascontiguousarray(eri, dtype=np.float64)
        if h1_arr.shape[0] != h1_arr.shape[1]:
            raise ValueError("h1 must be a square spatial-orbital matrix.")
        n_sites = h1_arr.shape[0]
        if (
            eri_arr.shape[0] != n_sites
            or eri_arr.shape[1] != n_sites
            or eri_arr.shape[2] != n_sites
            or eri_arr.shape[3] != n_sites
        ):
            raise ValueError("eri must have shape (n_sites, n_sites, n_sites, n_sites).")
        if orb_sym is None:
            orb_arr = np.zeros(n_sites, dtype=np.int64)
        else:
            orb_arr = np.ascontiguousarray(orb_sym, dtype=np.int64)
            if orb_arr.shape[0] != n_sites:
                raise ValueError("orb_sym must contain one label per active orbital.")
        self._system = new CppSystem(
            <const double*>cnp.PyArray_DATA(h1_arr),
            <const double*>cnp.PyArray_DATA(eri_arr),
            <size_t>n_sites,
            ecore,
            <int64_t>n_elec,
            <int64_t>two_s,
            <const int64_t*>cnp.PyArray_DATA(orb_arr),
            cutoff,
            <cpp_bool>include_half,
        )
        self._engine = new CppMovingEnvironment(self._system)

    def __dealloc__(self):
        if self._engine != NULL:
            del self._engine
            self._engine = NULL
        if self._system != NULL:
            del self._system
            self._system = NULL

    @property
    def system_stats(self):
        return {
            "kind": "su2_system",
            "backend": "cpp",
            "n_sites": int(self._system.n_sites()),
            "n_elec": int(self._system.n_elec()),
            "two_s": int(self._system.two_s()),
            "ecore": float(self._system.ecore()),
            "cutoff": float(self._system.cutoff()),
            "include_half": bool(self._system.include_half()),
            "revision": int(self._system.revision()),
            "stored_integral_elements": int(self._system.stored_integral_elements()),
            "stored_family_terms": int(self._system.stored_family_terms()),
            "normal_complementary_transition_count": int(
                self._system.normal_complementary_transition_count()
            ),
            "normal_complementary_component_action_count": int(
                self._system.normal_complementary_component_action_count()
            ),
            "normal_complementary_memory_bytes": int(
                self._system.normal_complementary_memory_bytes()
            ),
            "normal_complementary_primitive_bytes": int(
                self._system.normal_complementary_primitive_bytes()
            ),
            "memory_bytes": int(self._system.memory_bytes()),
        }

    def normal_complementary_plan(self, long long site):
        """Return one C++-owned SU(2) NC transition table for diagnostics.

        This export is intentionally a debug/validation view.  Production
        sweeps retain the immutable table in the C++ system owner and consume
        its route identifiers directly rather than rebuilding Python terms.
        """
        cdef const CppNormalComplementaryPlan* plan = (
            &self._system.normal_complementary_plan(<int64_t>site)
        )
        cdef Py_ssize_t n_terms = plan.transitions.size()
        cdef Py_ssize_t n_actions = plan.component_actions.size()
        cdef Py_ssize_t pos
        cdef cnp.ndarray[cnp.int64_t, ndim=2] left_quantum_numbers = np.empty(
            (plan.left_channels, 3), dtype=np.int64
        )
        cdef cnp.ndarray[cnp.int64_t, ndim=2] right_quantum_numbers = np.empty(
            (plan.right_channels, 3), dtype=np.int64
        )
        cdef cnp.ndarray[cnp.int64_t, ndim=1] left_component_offsets = np.empty(
            plan.left_channels + 1, dtype=np.int64
        )
        cdef cnp.ndarray[cnp.int64_t, ndim=1] right_component_offsets = np.empty(
            plan.right_channels + 1, dtype=np.int64
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] source = np.empty(
            n_terms, dtype=np.int32
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] target = np.empty(
            n_terms, dtype=np.int32
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] operator_ids = np.empty(
            n_terms, dtype=np.int32
        )
        cdef cnp.ndarray[cnp.int64_t, ndim=1] first = np.empty(
            n_terms, dtype=np.int64
        )
        cdef cnp.ndarray[cnp.int64_t, ndim=1] second = np.empty(
            n_terms, dtype=np.int64
        )
        cdef cnp.ndarray[cnp.double_t, ndim=1] coefficients = np.empty(
            n_terms, dtype=np.float64
        )
        cdef cnp.ndarray[cnp.uint64_t, ndim=1] family_masks = np.empty(
            n_terms, dtype=np.uint64
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] action_transition = np.empty(
            n_actions, dtype=np.int32
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] action_source_component = np.empty(
            n_actions, dtype=np.int32
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] action_target_component = np.empty(
            n_actions, dtype=np.int32
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] action_local_two_m = np.empty(
            n_actions, dtype=np.int32
        )
        cdef cnp.ndarray[cnp.double_t, ndim=1] action_coefficient = np.empty(
            n_actions, dtype=np.float64
        )
        for pos in range(plan.left_channels):
            left_quantum_numbers[pos, 0] = (
                plan.left_channel_quantum_numbers[pos].charge
            )
            left_quantum_numbers[pos, 1] = (
                plan.left_channel_quantum_numbers[pos].two_j
            )
            left_quantum_numbers[pos, 2] = (
                plan.left_channel_quantum_numbers[pos].point_group
            )
            left_component_offsets[pos] = plan.left_component_offsets[pos]
        left_component_offsets[plan.left_channels] = (
            plan.left_component_offsets[plan.left_channels]
        )
        for pos in range(plan.right_channels):
            right_quantum_numbers[pos, 0] = (
                plan.right_channel_quantum_numbers[pos].charge
            )
            right_quantum_numbers[pos, 1] = (
                plan.right_channel_quantum_numbers[pos].two_j
            )
            right_quantum_numbers[pos, 2] = (
                plan.right_channel_quantum_numbers[pos].point_group
            )
            right_component_offsets[pos] = plan.right_component_offsets[pos]
        right_component_offsets[plan.right_channels] = (
            plan.right_component_offsets[plan.right_channels]
        )
        for pos in range(n_terms):
            source[pos] = plan.transitions[pos].source
            target[pos] = plan.transitions[pos].target
            operator_ids[pos] = <int>plan.transitions[pos].local_operator
            first[pos] = plan.transitions[pos].first_index
            second[pos] = plan.transitions[pos].second_index
            coefficients[pos] = plan.transitions[pos].coefficient
            family_masks[pos] = plan.transitions[pos].family_mask
        for pos in range(n_actions):
            action_transition[pos] = plan.component_actions[pos].transition
            action_source_component[pos] = (
                plan.component_actions[pos].source_component
            )
            action_target_component[pos] = (
                plan.component_actions[pos].target_component
            )
            action_local_two_m[pos] = plan.component_actions[pos].local_two_m
            action_coefficient[pos] = plan.component_actions[pos].coefficient
        return {
            "site": int(plan.site),
            "left_channels": int(plan.left_channels),
            "right_channels": int(plan.right_channels),
            "left_channel_quantum_numbers": left_quantum_numbers,
            "right_channel_quantum_numbers": right_quantum_numbers,
            "left_component_offsets": left_component_offsets,
            "right_component_offsets": right_component_offsets,
            "source": source,
            "target": target,
            "operator": operator_ids,
            "first_index": first,
            "second_index": second,
            "coefficient": coefficients,
            "family_mask": family_masks,
            "component_transition": action_transition,
            "component_source": action_source_component,
            "component_target": action_target_component,
            "component_local_two_m": action_local_two_m,
            "component_coefficient": action_coefficient,
            "family_transition_counts": {
                "S": int(plan.family_transition_counts[0]),
                "R": int(plan.family_transition_counts[1]),
                "A": int(plan.family_transition_counts[2]),
                "P": int(plan.family_transition_counts[3]),
                "B": int(plan.family_transition_counts[4]),
                "Q": int(plan.family_transition_counts[5]),
            },
        }

    def normal_complementary_primitives(self, long long site):
        """Return a diagnostic copy of the C++-owned local primitive table."""

        cdef size_t count = self._system.normal_complementary_primitive_count()
        cdef const double* source = (
            self._system.normal_complementary_primitives(<int64_t>site)
        )
        cdef cnp.ndarray[cnp.double_t, ndim=4] result = np.empty(
            (3, 3, 20, 5),
            dtype=np.float64,
        )
        if count != <size_t>result.size:
            raise RuntimeError("C++ normal/complementary primitive size is invalid.")
        memcpy(
            <void*>cnp.PyArray_DATA(result),
            <const void*>source,
            count * sizeof(double),
        )
        return result

    def contextual_core(
        self,
        long long site,
        long long physical_output_charge,
        long long physical_input_charge,
        int boundary_bra_two_j,
        int boundary_ket_two_j,
        int physical_bra_two_j,
        int physical_ket_two_j,
        int next_bra_two_j,
        int next_ket_two_j,
        bint left=True,
        bint dual_right_basis=True,
    ):
        """Return one fusion-contextual reduced NC core from the C++ system."""

        cdef vector[CppContextualCoreBlock] blocks = self._engine.contextual_core(
            <int64_t>site,
            <int64_t>physical_output_charge,
            <int64_t>physical_input_charge,
            <int32_t>boundary_bra_two_j,
            <int32_t>boundary_ket_two_j,
            <int32_t>physical_bra_two_j,
            <int32_t>physical_ket_two_j,
            <int32_t>next_bra_two_j,
            <int32_t>next_ket_two_j,
            <cpp_bool>left,
            <cpp_bool>dual_right_basis,
        )
        cdef dict result = {}
        cdef Py_ssize_t index
        cdef CppContextualCoreBlock block
        cdef cnp.ndarray[cnp.double_t, ndim=4] values
        for index in range(blocks.size()):
            block = blocks[index]
            values = np.empty(
                (block.rows, block.cols, 1, 1),
                dtype=np.float64,
            )
            if block.values.size() != <size_t>(block.rows * block.cols):
                raise RuntimeError("C++ contextual NC core has an invalid shape.")
            memcpy(
                <void*>cnp.PyArray_DATA(values),
                <const void*>block.values.data(),
                block.values.size() * sizeof(double),
            )
            result[(int(block.left_channel), int(block.right_channel))] = values
        return result

    def apply_normal_complementary_core(
        self,
        long long site,
        object primitive_components,
        object input_values,
    ):
        """Apply one C++-owned NC local core to a batch of channel vectors."""

        cdef object raw_input = np.asarray(input_values)
        cdef bint vector_input = raw_input.ndim == 1
        cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] primitive_arr
        cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] input_arr
        cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] output_arr
        cdef size_t left_components = (
            self._system.normal_complementary_left_component_count(
                <int64_t>site
            )
        )
        cdef size_t right_components = (
            self._system.normal_complementary_right_component_count(
                <int64_t>site
            )
        )
        cdef size_t batch_size

        primitive_arr = np.ascontiguousarray(
            primitive_components,
            dtype=np.float64,
        ).reshape((20, 5))
        if raw_input.ndim not in {1, 2}:
            raise ValueError("input_values must be a vector or rank-2 batch.")
        input_arr = np.ascontiguousarray(
            raw_input,
            dtype=np.float64,
        ).reshape((left_components, -1))
        batch_size = <size_t>input_arr.shape[1]
        output_arr = np.empty(
            (right_components, batch_size),
            dtype=np.float64,
        )
        self._system.apply_normal_complementary_core(
            <int64_t>site,
            <const double*>cnp.PyArray_DATA(primitive_arr),
            <size_t>primitive_arr.size,
            <const double*>cnp.PyArray_DATA(input_arr),
            <size_t>input_arr.size,
            batch_size,
            <double*>cnp.PyArray_DATA(output_arr),
            <size_t>output_arr.size,
        )
        return output_arr[:, 0] if vector_input else output_arr

    def family(self, object name):
        cdef dict family_ids = {"S": 0, "R": 1, "A": 2, "P": 3, "B": 4, "Q": 5}
        cdef str normalized = str(name).upper()
        cdef int family_id
        cdef const CppFamilyData* family
        cdef Py_ssize_t n_terms
        cdef Py_ssize_t rank
        cdef Py_ssize_t pos
        cdef cnp.ndarray[cnp.int64_t, ndim=2] indices
        cdef cnp.ndarray[cnp.double_t, ndim=1] values
        if normalized not in family_ids:
            raise KeyError(f"Unknown SU(2) complementary family {name!r}.")
        family_id = family_ids[normalized]
        family = &self._system.family(family_id)
        rank = family.rank
        n_terms = family.values.size()
        indices = np.empty((n_terms, rank), dtype=np.int64)
        values = np.empty(n_terms, dtype=np.float64)
        for pos in range(n_terms):
            values[pos] = family.values[pos]
        for pos in range(n_terms * rank):
            indices[pos // rank, pos % rank] = family.indices[pos]
        return {
            "name": normalized,
            "rank": int(rank),
            "indices": indices,
            "values": values,
        }

    def family_partition_counts(self, object name, object side, long long bond):
        cdef dict family_ids = {"S": 0, "R": 1, "A": 2, "P": 3, "B": 4, "Q": 5}
        cdef str normalized = str(name).upper()
        cdef str normalized_side = str(side).lower()
        cdef size_t internal = 0
        cdef size_t cross = 0
        cdef size_t external = 0
        if normalized not in family_ids:
            raise KeyError(f"Unknown SU(2) complementary family {name!r}.")
        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        self._system.family_partition_counts(
            family_ids[normalized],
            <cpp_bool>(normalized_side == "left"),
            <int64_t>bond,
            &internal,
            &cross,
            &external,
        )
        return int(internal), int(cross), int(external)

    def install_boundary(
        self,
        object side,
        long long bond,
        object values,
        object offsets,
        object labels,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] value_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] label_arr
        cdef string side_name = str(side).encode()
        cdef object owner_key = (str(side).lower(), int(bond))
        cdef bint same_topology
        value_arr = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
        offset_arr = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
        label_arr = np.ascontiguousarray(labels, dtype=np.int64).reshape(-1)
        same_topology = self._engine.install_boundary(
            side_name,
            <int64_t>bond,
            <const double*>cnp.PyArray_DATA(value_arr),
            <size_t>value_arr.size,
            <const int64_t*>cnp.PyArray_DATA(offset_arr),
            <size_t>offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(label_arr),
            <size_t>label_arr.size,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        )
        self._boundary_value_owners[owner_key] = value_arr
        return bool(same_topology)

    def install_metric_boundary(
        self,
        object side,
        long long bond,
        object values,
        object offsets,
        object labels,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] value_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] label_arr
        cdef string side_name = str(side).encode()
        cdef object owner_key = ("metric", str(side).lower(), int(bond))
        cdef bint same_topology
        value_arr = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
        offset_arr = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
        label_arr = np.ascontiguousarray(labels, dtype=np.int64).reshape(-1)
        same_topology = self._engine.install_metric_boundary(
            side_name,
            <int64_t>bond,
            <const double*>cnp.PyArray_DATA(value_arr),
            <size_t>value_arr.size,
            <const int64_t*>cnp.PyArray_DATA(offset_arr),
            <size_t>offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(label_arr),
            <size_t>label_arr.size,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        )
        self._boundary_value_owners[owner_key] = value_arr
        return bool(same_topology)

    def boundary_installed(
        self,
        object side,
        long long bond,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef string side_name = str(side).lower().encode()
        return bool(self._engine.boundary_installed(
            side_name,
            <int64_t>bond,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        ))

    def metric_boundary_installed(
        self,
        object side,
        long long bond,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef string side_name = str(side).lower().encode()
        return bool(self._engine.metric_boundary_installed(
            side_name,
            <int64_t>bond,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        ))

    def advance_boundary(
        self,
        object side,
        long long parent_bond,
        long long child_bond,
        object routes,
        object bra_data,
        object bra_offsets,
        object bra_shape_offsets,
        object bra_shapes,
        object ket_data,
        object ket_offsets,
        object ket_shape_offsets,
        object ket_shapes,
        object mpo_data,
        object mpo_offsets,
        object mpo_shape_offsets,
        object mpo_shapes,
        object output_offsets,
        object output_shape_offsets,
        object output_shapes,
        object output_labels,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
        bint metric_boundary=False,
        object route_coefficients=None,
    ):
        """Advance one owned reduced boundary from a packed recursive route batch."""

        cdef str normalized_side = str(side).lower()
        cdef string side_name = normalized_side.encode()
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] route_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] route_coefficient_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] bra_data_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_shape_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] ket_data_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_shape_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] mpo_data_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] mpo_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] mpo_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] mpo_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_label_arr
        cdef cnp.ndarray result
        cdef cnp.npy_intp result_shape[1]
        cdef CppBoundaryBufferHandle* buffer_handle = NULL
        cdef _SU2BoundaryBuffer buffer_owner
        cdef size_t n_values
        cdef bint same_topology
        cdef object owner_key = (
            "metric" if metric_boundary else "hamiltonian",
            normalized_side,
            int(child_bond),
        )

        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        route_arr = np.ascontiguousarray(routes, dtype=np.int64).reshape((-1, 5))
        route_coefficient_arr = np.ascontiguousarray(
            (
                np.ones(route_arr.shape[0], dtype=np.float64)
                if route_coefficients is None
                else route_coefficients
            ),
            dtype=np.float64,
        ).reshape(-1)
        if route_coefficient_arr.size != route_arr.shape[0]:
            raise ValueError(
                "route_coefficients must match the boundary route count."
            )
        bra_data_arr = np.ascontiguousarray(bra_data, dtype=np.float64).reshape(-1)
        bra_offset_arr = np.ascontiguousarray(bra_offsets, dtype=np.int64).reshape(-1)
        bra_shape_offset_arr = np.ascontiguousarray(
            bra_shape_offsets, dtype=np.int64
        ).reshape(-1)
        bra_shape_arr = np.ascontiguousarray(bra_shapes, dtype=np.int64).reshape(-1)
        ket_data_arr = np.ascontiguousarray(ket_data, dtype=np.float64).reshape(-1)
        ket_offset_arr = np.ascontiguousarray(ket_offsets, dtype=np.int64).reshape(-1)
        ket_shape_offset_arr = np.ascontiguousarray(
            ket_shape_offsets, dtype=np.int64
        ).reshape(-1)
        ket_shape_arr = np.ascontiguousarray(ket_shapes, dtype=np.int64).reshape(-1)
        mpo_data_arr = np.ascontiguousarray(mpo_data, dtype=np.float64).reshape(-1)
        mpo_offset_arr = np.ascontiguousarray(mpo_offsets, dtype=np.int64).reshape(-1)
        mpo_shape_offset_arr = np.ascontiguousarray(
            mpo_shape_offsets, dtype=np.int64
        ).reshape(-1)
        mpo_shape_arr = np.ascontiguousarray(mpo_shapes, dtype=np.int64).reshape(-1)
        output_offset_arr = np.ascontiguousarray(
            output_offsets, dtype=np.int64
        ).reshape(-1)
        output_shape_offset_arr = np.ascontiguousarray(
            output_shape_offsets, dtype=np.int64
        ).reshape(-1)
        output_shape_arr = np.ascontiguousarray(
            output_shapes, dtype=np.int64
        ).reshape(-1)
        output_label_arr = np.ascontiguousarray(
            output_labels, dtype=np.int64
        ).reshape(-1)
        same_topology = self._engine.advance_boundary(
            side_name,
            <int64_t>parent_bond,
            <int64_t>child_bond,
            <cpp_bool>(normalized_side == "left"),
            <const int64_t*>cnp.PyArray_DATA(route_arr),
            <size_t>route_arr.shape[0],
            <const double*>cnp.PyArray_DATA(bra_data_arr),
            <size_t>bra_data_arr.size,
            <const int64_t*>cnp.PyArray_DATA(bra_offset_arr),
            <size_t>bra_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(bra_shape_offset_arr),
            <size_t>bra_shape_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(bra_shape_arr),
            <size_t>bra_shape_arr.size,
            <const double*>cnp.PyArray_DATA(ket_data_arr),
            <size_t>ket_data_arr.size,
            <const int64_t*>cnp.PyArray_DATA(ket_offset_arr),
            <size_t>ket_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(ket_shape_offset_arr),
            <size_t>ket_shape_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(ket_shape_arr),
            <size_t>ket_shape_arr.size,
            <const double*>cnp.PyArray_DATA(mpo_data_arr),
            <size_t>mpo_data_arr.size,
            <const int64_t*>cnp.PyArray_DATA(mpo_offset_arr),
            <size_t>mpo_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(mpo_shape_offset_arr),
            <size_t>mpo_shape_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(mpo_shape_arr),
            <size_t>mpo_shape_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_offset_arr),
            <size_t>output_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_shape_offset_arr),
            <size_t>output_shape_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_shape_arr),
            <size_t>output_shape_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_label_arr),
            <size_t>output_label_arr.size,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
            <const double*>cnp.PyArray_DATA(route_coefficient_arr),
            <cpp_bool>metric_boundary,
        )
        if metric_boundary:
            buffer_handle = self._engine.retain_metric_boundary_buffer(
                side_name,
                <int64_t>child_bond,
            )
        else:
            buffer_handle = self._engine.retain_boundary_buffer(
                side_name,
                <int64_t>child_bond,
            )
        n_values = boundary_buffer_size(buffer_handle)
        if n_values != <size_t>output_offset_arr[output_offset_arr.size - 1]:
            release_boundary_buffer(buffer_handle)
            raise RuntimeError("C++ boundary arena has an inconsistent size.")
        buffer_owner = _SU2BoundaryBuffer.__new__(_SU2BoundaryBuffer)
        buffer_owner._handle = buffer_handle
        buffer_handle = NULL
        result_shape[0] = <cnp.npy_intp>n_values
        result = cnp.PyArray_SimpleNewFromData(
            1,
            result_shape,
            cnp.NPY_FLOAT64,
            <void*>boundary_buffer_data(buffer_owner._handle),
        )
        cnp.set_array_base(result, buffer_owner)
        self._boundary_value_owners.pop(owner_key, None)
        return result, bool(same_topology)

    def advance_normal_complementary_boundary(
        self,
        object side,
        long long parent_bond,
        long long child_bond,
        long long site,
        bint reduced_physical,
        object routes,
        object bra_data,
        object bra_offsets,
        object bra_shape_offsets,
        object bra_shapes,
        object ket_data,
        object ket_offsets,
        object ket_shape_offsets,
        object ket_shapes,
        object output_offsets,
        object output_shape_offsets,
        object output_shapes,
        object output_labels,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
        bint dual_right_boundary=False,
        unsigned long long route_topology_revision=0,
    ):
        """Advance a reduced boundary from C++ NC and sector-route arenas.

        Route columns are parent, bra, ket, transition, physical output/input
        charge, output, then the six boundary/physical/next doubled spins.
        """

        cdef str normalized_side = str(side).lower()
        cdef string side_name = normalized_side.encode()
        cdef cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] route_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] bra_data_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_shape_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] ket_data_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_label_arr
        cdef cnp.ndarray result
        cdef cnp.npy_intp result_shape[1]
        cdef CppBoundaryBufferHandle* buffer_handle = NULL
        cdef _SU2BoundaryBuffer buffer_owner
        cdef size_t n_values
        cdef bint same_topology
        cdef object owner_key = (normalized_side, int(child_bond))

        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        if routes is None:
            route_arr = np.empty((0, 13), dtype=np.int32)
        else:
            route_arr = np.ascontiguousarray(
                routes,
                dtype=np.int32,
            ).reshape((-1, 13))
        bra_data_arr = np.ascontiguousarray(bra_data, dtype=np.float64).reshape(-1)
        bra_offset_arr = np.ascontiguousarray(bra_offsets, dtype=np.int64).reshape(-1)
        bra_shape_offset_arr = np.ascontiguousarray(
            bra_shape_offsets, dtype=np.int64
        ).reshape(-1)
        bra_shape_arr = np.ascontiguousarray(bra_shapes, dtype=np.int64).reshape(-1)
        ket_data_arr = np.ascontiguousarray(ket_data, dtype=np.float64).reshape(-1)
        ket_offset_arr = np.ascontiguousarray(ket_offsets, dtype=np.int64).reshape(-1)
        ket_shape_offset_arr = np.ascontiguousarray(
            ket_shape_offsets, dtype=np.int64
        ).reshape(-1)
        ket_shape_arr = np.ascontiguousarray(ket_shapes, dtype=np.int64).reshape(-1)
        output_offset_arr = np.ascontiguousarray(
            output_offsets, dtype=np.int64
        ).reshape(-1)
        output_shape_offset_arr = np.ascontiguousarray(
            output_shape_offsets, dtype=np.int64
        ).reshape(-1)
        output_shape_arr = np.ascontiguousarray(
            output_shapes, dtype=np.int64
        ).reshape(-1)
        output_label_arr = np.ascontiguousarray(
            output_labels, dtype=np.int64
        ).reshape(-1)
        same_topology = self._engine.advance_normal_complementary_boundary(
            side_name,
            <int64_t>parent_bond,
            <int64_t>child_bond,
            <cpp_bool>(normalized_side == "left"),
            <int64_t>site,
            <cpp_bool>reduced_physical,
            <cpp_bool>dual_right_boundary,
            <const int32_t*>cnp.PyArray_DATA(route_arr),
            <size_t>route_arr.shape[0],
            <const double*>cnp.PyArray_DATA(bra_data_arr),
            <size_t>bra_data_arr.size,
            <const int64_t*>cnp.PyArray_DATA(bra_offset_arr),
            <size_t>bra_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(bra_shape_offset_arr),
            <size_t>bra_shape_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(bra_shape_arr),
            <size_t>bra_shape_arr.size,
            <const double*>cnp.PyArray_DATA(ket_data_arr),
            <size_t>ket_data_arr.size,
            <const int64_t*>cnp.PyArray_DATA(ket_offset_arr),
            <size_t>ket_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(ket_shape_offset_arr),
            <size_t>ket_shape_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(ket_shape_arr),
            <size_t>ket_shape_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_offset_arr),
            <size_t>output_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_shape_offset_arr),
            <size_t>output_shape_offset_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_shape_arr),
            <size_t>output_shape_arr.size,
            <const int64_t*>cnp.PyArray_DATA(output_label_arr),
            <size_t>output_label_arr.size,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
            <uint64_t>route_topology_revision,
        )
        buffer_handle = self._engine.retain_boundary_buffer(
            side_name,
            <int64_t>child_bond,
        )
        n_values = boundary_buffer_size(buffer_handle)
        if n_values != <size_t>output_offset_arr[output_offset_arr.size - 1]:
            release_boundary_buffer(buffer_handle)
            raise RuntimeError("C++ NC boundary arena has an inconsistent size.")
        buffer_owner = _SU2BoundaryBuffer.__new__(_SU2BoundaryBuffer)
        buffer_owner._handle = buffer_handle
        buffer_handle = NULL
        result_shape[0] = <cnp.npy_intp>n_values
        result = cnp.PyArray_SimpleNewFromData(
            1,
            result_shape,
            cnp.NPY_FLOAT64,
            <void*>boundary_buffer_data(buffer_owner._handle),
        )
        cnp.set_array_base(result, buffer_owner)
        self._boundary_value_owners.pop(owner_key, None)
        return result, bool(same_topology)

    @property
    def active_bond(self):
        return int(self._engine.active_bond())

    def install_split_site(self, long long site, object tensor):
        """Pack one just-truncated rank-3 site into the C++ sweep owner."""

        cdef tuple items = tuple(tensor.data.items())
        cdef list packed = []
        cdef object key
        cdef object block
        cdef object raw
        cdef cnp.ndarray arr
        cdef tuple keys
        cdef tuple topology
        cdef object previous_topology
        cdef object metadata
        cdef object sector
        cdef object irrep
        cdef object charge
        cdef object two_j
        cdef object sector_dim
        cdef object qns
        cdef list leg_sectors = []
        cdef list leg_dimensions = []
        cdef list ordered_sectors
        cdef list ordered_dimensions
        cdef dict key_indices
        cdef vector[double] values
        cdef vector[int64_t] offsets
        cdef vector[int64_t] labels
        cdef vector[int64_t] shape_offsets
        cdef vector[int64_t] shapes
        cdef vector[int64_t] leg_sector_offsets
        cdef vector[int64_t] leg_sector_labels
        cdef vector[int64_t] leg_sector_dims
        cdef Py_ssize_t total_values = 0
        cdef Py_ssize_t total_shapes = 0
        cdef Py_ssize_t cursor = 0
        cdef Py_ssize_t index
        cdef Py_ssize_t dim
        cdef Py_ssize_t axis
        cdef unsigned long long topology_revision
        cdef unsigned long long numeric_revision
        cdef unsigned long long owner_revision
        cdef bint same_topology

        if site < 0:
            raise ValueError("split-site index must be nonnegative.")
        for key, block in items:
            raw = np.asarray(block)
            if np.iscomplexobj(raw):
                if raw.size:
                    scale = max(1.0, float(np.max(np.abs(raw.real))))
                    tolerance = (
                        16.0
                        * np.sqrt(np.finfo(np.float64).eps)
                        * scale
                    )
                    if float(np.max(np.abs(raw.imag))) > tolerance:
                        return None
                raw = raw.real
            arr = np.ascontiguousarray(raw, dtype=np.float64)
            packed.append(arr)
            total_values += arr.size
            total_shapes += arr.ndim

        qns = getattr(tensor, "qns", None)
        if qns is None or len(qns) != 3:
            return None
        for axis in range(3):
            ordered_sectors = []
            for sector in qns[axis]:
                if sector not in ordered_sectors:
                    ordered_sectors.append(sector)
            ordered_dimensions = []
            for sector in ordered_sectors:
                sector_dim = None
                for key, block in items:
                    if key[axis] == sector:
                        sector_dim = int(np.asarray(block).shape[axis])
                        break
                if sector_dim is None:
                    sector_dim = sum(
                        1 for candidate in qns[axis]
                        if candidate == sector
                    )
                if int(sector_dim) <= 0:
                    return None
                ordered_dimensions.append(int(sector_dim))
            leg_sectors.append(ordered_sectors)
            leg_dimensions.append(ordered_dimensions)

        keys = tuple(key for key, _block in items)
        topology = tuple(
            (key, tuple(int(value) for value in arr.shape))
            for (key, _block), arr in zip(items, packed)
        ) + (
            tuple(
                (
                    tuple(leg_sectors[axis]),
                    tuple(leg_dimensions[axis]),
                )
                for axis in range(3)
            ),
        )
        previous_topology = self._split_site_topologies.get(int(site))
        if previous_topology == topology:
            topology_revision = <unsigned long long>(
                self._split_site_revisions[int(site)][0]
            )
        else:
            self._split_site_topology_clock += 1
            topology_revision = self._split_site_topology_clock
        self._split_site_numeric_clock += 1
        numeric_revision = self._split_site_numeric_clock

        values.resize(total_values)
        offsets.resize(len(packed) + 1)
        labels.resize(6 * len(packed))
        shape_offsets.resize(len(packed) + 1)
        shapes.resize(total_shapes)
        leg_sector_offsets.resize(4)
        leg_sector_labels.resize(
            2 * sum(len(sectors) for sectors in leg_sectors)
        )
        leg_sector_dims.resize(
            sum(len(sectors) for sectors in leg_sectors)
        )
        offsets[0] = 0
        shape_offsets[0] = 0
        cursor = 0
        total_shapes = 0
        leg_sector_offsets[0] = 0
        dim = 0
        for axis in range(3):
            for index in range(len(leg_sectors[axis])):
                sector = leg_sectors[axis][index]
                charge = getattr(sector, "charge", None)
                irrep = getattr(sector, "irrep", None)
                two_j = getattr(irrep, "two_j", None)
                if two_j is None:
                    two_j = getattr(sector, "two_j", None)
                if charge is None or two_j is None:
                    return None
                leg_sector_labels[2 * dim] = int(charge)
                leg_sector_labels[2 * dim + 1] = int(two_j)
                leg_sector_dims[dim] = int(leg_dimensions[axis][index])
                dim += 1
            leg_sector_offsets[axis + 1] = dim
        for index in range(len(packed)):
            arr = packed[index]
            key = keys[index]
            if len(key) != 3:
                return None
            for axis in range(3):
                sector = key[axis]
                charge = getattr(sector, "charge", None)
                irrep = getattr(sector, "irrep", None)
                two_j = getattr(irrep, "two_j", None)
                if two_j is None:
                    two_j = getattr(sector, "two_j", None)
                if charge is None or two_j is None:
                    return None
                labels[6 * index + 2 * axis] = int(charge)
                labels[6 * index + 2 * axis + 1] = int(two_j)
            if arr.size:
                memcpy(
                    <void*>(values.data() + cursor),
                    <const void*>cnp.PyArray_DATA(arr),
                    arr.size * sizeof(double),
                )
            cursor += arr.size
            offsets[index + 1] = cursor
            for dim in range(arr.ndim):
                shapes[total_shapes] = arr.shape[dim]
                total_shapes += 1
            shape_offsets[index + 1] = total_shapes

        same_topology = self._engine.install_split_site(
            <int64_t>site,
            values,
            offsets,
            labels,
            shape_offsets,
            shapes,
            leg_sector_offsets,
            leg_sector_labels,
            leg_sector_dims,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        )
        key_indices = {
            key: int(index)
            for index, key in enumerate(keys)
        }
        self._split_site_keys[int(site)] = keys
        self._split_site_key_indices[int(site)] = key_indices
        self._split_site_topologies[int(site)] = topology
        self._split_site_revisions[int(site)] = (
            int(topology_revision),
            int(numeric_revision),
        )

        owner_revision = self._engine.split_site_owner_revision()
        metadata = getattr(tensor, "metadata", None)
        if isinstance(metadata, dict):
            metadata["_cpp_split_site"] = (
                int(owner_revision),
                int(site),
                int(topology_revision),
                int(numeric_revision),
            )
        return {
            "owner_revision": int(owner_revision),
            "site": int(site),
            "topology_revision": int(topology_revision),
            "numeric_revision": int(numeric_revision),
            "blocks": int(len(packed)),
            "values": int(total_values),
            "same_topology": bool(same_topology),
        }

    def install_mps(self, object sites):
        """Synchronize a complete rank-3 MPS with the persistent C++ owner."""

        cdef list records = []
        cdef object tensor
        cdef object record
        cdef object metadata
        cdef object marker
        cdef Py_ssize_t site
        for site, tensor in enumerate(sites):
            metadata = getattr(tensor, "metadata", None)
            marker = (
                metadata.get("_cpp_split_site")
                if isinstance(metadata, dict)
                else None
            )
            if (
                marker is not None
                and self.split_site_installed(site, marker)
            ):
                records.append({
                    "owner_revision": int(marker[0]),
                    "site": int(site),
                    "topology_revision": int(marker[2]),
                    "numeric_revision": int(marker[3]),
                    "blocks": int(len(tensor.data)),
                    "values": int(
                        sum(np.asarray(block).size for block in tensor.data.values())
                    ),
                    "same_topology": True,
                    "already_owned": True,
                })
                continue
            record = self.install_split_site(site, tensor)
            if record is None:
                raise ValueError(
                    "C++ MPS ownership requires real rank-3 SU(2) tensors."
                )
            records.append(record)
        return tuple(records)

    def install_state_average_mps(
        self,
        object roots,
        object weights,
        long long center_site,
    ):
        """Install a shared isometric chain and root-indexed center tensors."""

        cdef list root_sites = [
            list(getattr(root, "sites", root))
            for root in roots
        ]
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] weight_arr
        cdef tuple center_keys
        cdef object reference_center
        cdef object root_center
        cdef object key
        cdef object raw
        cdef object reference_raw
        cdef cnp.ndarray arr
        cdef vector[double] values
        cdef Py_ssize_t root_idx
        cdef Py_ssize_t site
        cdef Py_ssize_t cursor
        cdef Py_ssize_t total_values
        if len(root_sites) < 2:
            raise ValueError("State averaging requires at least two roots.")
        if center_site < 0 or center_site >= len(root_sites[0]):
            raise IndexError("State-average center site is outside the chain.")
        if any(len(sites) != len(root_sites[0]) for sites in root_sites):
            raise ValueError("State-average roots must have equal chain length.")
        weight_arr = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
        if weight_arr.size != len(root_sites):
            raise ValueError("State-average weights must match the root count.")

        reference_center = root_sites[0][center_site]
        center_keys = tuple(reference_center.data.keys())
        for root_idx in range(1, len(root_sites)):
            for site in range(len(root_sites[0])):
                if site == center_site:
                    continue
                if tuple(root_sites[root_idx][site].data.keys()) != tuple(
                    root_sites[0][site].data.keys()
                ):
                    raise ValueError(
                        "State-average roots do not share the non-center topology."
                    )
                for key in root_sites[0][site].data:
                    if not np.allclose(
                        root_sites[root_idx][site].data[key],
                        root_sites[0][site].data[key],
                        rtol=1.0e-11,
                        atol=1.0e-13,
                    ):
                        raise ValueError(
                            "State-average roots must share all non-center isometries."
                        )

        self.install_mps(root_sites[0])
        self._engine.configure_state_average(
            <const double*>cnp.PyArray_DATA(weight_arr),
            <size_t>weight_arr.size,
            <int64_t>center_site,
        )
        total_values = sum(
            np.asarray(reference_center.data[key]).size
            for key in center_keys
        )
        values.resize(total_values)
        for root_idx in range(len(root_sites)):
            root_center = root_sites[root_idx][center_site]
            if tuple(root_center.data.keys()) != center_keys:
                raise ValueError(
                    "State-average center tensors must share one packed topology."
                )
            cursor = 0
            for key in center_keys:
                reference_raw = np.asarray(reference_center.data[key])
                raw = np.asarray(root_center.data[key])
                if raw.shape != reference_raw.shape:
                    raise ValueError(
                        "State-average center block shapes must match."
                    )
                if np.iscomplexobj(raw):
                    if raw.size and float(np.max(np.abs(raw.imag))) > (
                        16.0 * np.sqrt(np.finfo(np.float64).eps)
                        * max(1.0, float(np.max(np.abs(raw.real))))
                    ):
                        raise ValueError(
                            "C++ state-average centers currently require real tensors."
                        )
                    raw = raw.real
                arr = np.ascontiguousarray(raw, dtype=np.float64)
                if arr.size:
                    memcpy(
                        <void*>(values.data() + cursor),
                        <const void*>cnp.PyArray_DATA(arr),
                        arr.size * sizeof(double),
                    )
                cursor += arr.size
            self._engine.install_state_average_center(
                <size_t>root_idx,
                <const double*>values.data(),
                <size_t>values.size(),
            )
        return {
            "roots": int(self._engine.state_average_roots()),
            "center_site": int(center_site),
            "center_values_per_root": int(total_values),
        }

    @property
    def state_average_roots(self):
        return int(self._engine.state_average_roots())

    def export_state_average_center(self):
        cdef const vector[vector[double]]* values = (
            &self._engine.state_average_center_values()
        )
        cdef Py_ssize_t root_idx
        cdef Py_ssize_t index
        cdef list roots = []
        if self._engine.state_average_center_site() < 0:
            return None
        for root_idx in range(values[0].size()):
            roots.append(np.asarray([
                values[0][root_idx][index]
                for index in range(values[0][root_idx].size())
            ], dtype=np.float64))
        return {
            "site": int(self._engine.state_average_center_site()),
            "values": roots,
        }

    def merge_active_bond(self):
        """Merge the active packed MPS sites entirely in C++."""

        cdef CppPackedSiteTensor merged
        cdef CppPackedSiteTensor channels
        self._engine.merge_active_bond()
        merged = self._engine.merged_site()
        channels = self._engine.merged_channel_site()
        return {
            "merged": {
                "values": np.asarray(
                    [
                        merged.values[index]
                        for index in range(merged.values.size())
                    ],
                    dtype=np.float64,
                ),
                "offsets": np.asarray(
                    [
                        merged.offsets[index]
                        for index in range(merged.offsets.size())
                    ],
                    dtype=np.int64,
                ),
                "labels": np.asarray(
                    [
                        merged.labels[index]
                        for index in range(merged.labels.size())
                    ],
                    dtype=np.int64,
                ).reshape((-1, 8)),
                "shape_offsets": np.asarray(
                    [
                        merged.shape_offsets[index]
                        for index in range(merged.shape_offsets.size())
                    ],
                    dtype=np.int64,
                ),
                "shapes": np.asarray(
                    [
                        merged.shapes[index]
                        for index in range(merged.shapes.size())
                    ],
                    dtype=np.int64,
                ),
            },
            "channels": {
                "values": np.asarray(
                    [
                        channels.values[index]
                        for index in range(channels.values.size())
                    ],
                    dtype=np.float64,
                ),
                "offsets": np.asarray(
                    [
                        channels.offsets[index]
                        for index in range(channels.offsets.size())
                    ],
                    dtype=np.int64,
                ),
                "labels": np.asarray(
                    [
                        channels.labels[index]
                        for index in range(channels.labels.size())
                    ],
                    dtype=np.int64,
                ).reshape((-1, 10)),
                "shape_offsets": np.asarray(
                    [
                        channels.shape_offsets[index]
                        for index in range(channels.shape_offsets.size())
                    ],
                    dtype=np.int64,
                ),
                "shapes": np.asarray(
                    [
                        channels.shapes[index]
                        for index in range(channels.shapes.size())
                    ],
                    dtype=np.int64,
                ),
            },
        }

    def stage_bond_update(
        self,
        object left,
        object right,
        unsigned long long kept_states=0,
        double truncation_seconds=0.0,
    ):
        """Install both split sites and atomically stage the C++ bond update."""

        cdef long long bond = self._engine.active_bond()
        cdef object left_record
        cdef object right_record
        if bond < 0:
            raise RuntimeError("No active C++ bond is available to stage.")
        left_record = self.install_split_site(bond, left)
        right_record = self.install_split_site(bond + 1, right)
        if left_record is None or right_record is None:
            raise ValueError(
                "C++ bond staging requires real-valued split-site tensors."
            )
        self._engine.stage_bond_update(
            <uint64_t>kept_states,
            truncation_seconds,
        )
        return {
            "bond": int(bond),
            "left": left_record,
            "right": right_record,
        }

    def split_site_installed(self, long long site, object split_marker):
        cdef tuple marker
        try:
            marker = tuple(split_marker)
            if (
                len(marker) != 4
                or int(marker[0]) != int(
                    self._engine.split_site_owner_revision()
                )
                or int(marker[1]) != int(site)
            ):
                return False
            return bool(
                self._engine.split_site_installed(
                    <int64_t>site,
                    <uint64_t>int(marker[2]),
                    <uint64_t>int(marker[3]),
                )
            )
        except (TypeError, ValueError):
            return False

    def advance_normal_complementary_boundary_from_split_site(
        self,
        object side,
        long long parent_bond,
        long long child_bond,
        long long site,
        bint reduced_physical,
        object routes,
        object bra_keys,
        object ket_keys,
        object split_marker,
        object output_offsets,
        object output_shape_offsets,
        object output_shapes,
        object output_labels,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
        bint dual_right_boundary=False,
        unsigned long long route_topology_revision=0,
    ):
        """Advance an NC boundary directly from the C++-owned split site."""

        cdef str normalized_side = str(side).lower()
        cdef string side_name = normalized_side.encode()
        cdef tuple marker = tuple(split_marker)
        cdef dict key_indices
        cdef cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] route_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_remap
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_remap
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_label_arr
        cdef cnp.ndarray result
        cdef cnp.npy_intp result_shape[1]
        cdef CppBoundaryBufferHandle* buffer_handle = NULL
        cdef _SU2BoundaryBuffer buffer_owner
        cdef size_t n_values
        cdef bint same_topology
        cdef object owner_key = (normalized_side, int(child_bond))

        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        if (
            len(marker) != 4
            or int(marker[0]) != int(self._engine.split_site_owner_revision())
            or int(marker[1]) != int(site)
        ):
            raise ValueError("split-site marker does not belong to this C++ owner.")
        if not self._engine.split_site_installed(
            <int64_t>site,
            <uint64_t>int(marker[2]),
            <uint64_t>int(marker[3]),
        ):
            raise ValueError("split-site numerical revision is no longer installed.")
        if self._engine.cached_normal_complementary_boundary_ready(
            side_name,
            <int64_t>child_bond,
            <int64_t>site,
            <uint64_t>topology_revision,
            <uint64_t>route_topology_revision,
        ):
            same_topology = (
                self._engine
                .replay_normal_complementary_boundary_from_split_site(
                    side_name,
                    <int64_t>child_bond,
                    <uint64_t>numeric_revision,
                )
            )
            buffer_handle = self._engine.retain_boundary_buffer(
                side_name,
                <int64_t>child_bond,
            )
            n_values = boundary_buffer_size(buffer_handle)
            buffer_owner = _SU2BoundaryBuffer.__new__(_SU2BoundaryBuffer)
            buffer_owner._handle = buffer_handle
            buffer_handle = NULL
            result_shape[0] = <cnp.npy_intp>n_values
            result = cnp.PyArray_SimpleNewFromData(
                1,
                result_shape,
                cnp.NPY_FLOAT64,
                <void*>boundary_buffer_data(buffer_owner._handle),
            )
            cnp.set_array_base(result, buffer_owner)
            self._boundary_value_owners.pop(owner_key, None)
            return result, bool(same_topology)
        key_indices = self._split_site_key_indices.get(int(site))
        if key_indices is None:
            raise ValueError("split-site key topology is unavailable.")
        try:
            bra_remap = np.ascontiguousarray(
                [key_indices[key] for key in bra_keys],
                dtype=np.int64,
            )
            ket_remap = np.ascontiguousarray(
                [key_indices[key] for key in ket_keys],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("boundary route references a missing split-site block.") from error

        if routes is None:
            route_arr = np.empty((0, 13), dtype=np.int32)
        else:
            route_arr = np.array(
                routes,
                dtype=np.int32,
                copy=True,
                order="C",
            ).reshape((-1, 13))
            if (
                np.any(route_arr[:, 1] < 0)
                or np.any(route_arr[:, 1] >= bra_remap.size)
                or np.any(route_arr[:, 2] < 0)
                or np.any(route_arr[:, 2] >= ket_remap.size)
            ):
                raise ValueError("split-site route block index is out of range.")
            route_arr[:, 1] = bra_remap[route_arr[:, 1]]
            route_arr[:, 2] = ket_remap[route_arr[:, 2]]
        output_offset_arr = np.ascontiguousarray(
            output_offsets, dtype=np.int64
        ).reshape(-1)
        output_shape_offset_arr = np.ascontiguousarray(
            output_shape_offsets, dtype=np.int64
        ).reshape(-1)
        output_shape_arr = np.ascontiguousarray(
            output_shapes, dtype=np.int64
        ).reshape(-1)
        output_label_arr = np.ascontiguousarray(
            output_labels, dtype=np.int64
        ).reshape(-1)
        same_topology = (
            self._engine.advance_normal_complementary_boundary_from_split_site(
                side_name,
                <int64_t>parent_bond,
                <int64_t>child_bond,
                <cpp_bool>(normalized_side == "left"),
                <int64_t>site,
                <cpp_bool>reduced_physical,
                <cpp_bool>dual_right_boundary,
                <const int32_t*>cnp.PyArray_DATA(route_arr),
                <size_t>route_arr.shape[0],
                <const int64_t*>cnp.PyArray_DATA(output_offset_arr),
                <size_t>output_offset_arr.size,
                <const int64_t*>cnp.PyArray_DATA(output_shape_offset_arr),
                <size_t>output_shape_offset_arr.size,
                <const int64_t*>cnp.PyArray_DATA(output_shape_arr),
                <size_t>output_shape_arr.size,
                <const int64_t*>cnp.PyArray_DATA(output_label_arr),
                <size_t>output_label_arr.size,
                <uint64_t>topology_revision,
                <uint64_t>numeric_revision,
                <uint64_t>route_topology_revision,
            )
        )
        buffer_handle = self._engine.retain_boundary_buffer(
            side_name,
            <int64_t>child_bond,
        )
        n_values = boundary_buffer_size(buffer_handle)
        if n_values != <size_t>output_offset_arr[output_offset_arr.size - 1]:
            release_boundary_buffer(buffer_handle)
            raise RuntimeError("C++ NC boundary arena has an inconsistent size.")
        buffer_owner = _SU2BoundaryBuffer.__new__(_SU2BoundaryBuffer)
        buffer_owner._handle = buffer_handle
        buffer_handle = NULL
        result_shape[0] = <cnp.npy_intp>n_values
        result = cnp.PyArray_SimpleNewFromData(
            1,
            result_shape,
            cnp.NPY_FLOAT64,
            <void*>boundary_buffer_data(buffer_owner._handle),
        )
        cnp.set_array_base(result, buffer_owner)
        self._boundary_value_owners.pop(owner_key, None)
        return result, bool(same_topology)

    def advance_metric_boundary_from_split_site(
        self,
        object side,
        long long parent_bond,
        long long child_bond,
        long long site,
        object routes,
        object bra_keys,
        object ket_keys,
        object split_marker,
        object mpo_data,
        object mpo_offsets,
        object mpo_shape_offsets,
        object mpo_shapes,
        object output_offsets,
        object output_shape_offsets,
        object output_shapes,
        object output_labels,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
        unsigned long long route_topology_revision=0,
        object route_coefficients=None,
    ):
        """Advance the norm boundary from the C++-owned canonical site."""

        cdef str normalized_side = str(side).lower()
        cdef string side_name = normalized_side.encode()
        cdef tuple marker = tuple(split_marker)
        cdef dict key_indices
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] route_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] route_coefficient_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] bra_remap
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ket_remap
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] mpo_data_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] mpo_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] mpo_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] mpo_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] output_label_arr
        cdef cnp.ndarray result
        cdef cnp.npy_intp result_shape[1]
        cdef CppBoundaryBufferHandle* buffer_handle = NULL
        cdef _SU2BoundaryBuffer buffer_owner
        cdef size_t n_values
        cdef bint same_topology
        cdef object owner_key = (
            "metric",
            normalized_side,
            int(child_bond),
        )

        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        if (
            len(marker) != 4
            or int(marker[0]) != int(
                self._engine.split_site_owner_revision()
            )
            or int(marker[1]) != int(site)
            or not self._engine.split_site_installed(
                <int64_t>site,
                <uint64_t>int(marker[2]),
                <uint64_t>int(marker[3]),
            )
        ):
            raise ValueError(
                "metric boundary split-site revision is not installed."
            )
        key_indices = self._split_site_key_indices.get(int(site))
        if key_indices is None:
            raise ValueError("split-site key topology is unavailable.")
        try:
            bra_remap = np.ascontiguousarray(
                [key_indices[key] for key in bra_keys],
                dtype=np.int64,
            )
            ket_remap = np.ascontiguousarray(
                [key_indices[key] for key in ket_keys],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "metric route references a missing split-site block."
            ) from error
        route_arr = np.array(
            routes,
            dtype=np.int64,
            copy=True,
            order="C",
        ).reshape((-1, 5))
        route_coefficient_arr = np.ascontiguousarray(
            (
                np.ones(route_arr.shape[0], dtype=np.float64)
                if route_coefficients is None
                else route_coefficients
            ),
            dtype=np.float64,
        ).reshape(-1)
        if route_coefficient_arr.size != route_arr.shape[0]:
            raise ValueError(
                "route_coefficients must match the metric route count."
            )
        if (
            np.any(route_arr[:, 1] < 0)
            or np.any(route_arr[:, 1] >= bra_remap.size)
            or np.any(route_arr[:, 2] < 0)
            or np.any(route_arr[:, 2] >= ket_remap.size)
        ):
            raise ValueError(
                "metric split-site route block index is out of range."
            )
        route_arr[:, 1] = bra_remap[route_arr[:, 1]]
        route_arr[:, 2] = ket_remap[route_arr[:, 2]]
        mpo_data_arr = np.ascontiguousarray(
            mpo_data,
            dtype=np.float64,
        ).reshape(-1)
        mpo_offset_arr = np.ascontiguousarray(
            mpo_offsets,
            dtype=np.int64,
        ).reshape(-1)
        mpo_shape_offset_arr = np.ascontiguousarray(
            mpo_shape_offsets,
            dtype=np.int64,
        ).reshape(-1)
        mpo_shape_arr = np.ascontiguousarray(
            mpo_shapes,
            dtype=np.int64,
        ).reshape(-1)
        output_offset_arr = np.ascontiguousarray(
            output_offsets,
            dtype=np.int64,
        ).reshape(-1)
        output_shape_offset_arr = np.ascontiguousarray(
            output_shape_offsets,
            dtype=np.int64,
        ).reshape(-1)
        output_shape_arr = np.ascontiguousarray(
            output_shapes,
            dtype=np.int64,
        ).reshape(-1)
        output_label_arr = np.ascontiguousarray(
            output_labels,
            dtype=np.int64,
        ).reshape(-1)
        same_topology = (
            self._engine.advance_metric_boundary_from_split_site(
                side_name,
                <int64_t>parent_bond,
                <int64_t>child_bond,
                <cpp_bool>(normalized_side == "left"),
                <int64_t>site,
                <const int64_t*>cnp.PyArray_DATA(route_arr),
                <size_t>route_arr.shape[0],
                <const double*>cnp.PyArray_DATA(route_coefficient_arr),
                <size_t>route_coefficient_arr.size,
                <const double*>cnp.PyArray_DATA(mpo_data_arr),
                <size_t>mpo_data_arr.size,
                <const int64_t*>cnp.PyArray_DATA(mpo_offset_arr),
                <size_t>mpo_offset_arr.size,
                <const int64_t*>cnp.PyArray_DATA(
                    mpo_shape_offset_arr
                ),
                <size_t>mpo_shape_offset_arr.size,
                <const int64_t*>cnp.PyArray_DATA(mpo_shape_arr),
                <size_t>mpo_shape_arr.size,
                <const int64_t*>cnp.PyArray_DATA(output_offset_arr),
                <size_t>output_offset_arr.size,
                <const int64_t*>cnp.PyArray_DATA(
                    output_shape_offset_arr
                ),
                <size_t>output_shape_offset_arr.size,
                <const int64_t*>cnp.PyArray_DATA(output_shape_arr),
                <size_t>output_shape_arr.size,
                <const int64_t*>cnp.PyArray_DATA(output_label_arr),
                <size_t>output_label_arr.size,
                <uint64_t>topology_revision,
                <uint64_t>numeric_revision,
                <uint64_t>route_topology_revision,
            )
        )
        buffer_handle = self._engine.retain_metric_boundary_buffer(
            side_name,
            <int64_t>child_bond,
        )
        n_values = boundary_buffer_size(buffer_handle)
        if (
            n_values
            != <size_t>output_offset_arr[output_offset_arr.size - 1]
        ):
            release_boundary_buffer(buffer_handle)
            raise RuntimeError(
                "C++ metric boundary arena has an inconsistent size."
            )
        buffer_owner = _SU2BoundaryBuffer.__new__(_SU2BoundaryBuffer)
        buffer_owner._handle = buffer_handle
        buffer_handle = NULL
        result_shape[0] = <cnp.npy_intp>n_values
        result = cnp.PyArray_SimpleNewFromData(
            1,
            result_shape,
            cnp.NPY_FLOAT64,
            <void*>boundary_buffer_data(buffer_owner._handle),
        )
        cnp.set_array_base(result, buffer_owner)
        self._boundary_value_owners.pop(owner_key, None)
        return result, bool(same_topology)

    def clear_boundaries(self):
        self._engine.clear_boundaries()
        self._boundary_value_owners.clear()

    def boundary_values(self, object side, long long bond, bint metric=False):
        """Copy one C++-owned reduced boundary arena for validation."""

        cdef string side_name = str(side).lower().encode()
        cdef CppBoundaryBufferHandle* handle = NULL
        cdef size_t n_values
        cdef cnp.ndarray[cnp.double_t, ndim=1] result
        if metric:
            n_values = self._engine.metric_boundary_value_count(
                side_name,
                <int64_t>bond,
            )
            result = np.empty(n_values, dtype=np.float64)
            self._engine.copy_metric_boundary_values(
                side_name,
                <int64_t>bond,
                <double*>cnp.PyArray_DATA(result),
                n_values,
            )
            return result
        else:
            n_values = self._engine.boundary_value_count(
                side_name,
                <int64_t>bond,
            )
            result = np.empty(n_values, dtype=np.float64)
            self._engine.copy_boundary_values(
                side_name,
                <int64_t>bond,
                <double*>cnp.PyArray_DATA(result),
                n_values,
            )
            return result

    def release_boundary(self, object side, long long bond):
        cdef str normalized_side = str(side).lower()
        cdef string side_name = normalized_side.encode()
        cdef object owner_key = (normalized_side, int(bond))
        cdef bint released = self._engine.release_boundary(
            side_name,
            <int64_t>bond,
        )
        self._boundary_value_owners.pop(owner_key, None)
        return bool(released)

    def install_local_operator(
        self,
        object key,
        object blocks,
        object input_starts,
        object output_starts,
        long long dimension,
    ):
        cdef tuple packed_blocks = tuple(
            np.ascontiguousarray(block, dtype=np.complex128)
            for block in blocks
        )
        cdef Py_ssize_t n_blocks = len(packed_blocks)
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] in_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] out_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rows
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] cols
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] pool
        cdef string key_name = str(key).encode()
        cdef Py_ssize_t block_index
        cdef Py_ssize_t cursor = 0
        cdef object block
        cdef uint64_t topology_revision
        cdef uint64_t numeric_revision
        cdef bint same_topology

        if n_blocks == 0 or dimension <= 0:
            raise ValueError("A local operator requires blocks and a dimension.")
        in_arr = np.ascontiguousarray(input_starts, dtype=np.int64).reshape(-1)
        out_arr = np.ascontiguousarray(output_starts, dtype=np.int64).reshape(-1)
        if in_arr.size != n_blocks or out_arr.size != n_blocks:
            raise ValueError("Local block starts must match the block count.")
        offsets = np.empty(n_blocks + 1, dtype=np.int64)
        rows = np.empty(n_blocks, dtype=np.int64)
        cols = np.empty(n_blocks, dtype=np.int64)
        offsets[0] = 0
        for block_index in range(n_blocks):
            block = packed_blocks[block_index]
            if block.ndim != 2:
                raise ValueError("Local operator blocks must be matrices.")
            rows[block_index] = block.shape[0]
            cols[block_index] = block.shape[1]
            cursor += block.size
            offsets[block_index + 1] = cursor
        pool = np.empty(cursor, dtype=np.complex128)
        cursor = 0
        for block in packed_blocks:
            pool[cursor:cursor + block.size] = block.reshape(-1)
            cursor += block.size
        topology_revision = <uint64_t>_cpp_array_revision(
            offsets,
            rows,
            cols,
            in_arr,
            out_arr,
            np.asarray([dimension], dtype=np.int64),
        )
        numeric_revision = <uint64_t>_cpp_array_revision(pool)
        same_topology = self._engine.install_local_operator(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(pool),
            <size_t>pool.size,
            <const int64_t*>cnp.PyArray_DATA(offsets),
            <const int64_t*>cnp.PyArray_DATA(rows),
            <const int64_t*>cnp.PyArray_DATA(cols),
            <const int64_t*>cnp.PyArray_DATA(in_arr),
            <const int64_t*>cnp.PyArray_DATA(out_arr),
            <size_t>n_blocks,
            <size_t>dimension,
            topology_revision,
            numeric_revision,
        )
        self._local_operator_owner = pool
        return bool(same_topology)

    def clear_local_operator(self):
        self._engine.clear_local_operator()
        self._local_operator_owner = None

    def local_matvec(self, object key, object vector):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        input_arr = np.ascontiguousarray(vector, dtype=np.complex128).reshape(-1)
        output_arr = np.empty(input_arr.size, dtype=np.complex128)
        self._engine.local_matvec(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(input_arr),
            <cpp_complex[double]*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
        )
        return output_arr

    def local_diagonal(self, object key, long long dimension):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        output_arr = np.empty(dimension, dtype=np.complex128)
        self._engine.local_diagonal(
            key_name,
            <cpp_complex[double]*>cnp.PyArray_DATA(output_arr),
            <size_t>dimension,
        )
        return output_arr

    def local_davidson(
        self,
        object key,
        object diagonal,
        object guess,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bint accept_unconverged=False,
    ):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] diagonal_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] guess_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] vector_arr
        cdef string key_name = str(key).encode()
        cdef CppDavidsonResult result
        cdef Py_ssize_t index
        diagonal_arr = np.ascontiguousarray(diagonal, dtype=np.complex128).reshape(-1)
        guess_arr = np.ascontiguousarray(guess, dtype=np.complex128).reshape(-1)
        if diagonal_arr.size != guess_arr.size:
            raise ValueError("Davidson diagonal and guess dimensions differ.")
        result = self._engine.local_davidson(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(diagonal_arr),
            <const cpp_complex[double]*>cnp.PyArray_DATA(guess_arr),
            <size_t>diagonal_arr.size,
            tolerance,
            max_iterations,
            restart_dimension,
            <cpp_bool>accept_unconverged,
        )
        vector_arr = np.empty(result.vector.size(), dtype=np.complex128)
        for index in range(result.vector.size()):
            vector_arr[index] = complex(
                result.vector[index].real(),
                result.vector[index].imag(),
            )
        return {
            "kind": "cpp_su2_moving_environment_davidson",
            "accepted": bool(result.accepted),
            "energy": float(result.energy),
            "vector": vector_arr,
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "basis_size": int(result.basis_size),
            "restarts": int(result.restarts),
            "converged": bool(result.converged),
            "workspace_reused": bool(result.workspace_reused),
            "matvec_calls": int(result.matvec_calls),
        }

    def install_factor_routes(
        self,
        object key,
        object in_indices,
        object out_indices,
        object left_indices,
        object right_indices,
        object basis_offsets,
        object basis_shapes,
        object left_factor_indices,
        object left_offsets,
        object left_shape_offsets,
        object left_shapes,
        object left_data,
        object right_factor_indices,
        object right_offsets,
        object right_shape_offsets,
        object right_shapes,
        object right_data,
        long long total_dimension,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] in_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] out_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] basis_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] basis_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_factor_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_shape_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] left_data_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_factor_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_shape_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_shape_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] right_data_arr
        cdef str key_text = str(key)
        cdef string key_name = key_text.encode()
        cdef bint same_topology

        in_arr = np.ascontiguousarray(in_indices, dtype=np.int64).reshape(-1)
        out_arr = np.ascontiguousarray(out_indices, dtype=np.int64).reshape(-1)
        left_arr = np.ascontiguousarray(left_indices, dtype=np.int64).reshape(-1)
        right_arr = np.ascontiguousarray(right_indices, dtype=np.int64).reshape(-1)
        basis_offset_arr = np.ascontiguousarray(
            basis_offsets,
            dtype=np.int64,
        ).reshape(-1)
        basis_shape_arr = np.ascontiguousarray(basis_shapes, dtype=np.int64)
        left_factor_arr = np.ascontiguousarray(
            left_factor_indices,
            dtype=np.int64,
        ).reshape(-1)
        left_offset_arr = np.ascontiguousarray(
            left_offsets,
            dtype=np.int64,
        ).reshape(-1)
        left_shape_offset_arr = np.ascontiguousarray(
            left_shape_offsets,
            dtype=np.int64,
        ).reshape(-1)
        left_shape_arr = np.ascontiguousarray(
            left_shapes,
            dtype=np.int64,
        ).reshape(-1)
        left_data_arr = np.ascontiguousarray(
            left_data,
            dtype=np.float64,
        ).reshape(-1)
        right_factor_arr = np.ascontiguousarray(
            right_factor_indices,
            dtype=np.int64,
        ).reshape(-1)
        right_offset_arr = np.ascontiguousarray(
            right_offsets,
            dtype=np.int64,
        ).reshape(-1)
        right_shape_offset_arr = np.ascontiguousarray(
            right_shape_offsets,
            dtype=np.int64,
        ).reshape(-1)
        right_shape_arr = np.ascontiguousarray(
            right_shapes,
            dtype=np.int64,
        ).reshape(-1)
        right_data_arr = np.ascontiguousarray(
            right_data,
            dtype=np.float64,
        ).reshape(-1)
        if not (
            in_arr.size == out_arr.size
            and in_arr.size == left_arr.size
            and in_arr.size == right_arr.size
            and basis_shape_arr.ndim == 2
            and basis_shape_arr.shape[1] == 4
            and basis_offset_arr.size == basis_shape_arr.shape[0]
            and left_offset_arr.size == left_shape_offset_arr.size
            and right_offset_arr.size == right_shape_offset_arr.size
            and left_offset_arr.size >= 2
            and right_offset_arr.size >= 2
        ):
            raise ValueError("Packed SU(2) factor-route arrays are inconsistent.")
        if self._factor_route_key is not None and self._factor_route_key != key_text:
            self._engine.clear_factor_route_projection()
            self._factor_route_projection_owners = None
        same_topology = self._engine.install_factor_routes(
            key_name,
            <const int64_t*>cnp.PyArray_DATA(in_arr),
            <const int64_t*>cnp.PyArray_DATA(out_arr),
            <const int64_t*>cnp.PyArray_DATA(left_arr),
            <const int64_t*>cnp.PyArray_DATA(right_arr),
            <size_t>in_arr.size,
            <const int64_t*>cnp.PyArray_DATA(basis_offset_arr),
            <const int64_t*>cnp.PyArray_DATA(basis_shape_arr),
            <size_t>basis_shape_arr.shape[0],
            <const int64_t*>cnp.PyArray_DATA(left_factor_arr),
            <size_t>left_factor_arr.size,
            <const int64_t*>cnp.PyArray_DATA(left_offset_arr),
            <const int64_t*>cnp.PyArray_DATA(left_shape_offset_arr),
            <size_t>(left_offset_arr.size - 1),
            <const int64_t*>cnp.PyArray_DATA(left_shape_arr),
            <size_t>left_shape_arr.size,
            <const double*>cnp.PyArray_DATA(left_data_arr),
            <size_t>left_data_arr.size,
            <const int64_t*>cnp.PyArray_DATA(right_factor_arr),
            <size_t>right_factor_arr.size,
            <const int64_t*>cnp.PyArray_DATA(right_offset_arr),
            <const int64_t*>cnp.PyArray_DATA(right_shape_offset_arr),
            <size_t>(right_offset_arr.size - 1),
            <const int64_t*>cnp.PyArray_DATA(right_shape_arr),
            <size_t>right_shape_arr.size,
            <const double*>cnp.PyArray_DATA(right_data_arr),
            <size_t>right_data_arr.size,
            <size_t>total_dimension,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        )
        self._factor_route_key = key_text
        self._factor_route_revision = None
        self._factor_route_owners = (left_data_arr, right_data_arr)
        return bool(same_topology)

    def install_contextual_factor_routes(
        self,
        object key,
        long long bond,
        long long left_boundary_bond,
        long long right_boundary_bond,
        object basis_offsets,
        object basis_shapes,
        object basis_quantum_numbers,
        object left_sector_ids,
        object right_sector_ids,
        long long total_dimension,
        unsigned long long topology_revision,
        bint dual_right_basis=True,
    ):
        """Build and install contextual NC routes entirely in the C++ owner."""

        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] quantum_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_sector_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_sector_arr
        cdef str key_text = str(key)
        cdef string key_name = key_text.encode()
        cdef bint same_topology

        offset_arr = np.ascontiguousarray(
            basis_offsets,
            dtype=np.int64,
        ).reshape(-1)
        shape_arr = np.ascontiguousarray(basis_shapes, dtype=np.int64)
        quantum_arr = np.ascontiguousarray(
            basis_quantum_numbers,
            dtype=np.int64,
        )
        left_sector_arr = np.ascontiguousarray(
            left_sector_ids,
            dtype=np.int64,
        ).reshape(-1)
        right_sector_arr = np.ascontiguousarray(
            right_sector_ids,
            dtype=np.int64,
        ).reshape(-1)
        if not (
            shape_arr.ndim == 2
            and shape_arr.shape[1] == 4
            and quantum_arr.ndim == 2
            and quantum_arr.shape[1] == 10
            and offset_arr.size == shape_arr.shape[0]
            and quantum_arr.shape[0] == shape_arr.shape[0]
            and left_sector_arr.size == shape_arr.shape[0]
            and right_sector_arr.size == shape_arr.shape[0]
            and total_dimension > 0
        ):
            raise ValueError(
                "Contextual SU(2) basis metadata has inconsistent shapes."
            )
        if self._factor_route_key is not None and self._factor_route_key != key_text:
            self._engine.clear_factor_route_projection()
            self._factor_route_projection_owners = None
        same_topology = self._engine.install_contextual_factor_routes(
            key_name,
            <int64_t>bond,
            <int64_t>left_boundary_bond,
            <int64_t>right_boundary_bond,
            <const int64_t*>cnp.PyArray_DATA(offset_arr),
            <const int64_t*>cnp.PyArray_DATA(shape_arr),
            <const int64_t*>cnp.PyArray_DATA(quantum_arr),
            <const int64_t*>cnp.PyArray_DATA(left_sector_arr),
            <const int64_t*>cnp.PyArray_DATA(right_sector_arr),
            <size_t>shape_arr.shape[0],
            <size_t>total_dimension,
            <uint64_t>topology_revision,
            <cpp_bool>dual_right_basis,
        )
        self._factor_route_key = key_text
        self._factor_route_revision = (
            key_text,
            int(topology_revision),
            int(total_dimension),
        )
        self._factor_route_owners = None
        return {
            "same_topology": bool(same_topology),
            "factor_route_count": int(self._engine.factor_route_count()),
            "family_term_counts": {
                "S": int(self._engine.complementary_family_route_count(0)),
                "R": int(self._engine.complementary_family_route_count(1)),
                "A": int(self._engine.complementary_family_route_count(2)),
                "P": int(self._engine.complementary_family_route_count(3)),
                "B": int(self._engine.complementary_family_route_count(4)),
                "Q": int(self._engine.complementary_family_route_count(5)),
                "unlabeled": int(
                    self._engine.unlabeled_family_route_count()
                ),
            },
        }

    def prepare_active_bond_complementary_actions(
        self,
        long long left_boundary_bond,
        long long right_boundary_bond,
        long long expected_basis,
        long long expected_dimension,
        bint dual_right_basis=True,
    ):
        """Build active-bond S/R/A/P/B/Q actions from C++-owned topology."""

        cdef int status
        cdef string key_name
        if expected_basis <= 0 or expected_dimension <= 0:
            return {"compatible": False}
        status = self._engine.prepare_active_bond_complementary_actions(
            <int64_t>left_boundary_bond,
            <int64_t>right_boundary_bond,
            <size_t>expected_basis,
            <size_t>expected_dimension,
            <cpp_bool>dual_right_basis,
        )
        if status < 0:
            return {
                "compatible": False,
                "reason": int(-status),
            }
        key_name = self._engine.factor_route_key()
        key_text = (<bytes>key_name).decode()
        self._factor_route_key = key_text
        self._factor_route_revision = (
            key_text,
            int(expected_basis),
            int(expected_dimension),
        )
        self._factor_route_owners = None
        return {
            "compatible": True,
            "same_topology": bool(status > 0),
            "factor_route_key": key_text,
            "factor_route_count": int(self._engine.factor_route_count()),
            "family_term_counts": {
                "S": int(self._engine.complementary_family_route_count(0)),
                "R": int(self._engine.complementary_family_route_count(1)),
                "A": int(self._engine.complementary_family_route_count(2)),
                "P": int(self._engine.complementary_family_route_count(3)),
                "B": int(self._engine.complementary_family_route_count(4)),
                "Q": int(self._engine.complementary_family_route_count(5)),
                "unlabeled": int(
                    self._engine.unlabeled_family_route_count()
                ),
            },
        }

    def prepare_active_bond_metric_routes(
        self,
        long long left_boundary_bond,
        long long right_boundary_bond,
    ):
        """Build reduced norm routes from the C++-owned active basis."""

        cdef size_t route_count
        cdef string key_name
        route_count = self._engine.prepare_active_bond_metric_routes(
            <int64_t>left_boundary_bond,
            <int64_t>right_boundary_bond,
        )
        key_name = self._engine.metric_key()
        self._factorized_metric_owners = None
        return {
            "metric_key": (<bytes>key_name).decode(),
            "metric_route_count": int(route_count),
        }

    def install_contextual_metric_routes(
        self,
        object key,
        long long left_boundary_bond,
        long long right_boundary_bond,
        object basis_offsets,
        object basis_shapes,
        object basis_quantum_numbers,
        object left_sector_ids,
        object right_sector_ids,
        long long total_dimension,
        unsigned long long topology_revision,
    ):
        """Build the fully reduced identity metric from C++ boundary arenas."""

        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] quantum_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_sector_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_sector_arr
        cdef str key_text = str(key)
        cdef string key_name = key_text.encode()
        cdef size_t route_count

        offset_arr = np.ascontiguousarray(
            basis_offsets,
            dtype=np.int64,
        ).reshape(-1)
        shape_arr = np.ascontiguousarray(basis_shapes, dtype=np.int64)
        quantum_arr = np.ascontiguousarray(
            basis_quantum_numbers,
            dtype=np.int64,
        )
        left_sector_arr = np.ascontiguousarray(
            left_sector_ids,
            dtype=np.int64,
        ).reshape(-1)
        right_sector_arr = np.ascontiguousarray(
            right_sector_ids,
            dtype=np.int64,
        ).reshape(-1)
        if not (
            shape_arr.ndim == 2
            and shape_arr.shape[1] == 4
            and quantum_arr.ndim == 2
            and quantum_arr.shape[1] == 10
            and offset_arr.size == shape_arr.shape[0]
            and quantum_arr.shape[0] == shape_arr.shape[0]
            and left_sector_arr.size == shape_arr.shape[0]
            and right_sector_arr.size == shape_arr.shape[0]
            and total_dimension > 0
        ):
            raise ValueError(
                "Contextual metric basis metadata has inconsistent shapes."
            )
        route_count = self._engine.install_contextual_metric_routes(
            key_name,
            <int64_t>left_boundary_bond,
            <int64_t>right_boundary_bond,
            <const int64_t*>cnp.PyArray_DATA(offset_arr),
            <const int64_t*>cnp.PyArray_DATA(shape_arr),
            <const int64_t*>cnp.PyArray_DATA(quantum_arr),
            <const int64_t*>cnp.PyArray_DATA(left_sector_arr),
            <const int64_t*>cnp.PyArray_DATA(right_sector_arr),
            <size_t>shape_arr.shape[0],
            <size_t>total_dimension,
            <uint64_t>topology_revision,
        )
        self._factorized_metric_owners = None
        return {
            "metric_key": key_text,
            "metric_route_count": int(route_count),
        }

    def install_raw_factor_routes(
        self,
        object key,
        object in_indices,
        object out_indices,
        object left_indices,
        object right_indices,
        object basis_offsets,
        object basis_shapes,
        object left_factor_indices,
        object left_source,
        object right_factor_indices,
        object right_source,
        long long total_dimension,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
        object left_family_masks=None,
        object right_family_masks=None,
    ):
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] in_arr
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] out_arr
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] left_arr
        cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] right_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] basis_offset_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] basis_shape_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] left_factor_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] right_factor_arr
        cdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] left_family_mask_arr
        cdef cnp.ndarray[cnp.uint64_t, ndim=1, mode="c"] right_family_mask_arr
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lb_ids
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lw_ids
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lb_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lb_shape_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lb_shapes
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] lb_data
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lw_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lw_shape_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] lw_shapes
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] lw_data
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rb_ids
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rw_ids
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rb_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rb_shape_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rb_shapes
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] rb_data
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rw_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rw_shape_offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rw_shapes
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] rw_data
        cdef tuple left_values = tuple(left_source)
        cdef tuple right_values = tuple(right_source)
        cdef str key_text = str(key)
        cdef string key_name = key_text.encode()
        cdef bint same_topology

        if len(left_values) != 10 or len(right_values) != 10:
            raise ValueError("Each raw factor side requires ten source arrays.")
        if self._factor_route_revision == (
            key_text,
            int(topology_revision),
            int(numeric_revision),
            int(total_dimension),
        ):
            return True
        in_arr = np.ascontiguousarray(in_indices, dtype=np.int32).reshape(-1)
        out_arr = np.ascontiguousarray(out_indices, dtype=np.int32).reshape(-1)
        left_arr = np.ascontiguousarray(left_indices, dtype=np.int32).reshape(-1)
        right_arr = np.ascontiguousarray(right_indices, dtype=np.int32).reshape(-1)
        basis_offset_arr = np.ascontiguousarray(
            basis_offsets,
            dtype=np.int64,
        ).reshape(-1)
        basis_shape_arr = np.ascontiguousarray(basis_shapes, dtype=np.int64)
        left_factor_arr = np.ascontiguousarray(
            left_factor_indices,
            dtype=np.int64,
        ).reshape(-1)
        right_factor_arr = np.ascontiguousarray(
            right_factor_indices,
            dtype=np.int64,
        ).reshape(-1)
        if left_family_masks is None:
            left_family_mask_arr = np.zeros(
                left_factor_arr.size,
                dtype=np.uint64,
            )
        else:
            left_family_mask_arr = np.ascontiguousarray(
                left_family_masks,
                dtype=np.uint64,
            ).reshape(-1)
        if right_family_masks is None:
            right_family_mask_arr = np.zeros(
                right_factor_arr.size,
                dtype=np.uint64,
            )
        else:
            right_family_mask_arr = np.ascontiguousarray(
                right_family_masks,
                dtype=np.uint64,
            ).reshape(-1)
        lb_ids = np.ascontiguousarray(left_values[0], dtype=np.int64).reshape(-1)
        lw_ids = np.ascontiguousarray(left_values[1], dtype=np.int64).reshape(-1)
        lb_offsets = np.ascontiguousarray(left_values[2], dtype=np.int64).reshape(-1)
        lb_shape_offsets = np.ascontiguousarray(
            left_values[3], dtype=np.int64
        ).reshape(-1)
        lb_shapes = np.ascontiguousarray(left_values[4], dtype=np.int64).reshape(-1)
        lb_data = np.ascontiguousarray(left_values[5], dtype=np.float64).reshape(-1)
        lw_offsets = np.ascontiguousarray(left_values[6], dtype=np.int64).reshape(-1)
        lw_shape_offsets = np.ascontiguousarray(
            left_values[7], dtype=np.int64
        ).reshape(-1)
        lw_shapes = np.ascontiguousarray(left_values[8], dtype=np.int64).reshape(-1)
        lw_data = np.ascontiguousarray(left_values[9], dtype=np.float64).reshape(-1)
        rb_ids = np.ascontiguousarray(right_values[0], dtype=np.int64).reshape(-1)
        rw_ids = np.ascontiguousarray(right_values[1], dtype=np.int64).reshape(-1)
        rb_offsets = np.ascontiguousarray(right_values[2], dtype=np.int64).reshape(-1)
        rb_shape_offsets = np.ascontiguousarray(
            right_values[3], dtype=np.int64
        ).reshape(-1)
        rb_shapes = np.ascontiguousarray(right_values[4], dtype=np.int64).reshape(-1)
        rb_data = np.ascontiguousarray(right_values[5], dtype=np.float64).reshape(-1)
        rw_offsets = np.ascontiguousarray(right_values[6], dtype=np.int64).reshape(-1)
        rw_shape_offsets = np.ascontiguousarray(
            right_values[7], dtype=np.int64
        ).reshape(-1)
        rw_shapes = np.ascontiguousarray(right_values[8], dtype=np.int64).reshape(-1)
        rw_data = np.ascontiguousarray(right_values[9], dtype=np.float64).reshape(-1)
        if not (
            in_arr.size == out_arr.size
            and in_arr.size == left_arr.size
            and in_arr.size == right_arr.size
            and left_family_mask_arr.size == left_factor_arr.size
            and right_family_mask_arr.size == right_factor_arr.size
            and basis_shape_arr.ndim == 2
            and basis_shape_arr.shape[1] == 4
            and basis_offset_arr.size == basis_shape_arr.shape[0]
            and lb_ids.size == lw_ids.size
            and rb_ids.size == rw_ids.size
            and lb_offsets.size == lb_shape_offsets.size
            and lw_offsets.size == lw_shape_offsets.size
            and rb_offsets.size == rb_shape_offsets.size
            and rw_offsets.size == rw_shape_offsets.size
            and lb_offsets.size >= 2
            and lw_offsets.size >= 2
            and rb_offsets.size >= 2
            and rw_offsets.size >= 2
        ):
            raise ValueError("Raw packed SU(2) factor-route arrays are inconsistent.")
        if self._factor_route_key is not None and self._factor_route_key != key_text:
            self._engine.clear_factor_route_projection()
            self._factor_route_projection_owners = None
        same_topology = self._engine.install_raw_factor_routes(
            key_name,
            <const int32_t*>cnp.PyArray_DATA(in_arr),
            <const int32_t*>cnp.PyArray_DATA(out_arr),
            <const int32_t*>cnp.PyArray_DATA(left_arr),
            <const int32_t*>cnp.PyArray_DATA(right_arr),
            <size_t>in_arr.size,
            sizeof(int32_t),
            sizeof(int32_t),
            <const int64_t*>cnp.PyArray_DATA(basis_offset_arr),
            <const int64_t*>cnp.PyArray_DATA(basis_shape_arr),
            <size_t>basis_shape_arr.shape[0],
            <const int64_t*>cnp.PyArray_DATA(left_factor_arr),
            <size_t>left_factor_arr.size,
            <const int64_t*>cnp.PyArray_DATA(lb_ids),
            <const int64_t*>cnp.PyArray_DATA(lw_ids),
            <size_t>lb_ids.size,
            <const int64_t*>cnp.PyArray_DATA(lb_offsets),
            <const int64_t*>cnp.PyArray_DATA(lb_shape_offsets),
            <size_t>(lb_offsets.size - 1),
            <const int64_t*>cnp.PyArray_DATA(lb_shapes),
            <size_t>lb_shapes.size,
            <const double*>cnp.PyArray_DATA(lb_data),
            <size_t>lb_data.size,
            <const int64_t*>cnp.PyArray_DATA(lw_offsets),
            <const int64_t*>cnp.PyArray_DATA(lw_shape_offsets),
            <size_t>(lw_offsets.size - 1),
            <const int64_t*>cnp.PyArray_DATA(lw_shapes),
            <size_t>lw_shapes.size,
            <const double*>cnp.PyArray_DATA(lw_data),
            <size_t>lw_data.size,
            <const int64_t*>cnp.PyArray_DATA(right_factor_arr),
            <size_t>right_factor_arr.size,
            <const int64_t*>cnp.PyArray_DATA(rb_ids),
            <const int64_t*>cnp.PyArray_DATA(rw_ids),
            <size_t>rb_ids.size,
            <const int64_t*>cnp.PyArray_DATA(rb_offsets),
            <const int64_t*>cnp.PyArray_DATA(rb_shape_offsets),
            <size_t>(rb_offsets.size - 1),
            <const int64_t*>cnp.PyArray_DATA(rb_shapes),
            <size_t>rb_shapes.size,
            <const double*>cnp.PyArray_DATA(rb_data),
            <size_t>rb_data.size,
            <const int64_t*>cnp.PyArray_DATA(rw_offsets),
            <const int64_t*>cnp.PyArray_DATA(rw_shape_offsets),
            <size_t>(rw_offsets.size - 1),
            <const int64_t*>cnp.PyArray_DATA(rw_shapes),
            <size_t>rw_shapes.size,
            <const double*>cnp.PyArray_DATA(rw_data),
            <size_t>rw_data.size,
            <const uint64_t*>cnp.PyArray_DATA(left_family_mask_arr),
            <const uint64_t*>cnp.PyArray_DATA(right_family_mask_arr),
            sizeof(uint64_t),
            <size_t>total_dimension,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
            <cpp_bool>False,
            <cpp_bool>False,
        )
        self._factor_route_key = key_text
        self._factor_route_revision = (
            key_text,
            int(topology_revision),
            int(numeric_revision),
            int(total_dimension),
        )
        self._factor_route_owners = (
            lb_ids, lw_ids, lb_offsets, lb_shape_offsets, lb_shapes, lb_data,
            lw_offsets, lw_shape_offsets, lw_shapes, lw_data,
            rb_ids, rw_ids, rb_offsets, rb_shape_offsets, rb_shapes, rb_data,
            rw_offsets, rw_shape_offsets, rw_shapes, rw_data,
            left_family_mask_arr, right_family_mask_arr,
        )
        return bool(same_topology)

    def clear_factor_routes(self):
        self._engine.clear_factor_routes()
        self._factor_route_owners = None
        self._factor_route_projection_owners = None
        self._factorized_metric_owners = None
        self._factor_route_key = None
        self._factor_route_revision = None

    def set_factor_routes_hermitianized(self, bint enabled=True):
        self._engine.set_factor_routes_hermitianized(<cpp_bool>enabled)

    def factor_route_matvec(self, object key, object vector):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        input_arr = np.ascontiguousarray(vector, dtype=np.complex128).reshape(-1)
        output_arr = np.empty(input_arr.size, dtype=np.complex128)
        self._engine.factor_route_matvec(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(input_arr),
            <cpp_complex[double]*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
        )
        return output_arr

    def factor_route_real_matvec(self, object key, object vector):
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        input_arr = np.ascontiguousarray(vector, dtype=np.float64).reshape(-1)
        output_arr = np.empty(input_arr.size, dtype=np.float64)
        self._engine.factor_route_real_matvec(
            key_name,
            <const double*>cnp.PyArray_DATA(input_arr),
            <double*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
        )
        return output_arr

    def factor_route_davidson(
        self,
        object key,
        object diagonal,
        object guess,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bint accept_unconverged=False,
    ):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] diagonal_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] guess_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] vector_arr
        cdef string key_name = str(key).encode()
        cdef CppDavidsonResult result
        cdef Py_ssize_t index
        diagonal_arr = np.ascontiguousarray(
            diagonal,
            dtype=np.complex128,
        ).reshape(-1)
        guess_arr = np.ascontiguousarray(
            guess,
            dtype=np.complex128,
        ).reshape(-1)
        if diagonal_arr.size != guess_arr.size:
            raise ValueError("Davidson diagonal and guess dimensions differ.")
        result = self._engine.factor_route_davidson(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(diagonal_arr),
            <const cpp_complex[double]*>cnp.PyArray_DATA(guess_arr),
            <size_t>diagonal_arr.size,
            tolerance,
            max_iterations,
            restart_dimension,
            <cpp_bool>accept_unconverged,
        )
        vector_arr = np.empty(result.vector.size(), dtype=np.complex128)
        for index in range(result.vector.size()):
            vector_arr[index] = complex(
                result.vector[index].real(),
                result.vector[index].imag(),
            )
        return {
            "kind": "cpp_su2_factor_route_davidson",
            "accepted": bool(result.accepted),
            "energy": float(result.energy),
            "vector": vector_arr,
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "basis_size": int(result.basis_size),
            "restarts": int(result.restarts),
            "converged": bool(result.converged),
            "workspace_reused": bool(result.workspace_reused),
            "matvec_calls": int(result.matvec_calls),
        }

    def active_bond_complementary_davidson(
        self,
        object key,
        object guess,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bint accept_unconverged=False,
    ):
        """Apply S/R/A/P/B/Q actions and solve the active bond in C++."""

        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] guess_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] vector_arr
        cdef string key_name = str(key).encode()
        cdef CppDavidsonResult result
        cdef Py_ssize_t index
        guess_arr = np.ascontiguousarray(
            guess,
            dtype=np.complex128,
        ).reshape(-1)
        result = self._engine.active_bond_complementary_davidson(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(guess_arr),
            <size_t>guess_arr.size,
            tolerance,
            max_iterations,
            restart_dimension,
            <cpp_bool>accept_unconverged,
        )
        vector_arr = np.empty(result.vector.size(), dtype=np.complex128)
        for index in range(result.vector.size()):
            vector_arr[index] = complex(
                result.vector[index].real(),
                result.vector[index].imag(),
            )
        return {
            "kind": "cpp_su2_active_complementary_davidson",
            "accepted": bool(result.accepted),
            "energy": float(result.energy),
            "vector": vector_arr,
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "basis_size": int(result.basis_size),
            "restarts": int(result.restarts),
            "converged": bool(result.converged),
            "workspace_reused": bool(result.workspace_reused),
            "matvec_calls": int(result.matvec_calls),
        }

    def solve_active_bond_canonical(
        self,
        object metric_key=None,
        double projection_tolerance=1.0e-12,
        long long max_component_elements=4194304,
        long long max_transform_elements=8388608,
        double davidson_tolerance=1.0e-8,
        int max_iterations=100,
        object max_space=None,
        long long workspace_budget_bytes=33554432,
        long long workspace_basis_arrays=3,
        bint accept_unconverged=True,
        bint prepare_owned=False,
        object left_boundary_bond=None,
        object right_boundary_bond=None,
        bint dual_right_basis=True,
    ):
        """Solve the active canonical bond without exporting its vector."""

        cdef string metric_key_name
        cdef int requested_restart_dimension = (
            -1 if max_space is None else int(max_space)
        )
        cdef CppActiveBondCanonicalSolveResult result
        cdef string projection_key_name
        if (
            max_component_elements <= 0
            or max_transform_elements <= 0
            or workspace_budget_bytes <= 0
            or workspace_basis_arrays <= 0
        ):
            return {"compatible": False}
        if prepare_owned:
            if left_boundary_bond is None or right_boundary_bond is None:
                raise ValueError(
                    "Owned active-bond preparation requires both boundary bonds."
                )
            result = (
                self._engine.prepare_and_solve_active_bond_canonical(
                    <int64_t>int(left_boundary_bond),
                    <int64_t>int(right_boundary_bond),
                    <cpp_bool>dual_right_basis,
                    projection_tolerance,
                    <size_t>max_component_elements,
                    <size_t>max_transform_elements,
                    davidson_tolerance,
                    max_iterations,
                    requested_restart_dimension,
                    <size_t>workspace_budget_bytes,
                    <size_t>workspace_basis_arrays,
                    <cpp_bool>accept_unconverged,
                )
            )
        else:
            if metric_key is None:
                raise ValueError(
                    "metric_key is required unless prepare_owned is true."
                )
            metric_key_name = str(metric_key).encode()
            result = self._engine.solve_active_bond_canonical(
                metric_key_name,
                projection_tolerance,
                <size_t>max_component_elements,
                <size_t>max_transform_elements,
                davidson_tolerance,
                max_iterations,
                requested_restart_dimension,
                <size_t>workspace_budget_bytes,
                <size_t>workspace_basis_arrays,
                <cpp_bool>accept_unconverged,
            )
        projection_key_name = result.projection.projection_key
        return {
            "kind": "cpp_su2_active_canonical_solve",
            "prepared_owned": bool(prepare_owned),
            "complementary_action_status": int(
                result.complementary_action_status
            ),
            "metric_routes": int(result.metric_routes),
            "compatible": bool(result.projection.compatible),
            "accepted": bool(result.davidson.accepted),
            "energy": float(result.davidson.energy),
            "residual_norm": float(result.davidson.residual_norm),
            "iterations": int(result.davidson.iterations),
            "basis_size": int(result.davidson.basis_size),
            "restarts": int(result.davidson.restarts),
            "converged": bool(result.davidson.converged),
            "workspace_reused": bool(
                result.davidson.workspace_reused
            ),
            "matvec_calls": int(result.davidson.matvec_calls),
            "projection_key": (
                (<bytes>projection_key_name).decode()
                if result.projection.compatible
                else None
            ),
            "parent_dimension": int(
                result.projection.parent_dimension
            ),
            "orthonormal_dimension": int(
                result.projection.orthonormal_dimension
            ),
            "projection_reused": bool(result.projection.reused),
            "projection_components": int(
                result.projection.components
            ),
            "projection_max_component_dimension": int(
                result.projection.max_component_dimension
            ),
            "projection_transform_elements": int(
                result.projection.transform_elements
            ),
            "projection_whitening_residual": float(
                result.projection.whitening_residual
            ),
            "projection_build_seconds": float(
                result.projection.build_seconds
            ),
            "requested_max_space": int(
                result.requested_restart_dimension
            ),
            "workspace_max_space": int(
                result.workspace_restart_dimension
            ),
            "estimated_basis_workspace_bytes": int(
                result.estimated_workspace_bytes
            ),
            "solve_seconds": float(result.solve_seconds),
        }

    def active_bond_complementary_action_ready(
        self,
        object key,
        long long dimension,
    ):
        cdef string key_name = str(key).encode()
        return bool(
            self._engine.active_bond_complementary_action_ready(
                key_name,
                <size_t>dimension,
            )
        )

    def solve_active_bond_state_average(
        self,
        long long left_boundary_bond,
        long long right_boundary_bond,
        double projection_tolerance=1.0e-12,
        long long max_component_elements=4194304,
        long long max_transform_elements=8388608,
        double davidson_tolerance=1.0e-8,
        int max_iterations=100,
        object max_space=None,
        long long workspace_budget_bytes=67108864,
        long long workspace_basis_arrays=3,
        bint accept_unconverged=True,
        bint dual_right_basis=True,
    ):
        """Solve all roots on the shared active SU(2) bond in C++."""

        cdef int requested_restart_dimension = (
            -1 if max_space is None else int(max_space)
        )
        cdef CppActiveBondStateAverageSolveResult result
        cdef string projection_key_name
        cdef Py_ssize_t index
        if self._engine.state_average_roots() < 2:
            raise RuntimeError("No C++ state-average center bundle is installed.")
        result = self._engine.prepare_and_solve_active_bond_state_average(
            <int64_t>left_boundary_bond,
            <int64_t>right_boundary_bond,
            <cpp_bool>dual_right_basis,
            projection_tolerance,
            <size_t>max_component_elements,
            <size_t>max_transform_elements,
            davidson_tolerance,
            max_iterations,
            requested_restart_dimension,
            <size_t>workspace_budget_bytes,
            <size_t>workspace_basis_arrays,
            <cpp_bool>accept_unconverged,
        )
        projection_key_name = result.projection.projection_key
        return {
            "kind": "cpp_su2_state_average_block_davidson",
            "complementary_action_status": int(
                result.complementary_action_status
            ),
            "metric_routes": int(result.metric_routes),
            "compatible": bool(result.projection.compatible),
            "accepted": bool(result.davidson.accepted),
            "energies": [
                float(result.davidson.energies[index])
                for index in range(result.davidson.energies.size())
            ],
            "residual_norms": [
                float(result.davidson.residual_norms[index])
                for index in range(result.davidson.residual_norms.size())
            ],
            "iterations": int(result.davidson.iterations),
            "basis_size": int(result.davidson.basis_size),
            "restarts": int(result.davidson.restarts),
            "converged": bool(result.davidson.converged),
            "workspace_reused": bool(result.davidson.workspace_reused),
            "matvec_calls": int(result.davidson.matvec_calls),
            "projection_key": (
                (<bytes>projection_key_name).decode()
                if result.projection.compatible
                else None
            ),
            "parent_dimension": int(result.projection.parent_dimension),
            "orthonormal_dimension": int(
                result.projection.orthonormal_dimension
            ),
            "projection_reused": bool(result.projection.reused),
            "projection_components": int(result.projection.components),
            "projection_build_seconds": float(
                result.projection.build_seconds
            ),
            "requested_max_space": int(
                result.requested_restart_dimension
            ),
            "workspace_max_space": int(
                result.workspace_restart_dimension
            ),
            "estimated_basis_workspace_bytes": int(
                result.estimated_workspace_bytes
            ),
            "solve_seconds": float(result.solve_seconds),
        }

    def factor_route_installed(self, object key, long long dimension):
        cdef string key_name = str(key).encode()
        return bool(
            self._engine.factor_route_installed(
                key_name,
                <size_t>dimension,
            )
        )

    def factor_route_diagonal(self, object key, long long dimension):
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        output_arr = np.empty(dimension, dtype=np.float64)
        self._engine.factor_route_diagonal(
            key_name,
            <double*>cnp.PyArray_DATA(output_arr),
            <size_t>dimension,
        )
        return output_arr

    def install_factor_route_projection(
        self,
        object key,
        object factor_route_key,
        object component_indices,
        object component_transforms,
        object orthonormal_offsets,
        long long parent_dimension,
        long long orthonormal_dimension,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef tuple index_values = tuple(component_indices)
        cdef tuple transform_values = tuple(component_transforms)
        cdef tuple offset_values = tuple(orthonormal_offsets)
        cdef Py_ssize_t n_components = len(index_values)
        cdef Py_ssize_t component
        cdef cnp.ndarray index_arr
        cdef cnp.ndarray transform_arr
        cdef const double* real_transform
        cdef const cpp_complex[double]* complex_transform
        cdef string key_name = str(key).encode()
        cdef string factor_key_name = str(factor_route_key).encode()
        cdef bint same_topology
        cdef list owners = []

        if (
            n_components == 0
            or len(transform_values) != n_components
            or len(offset_values) != n_components
        ):
            raise ValueError("Projection components and offsets must align.")
        same_topology = self._engine.begin_factor_route_projection(
            key_name,
            factor_key_name,
            <size_t>parent_dimension,
            <size_t>orthonormal_dimension,
            <size_t>n_components,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        )
        for component in range(n_components):
            index_arr = np.ascontiguousarray(
                index_values[component],
                dtype=np.int64,
            ).reshape(-1)
            if np.iscomplexobj(transform_values[component]):
                transform_arr = np.ascontiguousarray(
                    transform_values[component],
                    dtype=np.complex128,
                )
                real_transform = NULL
                complex_transform = (
                    <const cpp_complex[double]*>cnp.PyArray_DATA(transform_arr)
                )
            else:
                transform_arr = np.ascontiguousarray(
                    transform_values[component],
                    dtype=np.float64,
                )
                real_transform = <const double*>cnp.PyArray_DATA(transform_arr)
                complex_transform = NULL
            if (
                transform_arr.ndim != 2
                or transform_arr.shape[0] != index_arr.size
            ):
                raise ValueError(
                    "Each projection transform must align with its parent indices."
                )
            self._engine.install_factor_route_projection_component(
                <size_t>component,
                <const int64_t*>cnp.PyArray_DATA(index_arr),
                <size_t>index_arr.size,
                real_transform,
                complex_transform,
                <size_t>transform_arr.shape[1],
                <size_t>int(offset_values[component]),
            )
            owners.append((index_arr, transform_arr))
        self._engine.finish_factor_route_projection()
        self._factor_route_projection_owners = tuple(owners)
        return bool(same_topology)

    def install_indexed_factor_route_projection(
        self,
        object key,
        object factor_route_key,
        object blocks,
        long long parent_dimension,
        long long orthonormal_dimension,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef tuple block_values = tuple(blocks)
        cdef Py_ssize_t n_components = len(block_values)
        cdef Py_ssize_t component
        cdef object row_slice
        cdef object orthonormal_indices
        cdef object transform
        cdef object block
        cdef bint is_kronecker
        cdef cnp.ndarray parent_index_arr
        cdef cnp.ndarray orthonormal_index_arr
        cdef cnp.ndarray transform_arr
        cdef const double* real_transform
        cdef const cpp_complex[double]* complex_transform
        cdef string key_name = str(key).encode()
        cdef string factor_key_name = str(factor_route_key).encode()
        cdef bint same_topology
        cdef list owners = []

        if n_components == 0:
            raise ValueError(
                "An indexed factor-route projection requires blocks."
            )
        same_topology = self._engine.begin_factor_route_projection(
            key_name,
            factor_key_name,
            <size_t>parent_dimension,
            <size_t>orthonormal_dimension,
            <size_t>n_components,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        )
        for component in range(n_components):
            block = block_values[component]
            is_kronecker = hasattr(block, "local_transform")
            if is_kronecker:
                row_slice = block.row_slice
                orthonormal_indices = block.orthonormal_indices
                transform = block.local_transform
            else:
                row_slice, orthonormal_indices, transform = block
            orthonormal_index_arr = np.ascontiguousarray(
                orthonormal_indices,
                dtype=np.int64,
            ).reshape(-1)
            if np.iscomplexobj(transform):
                transform_arr = np.ascontiguousarray(
                    transform,
                    dtype=np.complex128,
                )
                real_transform = NULL
                complex_transform = (
                    <const cpp_complex[double]*>cnp.PyArray_DATA(transform_arr)
                )
            else:
                transform_arr = np.ascontiguousarray(
                    transform,
                    dtype=np.float64,
                )
                real_transform = <const double*>cnp.PyArray_DATA(transform_arr)
                complex_transform = NULL
            if is_kronecker:
                if (
                    transform_arr.ndim != 2
                    or transform_arr.shape[0] != int(block.selected_dim)
                    or transform_arr.shape[1] != int(block.local_dim)
                ):
                    raise ValueError(
                        "Kronecker transform dimensions do not match its block."
                    )
                self._engine.install_factor_route_projection_kronecker_component(
                    <size_t>component,
                    <size_t>int(row_slice.start),
                    <const int64_t*>cnp.PyArray_DATA(orthonormal_index_arr),
                    <size_t>orthonormal_index_arr.size,
                    <size_t>int(block.left_dim),
                    <size_t>int(block.selected_dim),
                    <size_t>int(block.local_dim),
                    <size_t>int(block.right_dim),
                    real_transform,
                    complex_transform,
                )
                owners.append((orthonormal_index_arr, transform_arr))
            else:
                parent_index_arr = np.arange(
                    int(row_slice.start),
                    int(row_slice.stop),
                    dtype=np.int64,
                )
                if (
                    transform_arr.ndim != 2
                    or transform_arr.shape[0] != parent_index_arr.size
                    or transform_arr.shape[1] != orthonormal_index_arr.size
                ):
                    raise ValueError(
                        "Indexed transform dimensions do not match its indices."
                    )
                self._engine.install_factor_route_projection_indexed_component(
                    <size_t>component,
                    <const int64_t*>cnp.PyArray_DATA(parent_index_arr),
                    <size_t>parent_index_arr.size,
                    <const int64_t*>cnp.PyArray_DATA(orthonormal_index_arr),
                    <size_t>orthonormal_index_arr.size,
                    real_transform,
                    complex_transform,
                )
                owners.append(
                    (parent_index_arr, orthonormal_index_arr, transform_arr)
                )
        self._engine.finish_factor_route_projection()
        self._factor_route_projection_owners = tuple(owners)
        return bool(same_topology)

    def clear_factor_route_projection(self):
        self._engine.clear_factor_route_projection()
        self._factor_route_projection_owners = None

    def factor_route_projected_matvec(self, object key, object vector):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        input_arr = np.ascontiguousarray(vector, dtype=np.complex128).reshape(-1)
        output_arr = np.empty(input_arr.size, dtype=np.complex128)
        self._engine.factor_route_projected_matvec(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(input_arr),
            <cpp_complex[double]*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
        )
        return output_arr

    def factor_route_projected_real_matvec(self, object key, object vector):
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        input_arr = np.ascontiguousarray(vector, dtype=np.float64).reshape(-1)
        output_arr = np.empty(input_arr.size, dtype=np.float64)
        self._engine.factor_route_projected_real_matvec(
            key_name,
            <const double*>cnp.PyArray_DATA(input_arr),
            <double*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
        )
        return output_arr

    def factor_route_projected_davidson(
        self,
        object key,
        object diagonal,
        object guess,
        double tolerance,
        int max_iterations,
        int restart_dimension,
        bint accept_unconverged=False,
    ):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] diagonal_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] guess_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] vector_arr
        cdef string key_name = str(key).encode()
        cdef CppDavidsonResult result
        cdef Py_ssize_t index
        diagonal_arr = np.ascontiguousarray(diagonal, dtype=np.complex128).reshape(-1)
        guess_arr = np.ascontiguousarray(guess, dtype=np.complex128).reshape(-1)
        if diagonal_arr.size != guess_arr.size:
            raise ValueError("Davidson diagonal and guess dimensions differ.")
        result = self._engine.factor_route_projected_davidson(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(diagonal_arr),
            <const cpp_complex[double]*>cnp.PyArray_DATA(guess_arr),
            <size_t>diagonal_arr.size,
            tolerance,
            max_iterations,
            restart_dimension,
            <cpp_bool>accept_unconverged,
        )
        vector_arr = np.empty(result.vector.size(), dtype=np.complex128)
        for index in range(result.vector.size()):
            vector_arr[index] = complex(
                result.vector[index].real(),
                result.vector[index].imag(),
            )
        return {
            "kind": "cpp_su2_factor_route_davidson",
            "accepted": bool(result.accepted),
            "energy": float(result.energy),
            "vector": vector_arr,
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "basis_size": int(result.basis_size),
            "restarts": int(result.restarts),
            "converged": bool(result.converged),
            "workspace_reused": bool(result.workspace_reused),
            "matvec_calls": int(result.matvec_calls),
        }

    def install_factorized_metric(
        self,
        object key,
        object routes,
        long long dimension,
        unsigned long long topology_revision,
        unsigned long long numeric_revision,
    ):
        cdef tuple route_values = tuple(routes)
        cdef Py_ssize_t n_routes = len(route_values)
        cdef Py_ssize_t route_index
        cdef object route
        cdef object input_entry
        cdef object output_entry
        cdef object left_value
        cdef object right_value
        cdef cnp.ndarray input_shape
        cdef cnp.ndarray output_shape
        cdef cnp.ndarray left_arr
        cdef cnp.ndarray right_arr
        cdef const double* left_real
        cdef const cpp_complex[double]* left_complex
        cdef const double* right_real
        cdef const cpp_complex[double]* right_complex
        cdef string key_name = str(key).encode()
        cdef list owners = []
        cdef bint same_topology

        if n_routes == 0:
            raise ValueError("A factorized metric requires at least one route.")
        same_topology = self._engine.begin_factorized_metric(
            key_name,
            <size_t>dimension,
            <size_t>n_routes,
            <uint64_t>topology_revision,
            <uint64_t>numeric_revision,
        )
        for route_index in range(n_routes):
            route = route_values[route_index]
            if len(route) != 4:
                raise ValueError(
                    "Metric routes must be (input_entry, output_entry, left, right)."
                )
            input_entry, output_entry, left_value, right_value = route
            input_shape = np.ascontiguousarray(
                input_entry.shape,
                dtype=np.int64,
            ).reshape(-1)
            output_shape = np.ascontiguousarray(
                output_entry.shape,
                dtype=np.int64,
            ).reshape(-1)
            if input_shape.size != 4 or output_shape.size != 4:
                raise ValueError("Metric route entries must have rank four.")
            left_arr = np.ascontiguousarray(left_value)
            right_arr = np.ascontiguousarray(right_value)
            if left_arr.ndim != 2 or right_arr.ndim != 2:
                raise ValueError("Metric factors must be matrices.")
            if (
                left_arr.shape[0] != output_shape[0]
                or left_arr.shape[1] != input_shape[0]
                or right_arr.shape[0] != output_shape[3]
                or right_arr.shape[1] != input_shape[3]
            ):
                raise ValueError("Metric factor and route dimensions differ.")
            left_real = NULL
            left_complex = NULL
            right_real = NULL
            right_complex = NULL
            if np.iscomplexobj(left_arr):
                left_arr = np.ascontiguousarray(left_arr, dtype=np.complex128)
                left_complex = (
                    <const cpp_complex[double]*>cnp.PyArray_DATA(left_arr)
                )
            else:
                left_arr = np.ascontiguousarray(left_arr, dtype=np.float64)
                left_real = <const double*>cnp.PyArray_DATA(left_arr)
            if np.iscomplexobj(right_arr):
                right_arr = np.ascontiguousarray(right_arr, dtype=np.complex128)
                right_complex = (
                    <const cpp_complex[double]*>cnp.PyArray_DATA(right_arr)
                )
            else:
                right_arr = np.ascontiguousarray(right_arr, dtype=np.float64)
                right_real = <const double*>cnp.PyArray_DATA(right_arr)
            self._engine.install_factorized_metric_route(
                <size_t>route_index,
                <int64_t>int(input_entry.offset),
                <int64_t>int(output_entry.offset),
                <const int64_t*>cnp.PyArray_DATA(input_shape),
                <const int64_t*>cnp.PyArray_DATA(output_shape),
                left_real,
                left_complex,
                right_real,
                right_complex,
            )
            owners.append(
                (input_shape, output_shape, left_arr, right_arr)
            )
        self._engine.finish_factorized_metric()
        self._factorized_metric_owners = tuple(owners)
        return bool(same_topology)

    def clear_factorized_metric(self):
        self._engine.clear_factorized_metric()
        self._factorized_metric_owners = None

    def factorized_metric_matvec(self, object key, object vector):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        input_arr = np.ascontiguousarray(vector, dtype=np.complex128).reshape(-1)
        output_arr = np.empty(input_arr.size, dtype=np.complex128)
        self._engine.factorized_metric_matvec(
            key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(input_arr),
            <cpp_complex[double]*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
        )
        return output_arr

    def factorized_metric_real_matvec(self, object key, object vector):
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        input_arr = np.ascontiguousarray(vector, dtype=np.float64).reshape(-1)
        output_arr = np.empty(input_arr.size, dtype=np.float64)
        self._engine.factorized_metric_real_matvec(
            key_name,
            <const double*>cnp.PyArray_DATA(input_arr),
            <double*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
        )
        return output_arr

    def factorized_metric_real_diagonal(self, object key, long long dimension):
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] output_arr
        cdef string key_name = str(key).encode()
        if dimension <= 0:
            raise ValueError("Metric dimension must be positive.")
        output_arr = np.empty(dimension, dtype=np.float64)
        self._engine.factorized_metric_real_diagonal(
            key_name,
            <double*>cnp.PyArray_DATA(output_arr),
            <size_t>dimension,
        )
        return output_arr

    def prepare_canonical_reduced_projection(
        self,
        object metric_key,
        double tolerance=1.0e-12,
        long long max_component_elements=4194304,
        long long max_transform_elements=8388608,
    ):
        """Build an owned blockwise metric whitening beside the active routes."""

        cdef string metric_key_name = str(metric_key).encode()
        cdef CppCanonicalProjectionInfo result
        cdef string projection_key_name
        if max_component_elements <= 0 or max_transform_elements <= 0:
            return {"compatible": False}
        result = self._engine.prepare_canonical_reduced_projection(
            metric_key_name,
            tolerance,
            <size_t>max_component_elements,
            <size_t>max_transform_elements,
        )
        projection_key_name = result.projection_key
        return {
            "compatible": bool(result.compatible),
            "reused": bool(result.reused),
            "projection_key": (
                (<bytes>projection_key_name).decode()
                if result.compatible
                else None
            ),
            "parent_dimension": int(result.parent_dimension),
            "orthonormal_dimension": int(
                result.orthonormal_dimension
            ),
            "components": int(result.components),
            "max_component_dimension": int(
                result.max_component_dimension
            ),
            "transform_elements": int(result.transform_elements),
            "whitening_residual": float(result.whitening_residual),
            "build_seconds": float(result.build_seconds),
        }

    def canonical_reduced_projection_guess(
        self,
        object projection_key,
        object metric_key,
        object parent_guess,
        long long orthonormal_dimension,
    ):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] parent_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] output_arr
        cdef string projection_key_name = str(projection_key).encode()
        cdef string metric_key_name = str(metric_key).encode()
        if orthonormal_dimension <= 0:
            raise ValueError("Orthonormal dimension must be positive.")
        parent_arr = np.ascontiguousarray(
            parent_guess,
            dtype=np.complex128,
        ).reshape(-1)
        output_arr = np.empty(
            orthonormal_dimension,
            dtype=np.complex128,
        )
        self._engine.canonical_reduced_projection_guess(
            projection_key_name,
            metric_key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(parent_arr),
            <cpp_complex[double]*>cnp.PyArray_DATA(output_arr),
            <size_t>parent_arr.size,
            <size_t>output_arr.size,
        )
        return output_arr

    def lift_factor_route_projection_vector(
        self,
        object projection_key,
        object orthonormal_vector,
        long long parent_dimension,
    ):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] input_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] output_arr
        cdef string projection_key_name = str(projection_key).encode()
        if parent_dimension <= 0:
            raise ValueError("Parent dimension must be positive.")
        input_arr = np.ascontiguousarray(
            orthonormal_vector,
            dtype=np.complex128,
        ).reshape(-1)
        output_arr = np.empty(parent_dimension, dtype=np.complex128)
        self._engine.lift_factor_route_projection_vector(
            projection_key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(input_arr),
            <cpp_complex[double]*>cnp.PyArray_DATA(output_arr),
            <size_t>input_arr.size,
            <size_t>output_arr.size,
        )
        return output_arr

    def factor_route_generalized_davidson(
        self,
        object factor_route_key,
        object metric_key,
        object h_diagonal,
        object n_diagonal,
        object guess,
        double energy_tolerance,
        double residual_tolerance,
        double linear_dependence_tolerance,
        int max_iterations,
        int restart_dimension,
        bint accept_unconverged=False,
    ):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] h_diag_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] n_diag_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] guess_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] vector_arr
        cdef string factor_key_name = str(factor_route_key).encode()
        cdef string metric_key_name = str(metric_key).encode()
        cdef CppDavidsonResult result
        cdef Py_ssize_t index
        h_diag_arr = np.ascontiguousarray(
            h_diagonal,
            dtype=np.complex128,
        ).reshape(-1)
        n_diag_arr = np.ascontiguousarray(
            n_diagonal,
            dtype=np.complex128,
        ).reshape(-1)
        guess_arr = np.ascontiguousarray(
            guess,
            dtype=np.complex128,
        ).reshape(-1)
        if (
            h_diag_arr.size != n_diag_arr.size
            or h_diag_arr.size != guess_arr.size
        ):
            raise ValueError(
                "Generalized Davidson diagonal and guess dimensions differ."
            )
        result = self._engine.factor_route_generalized_davidson(
            factor_key_name,
            metric_key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(h_diag_arr),
            <const cpp_complex[double]*>cnp.PyArray_DATA(n_diag_arr),
            <const cpp_complex[double]*>cnp.PyArray_DATA(guess_arr),
            <size_t>guess_arr.size,
            energy_tolerance,
            residual_tolerance,
            linear_dependence_tolerance,
            max_iterations,
            restart_dimension,
            <cpp_bool>accept_unconverged,
        )
        vector_arr = np.empty(result.vector.size(), dtype=np.complex128)
        for index in range(result.vector.size()):
            vector_arr[index] = complex(
                result.vector[index].real(),
                result.vector[index].imag(),
            )
        return {
            "kind": "cpp_su2_factor_route_generalized_davidson",
            "accepted": bool(result.accepted),
            "energy": float(result.energy),
            "vector": vector_arr,
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "basis_size": int(result.basis_size),
            "restarts": int(result.restarts),
            "converged": bool(result.converged),
            "workspace_reused": bool(result.workspace_reused),
            "matvec_calls": int(result.matvec_calls),
            "norm_matvec_calls": int(result.norm_matvec_calls),
        }

    def active_bond_complementary_generalized_davidson(
        self,
        object factor_route_key,
        object metric_key,
        object guess,
        double energy_tolerance,
        double residual_tolerance,
        double linear_dependence_tolerance,
        int max_iterations,
        int restart_dimension,
        bint accept_unconverged=False,
    ):
        """Solve the active complementary action and reduced metric in C++."""

        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] guess_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] vector_arr
        cdef string factor_key_name = str(factor_route_key).encode()
        cdef string metric_key_name = str(metric_key).encode()
        cdef CppDavidsonResult result
        cdef Py_ssize_t index
        guess_arr = np.ascontiguousarray(
            guess,
            dtype=np.complex128,
        ).reshape(-1)
        result = (
            self._engine.active_bond_complementary_generalized_davidson(
                factor_key_name,
                metric_key_name,
                <const cpp_complex[double]*>cnp.PyArray_DATA(guess_arr),
                <size_t>guess_arr.size,
                energy_tolerance,
                residual_tolerance,
                linear_dependence_tolerance,
                max_iterations,
                restart_dimension,
                <cpp_bool>accept_unconverged,
            )
        )
        vector_arr = np.empty(result.vector.size(), dtype=np.complex128)
        for index in range(result.vector.size()):
            vector_arr[index] = complex(
                result.vector[index].real(),
                result.vector[index].imag(),
            )
        return {
            "kind": "cpp_su2_active_complementary_generalized_davidson",
            "accepted": bool(result.accepted),
            "energy": float(result.energy),
            "vector": vector_arr,
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "basis_size": int(result.basis_size),
            "restarts": int(result.restarts),
            "converged": bool(result.converged),
            "workspace_reused": bool(result.workspace_reused),
            "matvec_calls": int(result.matvec_calls),
            "norm_matvec_calls": int(result.norm_matvec_calls),
        }

    def factor_route_projected_generalized_davidson(
        self,
        object projection_key,
        object metric_key,
        object h_diagonal,
        object n_diagonal,
        object guess,
        double energy_tolerance,
        double residual_tolerance,
        double linear_dependence_tolerance,
        int max_iterations,
        int restart_dimension,
        bint accept_unconverged=False,
    ):
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] h_diag_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] n_diag_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] guess_arr
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] vector_arr
        cdef string projection_key_name = str(projection_key).encode()
        cdef string metric_key_name = str(metric_key).encode()
        cdef CppDavidsonResult result
        cdef Py_ssize_t index
        h_diag_arr = np.ascontiguousarray(
            h_diagonal,
            dtype=np.complex128,
        ).reshape(-1)
        n_diag_arr = np.ascontiguousarray(
            n_diagonal,
            dtype=np.complex128,
        ).reshape(-1)
        guess_arr = np.ascontiguousarray(
            guess,
            dtype=np.complex128,
        ).reshape(-1)
        if (
            h_diag_arr.size != n_diag_arr.size
            or h_diag_arr.size != guess_arr.size
        ):
            raise ValueError(
                "Projected generalized Davidson dimensions differ."
            )
        result = self._engine.factor_route_projected_generalized_davidson(
            projection_key_name,
            metric_key_name,
            <const cpp_complex[double]*>cnp.PyArray_DATA(h_diag_arr),
            <const cpp_complex[double]*>cnp.PyArray_DATA(n_diag_arr),
            <const cpp_complex[double]*>cnp.PyArray_DATA(guess_arr),
            <size_t>guess_arr.size,
            energy_tolerance,
            residual_tolerance,
            linear_dependence_tolerance,
            max_iterations,
            restart_dimension,
            <cpp_bool>accept_unconverged,
        )
        vector_arr = np.empty(result.vector.size(), dtype=np.complex128)
        for index in range(result.vector.size()):
            vector_arr[index] = complex(
                result.vector[index].real(),
                result.vector[index].imag(),
            )
        return {
            "kind": "cpp_su2_projected_factor_route_generalized_davidson",
            "accepted": bool(result.accepted),
            "energy": float(result.energy),
            "vector": vector_arr,
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "basis_size": int(result.basis_size),
            "restarts": int(result.restarts),
            "converged": bool(result.converged),
            "workspace_reused": bool(result.workspace_reused),
            "matvec_calls": int(result.matvec_calls),
            "norm_matvec_calls": int(result.norm_matvec_calls),
        }

    def blockwise_svd(
        self,
        object matrices,
        object state_weights,
        double cutoff=1.0e-10,
        object max_bond=None,
        object max_bond_mode="reduced",
        bint retain_sector_topology=False,
    ):
        """Split all reduced bond sectors and truncate them in one C++ call."""

        cdef list packed_matrices = []
        cdef object value
        cdef cnp.ndarray matrix
        cdef Py_ssize_t n_blocks
        cdef Py_ssize_t block
        cdef Py_ssize_t cursor = 0
        cdef Py_ssize_t size
        cdef Py_ssize_t rank
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] packed_values
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offsets
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] rows
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] cols
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] weights
        cdef int64_t max_bond_value
        cdef string mode = str(max_bond_mode).encode()
        cdef CppBlockSVDResult result
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] left_values
        cdef cnp.ndarray[cnp.double_t, ndim=1, mode="c"] singular_values
        cdef cnp.ndarray[cnp.complex128_t, ndim=1, mode="c"] right_values
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] kept_indices
        cdef list left = []
        cdef list singular = []
        cdef list right = []
        cdef list kept = []
        cdef Py_ssize_t begin
        cdef Py_ssize_t end

        for value in matrices:
            matrix = np.ascontiguousarray(value, dtype=np.complex128)
            if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
                raise ValueError("blockwise_svd expects nonempty rank-2 matrices.")
            packed_matrices.append(matrix)
        n_blocks = len(packed_matrices)
        if n_blocks == 0:
            raise ValueError("blockwise_svd requires at least one matrix.")
        weights = np.ascontiguousarray(state_weights, dtype=np.int64).reshape(-1)
        if weights.size != n_blocks or np.any(weights <= 0):
            raise ValueError("state_weights must contain one positive value per block.")
        rows = np.empty(n_blocks, dtype=np.int64)
        cols = np.empty(n_blocks, dtype=np.int64)
        offsets = np.empty(n_blocks + 1, dtype=np.int64)
        offsets[0] = 0
        for block in range(n_blocks):
            matrix = packed_matrices[block]
            rows[block] = matrix.shape[0]
            cols[block] = matrix.shape[1]
            cursor += matrix.size
            offsets[block + 1] = cursor
        packed_values = np.empty(cursor, dtype=np.complex128)
        cursor = 0
        for block in range(n_blocks):
            matrix = packed_matrices[block]
            size = matrix.size
            memcpy(
                <void*>(<cnp.complex128_t*>cnp.PyArray_DATA(packed_values) + cursor),
                <const void*>cnp.PyArray_DATA(matrix),
                size * sizeof(cpp_complex[double]),
            )
            cursor += size
        max_bond_value = -1 if max_bond is None else int(max_bond)
        result = self._engine.blockwise_svd(
            <const cpp_complex[double]*>cnp.PyArray_DATA(packed_values),
            <size_t>packed_values.size,
            <const int64_t*>cnp.PyArray_DATA(offsets),
            <const int64_t*>cnp.PyArray_DATA(rows),
            <const int64_t*>cnp.PyArray_DATA(cols),
            <const int64_t*>cnp.PyArray_DATA(weights),
            <size_t>n_blocks,
            cutoff,
            max_bond_value,
            mode,
            <cpp_bool>retain_sector_topology,
        )
        left_values = np.empty(result.left_values.size(), dtype=np.complex128)
        singular_values = np.empty(result.singular_values.size(), dtype=np.float64)
        right_values = np.empty(result.right_values.size(), dtype=np.complex128)
        kept_indices = np.empty(result.kept_indices.size(), dtype=np.int64)
        if result.left_values.size():
            memcpy(
                <void*>cnp.PyArray_DATA(left_values),
                <const void*>result.left_values.data(),
                result.left_values.size() * sizeof(cpp_complex[double]),
            )
        if result.singular_values.size():
            memcpy(
                <void*>cnp.PyArray_DATA(singular_values),
                <const void*>result.singular_values.data(),
                result.singular_values.size() * sizeof(double),
            )
        if result.right_values.size():
            memcpy(
                <void*>cnp.PyArray_DATA(right_values),
                <const void*>result.right_values.data(),
                result.right_values.size() * sizeof(cpp_complex[double]),
            )
        if result.kept_indices.size():
            memcpy(
                <void*>cnp.PyArray_DATA(kept_indices),
                <const void*>result.kept_indices.data(),
                result.kept_indices.size() * sizeof(int64_t),
            )
        for block in range(n_blocks):
            rank = min(rows[block], cols[block])
            begin = result.left_offsets[block]
            end = result.left_offsets[block + 1]
            left.append(left_values[begin:end].reshape((rows[block], rank)))
            begin = result.singular_offsets[block]
            end = result.singular_offsets[block + 1]
            singular.append(singular_values[begin:end])
            begin = result.right_offsets[block]
            end = result.right_offsets[block + 1]
            right.append(right_values[begin:end].reshape((rank, cols[block])))
            begin = result.kept_offsets[block]
            end = result.kept_offsets[block + 1]
            kept.append(kept_indices[begin:end])
        return {
            "kind": "cpp_su2_blockwise_svd",
            "left": left,
            "singular_values": singular,
            "right": right,
            "kept_indices": kept,
            "truncation_error": float(result.truncation_error),
            "full_squared_norm": float(result.full_squared_norm),
            "kept_squared_norm": float(result.kept_squared_norm),
        }

    def active_bond_solution_ready(self):
        return bool(self._engine.active_bond_solution_ready())

    def split_active_bond_solution(
        self,
        double cutoff=1.0e-10,
        object max_bond=None,
        object max_bond_mode="reduced",
        bint retain_sector_topology=False,
        object absorb="right",
        bint install_and_stage=True,
    ):
        """Truncate the C++-owned active solution and install both new sites."""

        cdef int64_t max_bond_value = (
            -1 if max_bond is None else int(max_bond)
        )
        cdef string mode = str(max_bond_mode).encode()
        cdef string absorb_name = str(absorb).encode()
        cdef CppActiveBondSplitResult result
        cdef CppPackedSiteTensor left
        cdef CppPackedSiteTensor right
        cdef dict left_packed
        cdef dict right_packed
        result = self._engine.split_active_bond_solution(
            cutoff,
            max_bond_value,
            mode,
            <cpp_bool>retain_sector_topology,
            absorb_name,
            <cpp_bool>install_and_stage,
            <cpp_bool>True,
        )
        left = result.left
        right = result.right
        left_packed = {
            "values": np.asarray(
                [left.values[index] for index in range(left.values.size())],
                dtype=np.float64,
            ),
            "offsets": np.asarray(
                [left.offsets[index] for index in range(left.offsets.size())],
                dtype=np.int64,
            ),
            "labels": np.asarray(
                [left.labels[index] for index in range(left.labels.size())],
                dtype=np.int64,
            ).reshape((-1, 6)),
            "shape_offsets": np.asarray(
                [
                    left.shape_offsets[index]
                    for index in range(left.shape_offsets.size())
                ],
                dtype=np.int64,
            ),
            "shapes": np.asarray(
                [left.shapes[index] for index in range(left.shapes.size())],
                dtype=np.int64,
            ),
        }
        right_packed = {
            "values": np.asarray(
                [right.values[index] for index in range(right.values.size())],
                dtype=np.float64,
            ),
            "offsets": np.asarray(
                [right.offsets[index] for index in range(right.offsets.size())],
                dtype=np.int64,
            ),
            "labels": np.asarray(
                [right.labels[index] for index in range(right.labels.size())],
                dtype=np.int64,
            ).reshape((-1, 6)),
            "shape_offsets": np.asarray(
                [
                    right.shape_offsets[index]
                    for index in range(right.shape_offsets.size())
                ],
                dtype=np.int64,
            ),
            "shapes": np.asarray(
                [right.shapes[index] for index in range(right.shapes.size())],
                dtype=np.int64,
            ),
        }

        return {
            "kind": "cpp_su2_active_bond_split",
            "left": left_packed,
            "right": right_packed,
            "bond_labels": np.asarray(
                [
                    result.bond_labels[index]
                    for index in range(result.bond_labels.size())
                ],
                dtype=np.int64,
            ).reshape((-1, 2)),
            "bond_dims": np.asarray(
                [
                    result.bond_dims[index]
                    for index in range(result.bond_dims.size())
                ],
                dtype=np.int64,
            ),
            "singular_values": np.asarray(
                [
                    result.singular_values[index]
                    for index in range(result.singular_values.size())
                ],
                dtype=np.float64,
            ),
            "singular_offsets": np.asarray(
                [
                    result.singular_offsets[index]
                    for index in range(result.singular_offsets.size())
                ],
                dtype=np.int64,
            ),
            "truncation_error": float(result.truncation_error),
            "full_squared_norm": float(result.full_squared_norm),
            "kept_squared_norm": float(result.kept_squared_norm),
            "kept_states": int(result.kept_states),
            "left_revision": (
                int(result.left_topology_revision),
                int(result.left_numeric_revision),
            ),
            "right_revision": (
                int(result.right_topology_revision),
                int(result.right_numeric_revision),
            ),
        }

    def record_cpp_split_site(
        self,
        long long site,
        object tensor,
        object revision,
    ):
        """Attach a Python tensor view to a site already installed by C++."""

        cdef tuple keys = tuple(tensor.data)
        cdef tuple rev = tuple(revision)
        cdef object metadata
        if len(rev) != 2:
            raise ValueError("C++ split-site revision must have two entries.")
        self._split_site_keys[int(site)] = keys
        self._split_site_key_indices[int(site)] = {
            key: int(index) for index, key in enumerate(keys)
        }
        self._split_site_topologies.pop(int(site), None)
        self._split_site_revisions[int(site)] = (
            int(rev[0]),
            int(rev[1]),
        )
        metadata = getattr(tensor, "metadata", None)
        if isinstance(metadata, dict):
            metadata["_cpp_split_site"] = (
                int(self._engine.split_site_owner_revision()),
                int(site),
                int(rev[0]),
                int(rev[1]),
            )

    def sweep_bonds(self, object direction, long long n_sites):
        cdef string direction_name = str(direction).encode()
        cdef vector[int64_t] bonds = self._engine.sweep_bonds(
            direction_name,
            <int64_t>n_sites,
        )
        return tuple(int(bonds[index]) for index in range(bonds.size()))

    def begin_half_sweep(self, object direction, long long n_sites):
        cdef string direction_name = str(direction).encode()
        self._engine.begin_half_sweep(direction_name, <int64_t>n_sites)

    def prepare_owned_half_sweep(self):
        """Build any missing fixed-side reduced boundaries in C++."""

        self._engine.prepare_owned_half_sweep()

    def owned_half_sweep_ready(self):
        """Return whether C++ owns current site and boundary topology."""

        return bool(self._engine.owned_half_sweep_ready())

    def owned_half_sweep_readiness_code(self):
        return int(self._engine.owned_half_sweep_readiness_code())

    def execute_owned_half_sweep(
        self,
        double cutoff=1.0e-10,
        object max_bond=None,
        object max_bond_mode="reduced",
        bint retain_sector_topology=False,
        double projection_tolerance=1.0e-12,
        long long max_component_elements=4194304,
        long long max_transform_elements=8388608,
        double davidson_tolerance=1.0e-8,
        int max_iterations=100,
        object max_space=None,
        long long workspace_budget_bytes=33554432,
        long long workspace_basis_arrays=3,
        bint accept_unconverged=True,
    ):
        """Solve, split, and advance every bond without Python callbacks."""

        cdef int64_t max_bond_value = (
            -1 if max_bond is None else int(max_bond)
        )
        cdef int requested_restart_dimension = (
            -1 if max_space is None else int(max_space)
        )
        cdef string mode = str(max_bond_mode).encode()
        cdef vector[CppOwnedHalfSweepBondResult] results
        cdef const CppOwnedHalfSweepBondResult* record
        cdef const CppOwnedHalfSweepSplitSummary* split
        cdef list output = []
        cdef Py_ssize_t item
        cdef Py_ssize_t index
        cdef dict split_payload
        cdef dict solve_payload
        if (
            max_component_elements <= 0
            or max_transform_elements <= 0
            or workspace_budget_bytes <= 0
            or workspace_basis_arrays <= 0
        ):
            raise ValueError("Owned half-sweep workspace limits must be positive.")
        results = self._engine.execute_owned_half_sweep(
            cutoff,
            max_bond_value,
            mode,
            <cpp_bool>retain_sector_topology,
            projection_tolerance,
            <size_t>max_component_elements,
            <size_t>max_transform_elements,
            davidson_tolerance,
            max_iterations,
            requested_restart_dimension,
            <size_t>workspace_budget_bytes,
            <size_t>workspace_basis_arrays,
            <cpp_bool>accept_unconverged,
        )
        for item in range(results.size()):
            record = &results[item]
            split = &record[0].split
            split_payload = {
                "kind": "cpp_su2_owned_half_sweep_split_summary",
                "truncation_error": float(split[0].truncation_error),
                "full_squared_norm": float(split[0].full_squared_norm),
                "kept_squared_norm": float(split[0].kept_squared_norm),
                "kept_states": int(split[0].kept_states),
                "left_revision": (
                    int(split[0].left_topology_revision),
                    int(split[0].left_numeric_revision),
                ),
                "right_revision": (
                    int(split[0].right_topology_revision),
                    int(split[0].right_numeric_revision),
                ),
            }
            solve_payload = {
                "kind": "cpp_su2_owned_half_sweep_bond",
                "compatible": bool(record[0].solve.projection.compatible),
                "accepted": bool(record[0].solve.davidson.accepted),
                "energy": float(record[0].solve.davidson.energy),
                "residual_norm": float(
                    record[0].solve.davidson.residual_norm
                ),
                "iterations": int(record[0].solve.davidson.iterations),
                "basis_size": int(record[0].solve.davidson.basis_size),
                "restarts": int(record[0].solve.davidson.restarts),
                "converged": bool(record[0].solve.davidson.converged),
                "workspace_reused": bool(
                    record[0].solve.davidson.workspace_reused
                ),
                "matvec_calls": int(record[0].solve.davidson.matvec_calls),
                "requested_max_space": int(
                    record[0].solve.requested_restart_dimension
                ),
                "workspace_max_space": int(
                    record[0].solve.workspace_restart_dimension
                ),
                "estimated_basis_workspace_bytes": int(
                    record[0].solve.estimated_workspace_bytes
                ),
                "orthonormal_dimension": int(
                    record[0].solve.projection.orthonormal_dimension
                ),
                "parent_dimension": int(
                    record[0].solve.projection.parent_dimension
                ),
                "projection_reused": bool(
                    record[0].solve.projection.reused
                ),
                "projection_build_seconds": float(
                    record[0].solve.projection.build_seconds
                ),
                "solve_seconds": float(record[0].solve.solve_seconds),
                "state_average": bool(record[0].state_average),
                "state_energies": [
                    float(record[0].state_energies[index])
                    for index in range(
                        record[0].state_energies.size()
                    )
                ],
                "state_residual_norms": [
                    float(record[0].state_residual_norms[index])
                    for index in range(
                        record[0].state_residual_norms.size()
                    )
                ],
            }
            output.append(
                {
                    "bond": int(record[0].bond),
                    "solve": solve_payload,
                    "split": split_payload,
                }
            )
        return output

    def export_owned_split_sites(self):
        """Materialize the final C++-owned MPS once at solver completion."""

        cdef vector[CppOwnedSplitSiteExport] exported
        cdef const CppOwnedSplitSiteExport* record
        cdef const CppPackedSiteTensor* tensor
        cdef Py_ssize_t item
        cdef list output = []
        exported = self._engine.export_owned_split_sites()
        for item in range(exported.size()):
            record = &exported[item]
            tensor = &record[0].tensor
            output.append(
                {
                    "site": int(record[0].site),
                    "tensor": {
                        "values": np.asarray(
                            [
                                tensor[0].values[index]
                                for index in range(tensor[0].values.size())
                            ],
                            dtype=np.float64,
                        ),
                        "offsets": np.asarray(
                            [
                                tensor[0].offsets[index]
                                for index in range(tensor[0].offsets.size())
                            ],
                            dtype=np.int64,
                        ),
                        "labels": np.asarray(
                            [
                                tensor[0].labels[index]
                                for index in range(tensor[0].labels.size())
                            ],
                            dtype=np.int64,
                        ).reshape((-1, 6)),
                        "shape_offsets": np.asarray(
                            [
                                tensor[0].shape_offsets[index]
                                for index in range(
                                    tensor[0].shape_offsets.size()
                                )
                            ],
                            dtype=np.int64,
                        ),
                        "shapes": np.asarray(
                            [
                                tensor[0].shapes[index]
                                for index in range(tensor[0].shapes.size())
                            ],
                            dtype=np.int64,
                        ),
                    },
                    "leg_sector_offsets": np.asarray(
                        [
                            record[0].leg_sector_offsets[index]
                            for index in range(
                                record[0].leg_sector_offsets.size()
                            )
                        ],
                        dtype=np.int64,
                    ),
                    "leg_sector_labels": np.asarray(
                        [
                            record[0].leg_sector_labels[index]
                            for index in range(
                                record[0].leg_sector_labels.size()
                            )
                        ],
                        dtype=np.int64,
                    ).reshape((-1, 2)),
                    "leg_sector_dims": np.asarray(
                        [
                            record[0].leg_sector_dims[index]
                            for index in range(
                                record[0].leg_sector_dims.size()
                            )
                        ],
                        dtype=np.int64,
                    ),
                    "revision": (
                        int(record[0].topology_revision),
                        int(record[0].numeric_revision),
                    ),
                }
            )
        return output

    def execute_half_sweep(self, object callback):
        """Run the active bond transaction loop behind one C++ boundary."""

        cdef dict state
        cdef size_t completed
        if not callable(callback):
            raise TypeError("half-sweep bond executor must be callable.")
        state = {
            "callback": callback,
            "error": None,
            "traceback": None,
        }
        try:
            completed = self._engine.execute_half_sweep(
                _execute_half_sweep_bond,
                <void*><PyObject*>state,
            )
        except Exception:
            if state["error"] is not None:
                raise state["error"].with_traceback(state["traceback"])
            raise
        return int(completed)

    def begin_bond(self, long long bond):
        self._engine.begin_bond(<int64_t>bond)

    def claim_next_bond(self):
        return int(self._engine.claim_next_bond())

    def mark_bond_solved(self):
        self._engine.mark_bond_solved()

    def mark_bond_split(
        self,
        unsigned long long kept_states=0,
        double truncation_seconds=0.0,
    ):
        self._engine.mark_bond_split(
            <uint64_t>kept_states,
            truncation_seconds,
        )

    def mark_bond_advanced(self):
        self._engine.mark_bond_advanced()

    def commit_bond(
        self,
        unsigned long long matvec_calls=0,
        unsigned long long davidson_iterations=0,
        double matvec_seconds=0.0,
        double davidson_seconds=0.0,
        object energy=None,
    ):
        cdef double energy_value = float("nan") if energy is None else float(energy)
        self._engine.commit_bond(
            <uint64_t>matvec_calls,
            <uint64_t>davidson_iterations,
            matvec_seconds,
            davidson_seconds,
            energy_value,
        )

    def commit_bond_update(
        self,
        unsigned long long matvec_calls=0,
        unsigned long long davidson_iterations=0,
        double matvec_seconds=0.0,
        double davidson_seconds=0.0,
        object energy=None,
    ):
        cdef double energy_value = float("nan") if energy is None else float(energy)
        self._engine.commit_bond_update(
            <uint64_t>matvec_calls,
            <uint64_t>davidson_iterations,
            matvec_seconds,
            davidson_seconds,
            energy_value,
        )

    def record_bond(
        self,
        long long bond,
        unsigned long long matvec_calls=0,
        unsigned long long davidson_iterations=0,
        unsigned long long kept_states=0,
        double matvec_seconds=0.0,
        double davidson_seconds=0.0,
        double truncation_seconds=0.0,
    ):
        self._engine.record_bond(
            <int64_t>bond,
            <uint64_t>matvec_calls,
            <uint64_t>davidson_iterations,
            <uint64_t>kept_states,
            matvec_seconds,
            davidson_seconds,
            truncation_seconds,
        )

    def finish_half_sweep(self):
        self._engine.finish_half_sweep()

    def release_workspaces(self):
        self._engine.release_workspaces()
        self._local_operator_owner = None

    def abort_half_sweep(self):
        self._engine.abort_half_sweep()
        self._local_operator_owner = None
        self._factor_route_owners = None
        self._factor_route_projection_owners = None
        self._factorized_metric_owners = None
        self._factor_route_key = None

    @property
    def stats(self):
        return {
            "kind": "su2_moving_environment",
            "backend": "cpp",
            "system_revision": int(self._engine.system_revision()),
            "boundary_count": int(self._engine.boundary_count()),
            "borrowed_boundary_bytes": int(
                self._engine.borrowed_boundary_bytes()
            ),
            "owned_boundary_bytes": int(
                self._engine.owned_boundary_bytes()
            ),
            "local_operator_blocks": int(
                self._engine.local_operator_blocks()
            ),
            "borrowed_local_operator_bytes": int(
                self._engine.borrowed_local_operator_bytes()
            ),
            "factor_route_count": int(self._engine.factor_route_count()),
            "factor_route_table_bytes": int(
                self._engine.factor_route_table_bytes()
            ),
            "borrowed_factor_pool_bytes": int(
                self._engine.borrowed_factor_pool_bytes()
            ),
            "factor_route_scratch_bytes": int(
                self._engine.factor_route_scratch_bytes()
            ),
            "contextual_route_plan_builds": int(
                self._engine.contextual_route_plan_builds()
            ),
            "contextual_route_plan_hits": int(
                self._engine.contextual_route_plan_hits()
            ),
            "contextual_route_plan_shape_refreshes": int(
                self._engine.contextual_route_plan_shape_refreshes()
            ),
            "decomposed_action_plan_builds": int(
                self._engine.decomposed_action_plan_builds()
            ),
            "decomposed_action_plan_hits": int(
                self._engine.decomposed_action_plan_hits()
            ),
            "decomposed_action_plan_rebuilds": int(
                self._engine.decomposed_action_plan_rebuilds()
            ),
            "complementary_execution_graph_bytes": int(
                self._engine.complementary_execution_graph_bytes()
            ),
            "complementary_execution_graph_builds": int(
                self._engine.complementary_execution_graph_builds()
            ),
            "complementary_execution_graph_hits": int(
                self._engine.complementary_execution_graph_hits()
            ),
            "contextual_route_plan_count": int(
                self._engine.contextual_route_plan_count()
            ),
            "contextual_route_plan_bytes": int(
                self._engine.contextual_route_plan_bytes()
            ),
            "contextual_route_index_bytes": int(
                self._engine.contextual_route_index_bytes()
            ),
            "contextual_route_core_value_bytes": int(
                self._engine.contextual_route_core_value_bytes()
            ),
            "contextual_compiled_schedule_bytes": int(
                self._engine.contextual_compiled_schedule_bytes()
            ),
            "contextual_compiled_schedule_builds": int(
                self._engine.contextual_compiled_schedule_builds()
            ),
            "contextual_compiled_schedule_hits": int(
                self._engine.contextual_compiled_schedule_hits()
            ),
            "contextual_compiled_schedule_restore_seconds": float(
                self._engine.contextual_compiled_schedule_restore_seconds()
            ),
            "contextual_route_core_elements": int(
                self._engine.contextual_route_core_elements()
            ),
            "contextual_route_core_nonzero_elements": int(
                self._engine.contextual_route_core_nonzero_elements()
            ),
            "contextual_core_cache_count": int(
                self._engine.contextual_core_cache_count()
            ),
            "contextual_core_cache_bytes": int(
                self._engine.contextual_core_cache_bytes()
            ),
            "contextual_core_cache_hits": int(
                self._engine.contextual_core_cache_hits()
            ),
            "contextual_core_reuse_hits": int(
                self._engine.contextual_core_reuse_hits()
            ),
            "contextual_route_match_seconds": float(
                self._engine.contextual_route_match_seconds()
            ),
            "contextual_route_activation_seconds": float(
                self._engine.contextual_route_activation_seconds()
            ),
            "contextual_core_build_seconds": float(
                self._engine.contextual_core_build_seconds()
            ),
            "contextual_core_reuse_seconds": float(
                self._engine.contextual_core_reuse_seconds()
            ),
            "raw_route_setup_seconds": float(
                self._engine.raw_route_setup_seconds()
            ),
            "raw_route_group_seconds": float(
                self._engine.raw_route_group_seconds()
            ),
            "dense_pair_build_seconds": float(
                self._engine.dense_pair_build_seconds()
            ),
            "fused_factor_build_seconds": float(
                self._engine.fused_factor_build_seconds()
            ),
            "raw_execution_build_seconds": float(
                self._engine.raw_execution_build_seconds()
            ),
            "raw_factor_matvec_seconds": float(
                self._engine.raw_factor_matvec_seconds()
            ),
            "raw_input_pack_seconds": float(
                self._engine.raw_input_pack_seconds()
            ),
            "dense_pair_matvec_seconds": float(
                self._engine.dense_pair_matvec_seconds()
            ),
            "raw_execution_matvec_seconds": float(
                self._engine.raw_execution_matvec_seconds()
            ),
            "raw_execution_pack_seconds": float(
                self._engine.raw_execution_pack_seconds()
            ),
            "raw_pointer_execution_matvec_seconds": float(
                self._engine.raw_pointer_execution_matvec_seconds()
            ),
            "raw_pointer_execution_matvec_calls": int(
                self._engine.raw_pointer_execution_matvec_calls()
            ),
            "direct_complementary_action_seconds": float(
                self._engine.direct_complementary_action_seconds()
            ),
            "direct_complementary_action_calls": int(
                self._engine.direct_complementary_action_calls()
            ),
            "direct_complementary_actions": int(
                self._engine.direct_complementary_actions()
            ),
            "factorized_metric_matvec_seconds": float(
                self._engine.factorized_metric_matvec_seconds()
            ),
            "contextual_zero_core_cache_hits": int(
                self._engine.contextual_zero_core_cache_hits()
            ),
            "contextual_zero_core_cache_count": int(
                self._engine.contextual_zero_core_cache_count()
            ),
            "contextual_zero_core_cache_bytes": int(
                self._engine.contextual_zero_core_cache_bytes()
            ),
            "raw_factor_routes": bool(self._engine.raw_factor_routes()),
            "factor_routes_hermitianized": bool(
                self._engine.factor_routes_hermitianized()
            ),
            "borrowed_raw_factor_source_bytes": int(
                self._engine.borrowed_raw_factor_source_bytes()
            ),
            "raw_factor_cache_bytes": int(
                self._engine.raw_factor_cache_bytes()
            ),
            "peak_raw_factor_cache_bytes": int(
                self._engine.peak_raw_factor_cache_bytes()
            ),
            "factor_route_projection_components": int(
                self._engine.factor_route_projection_components()
            ),
            "factor_route_projection_index_bytes": int(
                self._engine.factor_route_projection_index_bytes()
            ),
            "borrowed_factor_route_transform_bytes": int(
                self._engine.borrowed_factor_route_transform_bytes()
            ),
            "factor_route_projection_scratch_bytes": int(
                self._engine.factor_route_projection_scratch_bytes()
            ),
            "factorized_metric_route_bytes": int(
                self._engine.factorized_metric_route_bytes()
            ),
            "factorized_metric_scratch_bytes": int(
                self._engine.factorized_metric_scratch_bytes()
            ),
            "davidson_workspace_bytes": int(
                self._engine.davidson_workspace_bytes()
            ),
            "state_average_roots": int(
                self._engine.state_average_roots()
            ),
            "state_average_center_site": int(
                self._engine.state_average_center_site()
            ),
            "peak_borrowed_local_operator_bytes": int(
                self._engine.peak_borrowed_local_operator_bytes()
            ),
            "peak_factor_route_table_bytes": int(
                self._engine.peak_factor_route_table_bytes()
            ),
            "peak_borrowed_factor_pool_bytes": int(
                self._engine.peak_borrowed_factor_pool_bytes()
            ),
            "peak_factor_route_scratch_bytes": int(
                self._engine.peak_factor_route_scratch_bytes()
            ),
            "peak_factor_route_projection_index_bytes": int(
                self._engine.peak_factor_route_projection_index_bytes()
            ),
            "peak_borrowed_factor_route_transform_bytes": int(
                self._engine.peak_borrowed_factor_route_transform_bytes()
            ),
            "peak_factor_route_projection_scratch_bytes": int(
                self._engine.peak_factor_route_projection_scratch_bytes()
            ),
            "local_topology_builds": int(
                self._engine.local_topology_builds()
            ),
            "local_numeric_refreshes": int(
                self._engine.local_numeric_refreshes()
            ),
            "local_matvec_calls": int(self._engine.local_matvec_calls()),
            "local_davidson_calls": int(
                self._engine.local_davidson_calls()
            ),
            "local_davidson_workspace_reuses": int(
                self._engine.local_davidson_workspace_reuses()
            ),
            "factor_route_topology_builds": int(
                self._engine.factor_route_topology_builds()
            ),
            "factor_route_numeric_refreshes": int(
                self._engine.factor_route_numeric_refreshes()
            ),
            "factor_route_matvec_calls": int(
                self._engine.factor_route_matvec_calls()
            ),
            "real_factor_route_matvec_calls": int(
                self._engine.real_factor_route_matvec_calls()
            ),
            "factor_route_diagonal_calls": int(
                self._engine.factor_route_diagonal_calls()
            ),
            "factor_route_davidson_calls": int(
                self._engine.factor_route_davidson_calls()
            ),
            "factor_route_scratch_growths": int(
                self._engine.factor_route_scratch_growths()
            ),
            "raw_factor_cache_hits": int(
                self._engine.raw_factor_cache_hits()
            ),
            "raw_factor_cache_misses": int(
                self._engine.raw_factor_cache_misses()
            ),
            "raw_factor_gemm_calls": int(
                self._engine.raw_factor_gemm_calls()
            ),
            "raw_output_product_calls": int(
                self._engine.raw_output_product_calls()
            ),
            "direct_source_factor_loads": int(
                self._engine.direct_source_factor_loads()
            ),
            "compact_right_panel_count": int(
                self._engine.compact_right_panel_count()
            ),
            "compact_right_panel_value_bytes": int(
                self._engine.compact_right_panel_value_bytes()
            ),
            "compact_right_panel_product_count": int(
                self._engine.compact_right_panel_product_count()
            ),
            "compact_right_panel_batch_count": int(
                self._engine.compact_right_panel_batch_count()
            ),
            "compact_right_panel_budget_bytes": int(
                self._engine.compact_right_panel_budget_bytes()
            ),
            "compact_right_panel_registry_builds": int(
                self._engine.compact_right_panel_registry_builds()
            ),
            "compact_right_panel_numeric_refreshes": int(
                self._engine.compact_right_panel_numeric_refreshes()
            ),
            "compact_right_panel_matvec_batches": int(
                self._engine.compact_right_panel_matvec_batches()
            ),
            "compact_right_panel_matvec_products": int(
                self._engine.compact_right_panel_matvec_products()
            ),
            "complementary_family_route_counts": {
                "S": int(self._engine.complementary_family_route_count(0)),
                "R": int(self._engine.complementary_family_route_count(1)),
                "A": int(self._engine.complementary_family_route_count(2)),
                "P": int(self._engine.complementary_family_route_count(3)),
                "B": int(self._engine.complementary_family_route_count(4)),
                "Q": int(self._engine.complementary_family_route_count(5)),
                "unlabeled": int(
                    self._engine.unlabeled_family_route_count()
                ),
            },
            "raw_factor_build_seconds": float(
                self._engine.raw_factor_build_seconds()
            ),
            "raw_route_group_count": int(
                self._engine.raw_route_group_count()
            ),
            "complementary_local_actions": bool(
                self._engine.complementary_local_actions()
            ),
            "complementary_local_action_count": int(
                self._engine.complementary_local_action_count()
            ),
            "complementary_local_term_count": int(
                self._engine.complementary_local_term_count()
            ),
            "complementary_local_action_bytes": int(
                self._engine.complementary_local_action_bytes()
            ),
            "fused_raw_route_group_count": int(
                self._engine.fused_raw_route_group_count()
            ),
            "fused_raw_route_count": int(
                self._engine.fused_raw_route_count()
            ),
            "dense_pair_kernel_count": int(
                self._engine.dense_pair_kernel_count()
            ),
            "dense_pair_execution_count": int(
                self._engine.dense_pair_execution_count()
            ),
            "dense_pair_kernel_elements": int(
                self._engine.dense_pair_kernel_elements()
            ),
            "dense_pair_route_count": int(
                self._engine.dense_pair_route_count()
            ),
            "resident_family_kernel_count": int(
                self._engine.dense_pair_kernel_count()
            ),
            "resident_family_kernel_bytes": int(
                self._engine.dense_pair_kernel_elements()
            ) * 8,
            "resident_family_route_count": int(
                self._engine.dense_pair_route_count()
            ),
            "resident_family_kernel_budget_bytes": 32_000_000,
            "resident_family_factor_pack_bytes": int(
                self._engine.dense_factor_pack_bytes()
            ),
            "resident_family_factor_pack_budget_bytes": 4_000_000,
            "resident_family_factor_pack_builds": int(
                self._engine.dense_factor_pack_builds()
            ),
            "resident_family_factor_pack_reuses": int(
                self._engine.dense_factor_pack_reuses()
            ),
            "raw_execution_group_count": int(
                self._engine.raw_execution_group_count()
            ),
            "raw_execution_action_count": int(
                self._engine.raw_execution_action_count()
            ),
            "right_grouped_execution_action_count": int(
                self._engine.right_grouped_execution_action_count()
            ),
            "peak_right_grouped_execution_action_count": int(
                self._engine.peak_right_grouped_execution_action_count()
            ),
            "peak_raw_execution_group_count": int(
                self._engine.peak_raw_execution_group_count()
            ),
            "peak_raw_execution_action_count": int(
                self._engine.peak_raw_execution_action_count()
            ),
            "raw_input_superchannel_count": int(
                self._engine.raw_input_superchannel_count()
            ),
            "raw_input_superchannel_tile_count": int(
                self._engine.raw_input_superchannel_tile_count()
            ),
            "raw_input_superchannel_batch_count": int(
                self._engine.raw_input_superchannel_batch_count()
            ),
            "peak_raw_input_superchannel_batch_count": int(
                self._engine.peak_raw_input_superchannel_batch_count()
            ),
            "peak_raw_channel_unique_left_count": int(
                self._engine.peak_raw_channel_unique_left_count()
            ),
            "peak_raw_channel_left_occurrence_count": int(
                self._engine.peak_raw_channel_left_occurrence_count()
            ),
            "peak_raw_shared_left_panel_count": int(
                self._engine.peak_raw_shared_left_panel_count()
            ),
            "peak_raw_shared_left_occurrence_count": int(
                self._engine.peak_raw_shared_left_occurrence_count()
            ),
            "peak_raw_output_fusion_wave_count": int(
                self._engine.peak_raw_output_fusion_wave_count()
            ),
            "peak_raw_output_fusion_group_count": int(
                self._engine.peak_raw_output_fusion_group_count()
            ),
            "peak_raw_output_fusion_tile_count": int(
                self._engine.peak_raw_output_fusion_tile_count()
            ),
            "peak_raw_output_fusion_workspace_bytes": int(
                self._engine.peak_raw_output_fusion_workspace_bytes()
            ),
            "raw_output_fusion_gemm_calls": int(
                self._engine.raw_output_fusion_gemm_calls()
            ),
            "raw_output_fusion_copied_elements": int(
                self._engine.raw_output_fusion_copied_elements()
            ),
            "grouped_output_product_backend": bool(
                self._engine.grouped_output_product_backend()
            ),
            "grouped_output_product_group_count": int(
                self._engine.grouped_output_product_group_count()
            ),
            "grouped_output_product_binding_count": int(
                self._engine.grouped_output_product_binding_count()
            ),
            "peak_grouped_output_candidate_binding_count": int(
                self._engine.peak_grouped_output_candidate_binding_count()
            ),
            "peak_grouped_output_candidate_work_counts": [
                int(
                    self._engine.peak_grouped_output_candidate_work_count(
                        bin_index
                    )
                )
                for bin_index in range(10)
            ],
            "grouped_output_product_batch_calls": int(
                self._engine.grouped_output_product_batch_calls()
            ),
            "grouped_output_products": int(
                self._engine.grouped_output_products()
            ),
            "peak_raw_shared_right_panel_count": int(
                self._engine.peak_raw_shared_right_panel_count()
            ),
            "peak_raw_shared_right_binding_count": int(
                self._engine.peak_raw_shared_right_binding_count()
            ),
            "peak_raw_shared_right_workspace_bytes": int(
                self._engine.peak_raw_shared_right_workspace_bytes()
            ),
            "raw_shared_right_gemm_calls": int(
                self._engine.raw_shared_right_gemm_calls()
            ),
            "raw_shared_right_copied_elements": int(
                self._engine.raw_shared_right_copied_elements()
            ),
            "reduced_contextual_routes": bool(
                self._engine.reduced_contextual_routes()
            ),
            "reduced_contextual_execution_count": int(
                self._engine.reduced_contextual_execution_count()
            ),
            "reduced_contextual_matrix_elements": int(
                self._engine.reduced_contextual_matrix_elements()
            ),
            "peak_reduced_contextual_execution_count": int(
                self._engine.peak_reduced_contextual_execution_count()
            ),
            "peak_reduced_contextual_matrix_elements": int(
                self._engine.peak_reduced_contextual_matrix_elements()
            ),
            "peak_borrowed_reduced_contextual_right_elements": int(
                self._engine.peak_borrowed_reduced_contextual_right_elements()
            ),
            "complementary_execution_slab_bytes": int(
                self._engine.complementary_execution_slab_bytes()
            ),
            "complementary_execution_slab_capacity_bytes": int(
                self._engine.complementary_execution_slab_capacity_bytes()
            ),
            "complementary_execution_slab_budget_bytes": int(
                self._engine.complementary_execution_slab_budget_bytes()
            ),
            "complementary_execution_slab_required_bytes": int(
                self._engine.complementary_execution_slab_required_bytes()
            ),
            "peak_complementary_execution_slab_required_bytes": int(
                self._engine.peak_complementary_execution_slab_required_bytes()
            ),
            "peak_complementary_left_required_bytes": int(
                self._engine.peak_complementary_left_required_bytes()
            ),
            "peak_complementary_right_required_bytes": int(
                self._engine.peak_complementary_right_required_bytes()
            ),
            "peak_complementary_left_cached_bytes": int(
                self._engine.peak_complementary_left_cached_bytes()
            ),
            "peak_complementary_right_cached_bytes": int(
                self._engine.peak_complementary_right_cached_bytes()
            ),
            "complementary_execution_slab_full_prepares": int(
                self._engine.complementary_execution_slab_full_prepares()
            ),
            "complementary_execution_slab_partial_prepares": int(
                self._engine.complementary_execution_slab_partial_prepares()
            ),
            "complementary_execution_slab_matvec_repacks": int(
                self._engine.complementary_execution_slab_matvec_repacks()
            ),
            "peak_reduced_contextual_boundary_rank": int(
                self._engine.peak_reduced_contextual_boundary_rank()
            ),
            "reduced_contextual_fallbacks": int(
                self._engine.reduced_contextual_fallbacks()
            ),
            "reduced_contextual_fallback_reason": int(
                self._engine.reduced_contextual_fallback_reason()
            ),
            "reduced_contextual_fallback_residual_norm": float(
                self._engine.reduced_contextual_fallback_residual_norm()
            ),
            "reduced_contextual_fallback_boundary_norm": float(
                self._engine.reduced_contextual_fallback_boundary_norm()
            ),
            "reduced_contextual_build_seconds": float(
                self._engine.reduced_contextual_build_seconds()
            ),
            "reduced_contextual_numeric_refresh_seconds": float(
                self._engine.reduced_contextual_numeric_refresh_seconds()
            ),
            "reduced_contextual_execution_refresh_seconds": float(
                self._engine.reduced_contextual_execution_refresh_seconds()
            ),
            "reduced_contextual_diagonal_seconds": float(
                self._engine.reduced_contextual_diagonal_seconds()
            ),
            "reduced_contextual_matvec_seconds": float(
                self._engine.reduced_contextual_matvec_seconds()
            ),
            "factor_route_projection_topology_builds": int(
                self._engine.factor_route_projection_topology_builds()
            ),
            "factor_route_projection_numeric_refreshes": int(
                self._engine.factor_route_projection_numeric_refreshes()
            ),
            "factor_route_projected_matvec_calls": int(
                self._engine.factor_route_projected_matvec_calls()
            ),
            "factor_route_projected_davidson_calls": int(
                self._engine.factor_route_projected_davidson_calls()
            ),
            "factor_route_generalized_davidson_calls": int(
                self._engine.factor_route_generalized_davidson_calls()
            ),
            "real_generalized_davidson_calls": int(
                self._engine.real_generalized_davidson_calls()
            ),
            "factorized_metric_matvec_calls": int(
                self._engine.factorized_metric_matvec_calls()
            ),
            "real_factorized_metric_matvec_calls": int(
                self._engine.real_factorized_metric_matvec_calls()
            ),
            "canonical_projection_builds": int(
                self._engine.canonical_projection_builds()
            ),
            "canonical_projection_reuses": int(
                self._engine.canonical_projection_reuses()
            ),
            "canonical_projection_davidson_calls": int(
                self._engine.canonical_projection_davidson_calls()
            ),
            "canonical_projection_transform_elements": int(
                self._engine.canonical_projection_transform_elements()
            ),
            "canonical_projection_max_component_dimension": int(
                self._engine.canonical_projection_max_component_dimension()
            ),
            "canonical_projection_cache_entries": int(
                self._engine.canonical_projection_cache_entries()
            ),
            "canonical_projection_cache_transform_elements": int(
                self._engine
                .canonical_projection_cache_transform_elements()
            ),
            "canonical_projection_cache_evictions": int(
                self._engine.canonical_projection_cache_evictions()
            ),
            "canonical_projection_whitening_residual": float(
                self._engine.canonical_projection_whitening_residual()
            ),
            "canonical_projection_build_seconds": float(
                self._engine.canonical_projection_build_seconds()
            ),
            "block_svd_calls": int(self._engine.block_svd_calls()),
            "block_svd_blocks": int(self._engine.block_svd_blocks()),
            "block_svd_workspace_growths": int(
                self._engine.block_svd_workspace_growths()
            ),
            "block_svd_seconds": float(self._engine.block_svd_seconds()),
            "block_svd_workspace_bytes": int(
                self._engine.block_svd_workspace_bytes()
            ),
            "split_site_owner_revision": int(
                self._engine.split_site_owner_revision()
            ),
            "split_site_installs": int(
                self._engine.split_site_installs()
            ),
            "split_site_topology_builds": int(
                self._engine.split_site_topology_builds()
            ),
            "split_site_boundary_uses": int(
                self._engine.split_site_boundary_uses()
            ),
            "cached_boundary_replays": int(
                self._engine.cached_boundary_replays()
            ),
            "split_site_count": int(
                self._engine.split_site_count()
            ),
            "split_site_bytes": int(
                self._engine.split_site_bytes()
            ),
            "site_merge_calls": int(
                self._engine.site_merge_calls()
            ),
            "site_merge_blocks": int(
                self._engine.site_merge_blocks()
            ),
            "site_merge_seconds": float(
                self._engine.site_merge_seconds()
            ),
            "site_merge_bytes": int(
                self._engine.site_merge_bytes()
            ),
            "active_bond_complementary_prepares": int(
                self._engine.active_bond_complementary_prepares()
            ),
            "active_bond_complementary_fallbacks": int(
                self._engine.active_bond_complementary_fallbacks()
            ),
            "active_bond_complementary_fallback_reason": int(
                self._engine.active_bond_complementary_fallback_reason()
            ),
            "active_bond_complementary_fallback_bond": int(
                self._engine.active_bond_complementary_fallback_bond()
            ),
            "active_bond_complementary_basis": int(
                self._engine.active_bond_complementary_basis()
            ),
            "active_bond_complementary_dimension": int(
                self._engine.active_bond_complementary_dimension()
            ),
            "active_bond_complementary_expected_basis": int(
                self._engine.active_bond_complementary_expected_basis()
            ),
            "active_bond_complementary_expected_dimension": int(
                self._engine.active_bond_complementary_expected_dimension()
            ),
            "active_bond_complementary_davidson_calls": int(
                self._engine.active_bond_complementary_davidson_calls()
            ),
            "active_bond_complementary_generalized_davidson_calls": int(
                self._engine.active_bond_complementary_generalized_davidson_calls()
            ),
            "active_bond_metric_prepares": int(
                self._engine.active_bond_metric_prepares()
            ),
            "active_bond_cpp_splits": int(
                self._engine.active_bond_cpp_splits()
            ),
            "boundary_topology_builds": int(
                self._engine.boundary_topology_builds()
            ),
            "boundary_numeric_refreshes": int(
                self._engine.boundary_numeric_refreshes()
            ),
            "boundary_reallocations": int(
                self._engine.boundary_reallocations()
            ),
            "boundary_update_topology_builds": int(
                self._engine.boundary_update_topology_builds()
            ),
            "boundary_update_calls": int(
                self._engine.boundary_update_calls()
            ),
            "boundary_update_routes": int(
                self._engine.boundary_update_routes()
            ),
            "boundary_update_seconds": float(
                self._engine.boundary_update_seconds()
            ),
            "normal_complementary_boundary_action_count": int(
                self._engine.normal_complementary_boundary_action_count()
            ),
            "normal_complementary_boundary_action_bytes": int(
                self._engine.normal_complementary_boundary_action_bytes()
            ),
            "metric_boundary_count": int(
                self._engine.metric_boundary_count()
            ),
            "metric_boundary_action_count": int(
                self._engine.metric_boundary_action_count()
            ),
            "metric_boundary_action_bytes": int(
                self._engine.metric_boundary_action_bytes()
            ),
            "half_sweeps": int(self._engine.half_sweeps()),
            "half_sweep_executor_calls": int(
                self._engine.half_sweep_executor_calls()
            ),
            "half_sweep_executor_bonds": int(
                self._engine.half_sweep_executor_bonds()
            ),
            "half_sweep_python_bond_callbacks": int(
                self._engine.half_sweep_python_bond_callbacks()
            ),
            "half_sweep_executor_seconds": float(
                self._engine.half_sweep_executor_seconds()
            ),
            "owned_half_sweep_calls": int(
                self._engine.owned_half_sweep_calls()
            ),
            "owned_half_sweep_bonds": int(
                self._engine.owned_half_sweep_bonds()
            ),
            "owned_half_sweep_seconds": float(
                self._engine.owned_half_sweep_seconds()
            ),
            "aborted_half_sweeps": int(
                self._engine.aborted_half_sweeps()
            ),
            "bond_steps": int(self._engine.bond_steps()),
            "bond_prepares": int(self._engine.bond_prepares()),
            "bond_solves": int(self._engine.bond_solves()),
            "bond_splits": int(self._engine.bond_splits()),
            "bond_advances": int(self._engine.bond_advances()),
            "staged_bond_updates": int(
                self._engine.staged_bond_updates()
            ),
            "committed_bond_updates": int(
                self._engine.committed_bond_updates()
            ),
            "lifecycle_phase": (
                "idle",
                "ready",
                "prepared",
                "solved",
                "split",
                "advanced",
            )[self._engine.lifecycle_phase_code()],
            "active_bond": int(self._engine.active_bond()),
            "matvec_calls": int(self._engine.matvec_calls()),
            "davidson_iterations": int(self._engine.davidson_iterations()),
            "kept_states": int(self._engine.kept_states()),
            "matvec_seconds": float(self._engine.matvec_seconds()),
            "davidson_seconds": float(self._engine.davidson_seconds()),
            "truncation_seconds": float(self._engine.truncation_seconds()),
            "last_half_sweep_energy": float(
                self._engine.last_half_sweep_energy()
            ),
            "memory_bytes": int(self._engine.memory_bytes()),
        }


def real_array_nonzero_mask(object arrays, double tol=0.0):
    """Return a C++ nonzero mask for a sequence of float64 arrays."""

    cdef list packed = []
    cdef object value
    cdef cnp.ndarray arr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mask
    cdef Py_ssize_t array_index
    cdef Py_ssize_t value_index
    cdef Py_ssize_t size
    cdef double *data
    cdef double threshold = max(0.0, tol)

    for value in arrays:
        arr = np.asarray(value)
        if arr.dtype != np.float64 or not arr.flags.c_contiguous:
            return None
        packed.append(arr)
    mask = np.zeros(len(packed), dtype=np.uint8)
    for array_index in range(len(packed)):
        arr = packed[array_index]
        size = arr.size
        data = <double *>cnp.PyArray_DATA(arr)
        if threshold == 0.0:
            for value_index in range(size):
                if data[value_index] != 0.0:
                    mask[array_index] = 1
                    break
        else:
            for value_index in range(size):
                if fabs(data[value_index]) > threshold:
                    mask[array_index] = 1
                    break
    return mask


def build_entry_family_masks(object offsets, object family_ids, object family_bits):
    """Build one uint64 family mask per packed factor-table entry."""

    cdef cnp.ndarray[cnp.int64_t, ndim=1] offset_arr = np.ascontiguousarray(
        offsets,
        dtype=np.int64,
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] family_arr = np.ascontiguousarray(
        family_ids,
        dtype=np.int64,
    )
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] bit_arr = np.ascontiguousarray(
        family_bits,
        dtype=np.uint64,
    )
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] masks
    cdef Py_ssize_t n_entries
    cdef Py_ssize_t entry
    cdef Py_ssize_t pos
    cdef Py_ssize_t start
    cdef Py_ssize_t stop
    cdef Py_ssize_t family
    cdef cnp.uint64_t mask

    if offset_arr.shape[0] == 0:
        return np.zeros(0, dtype=np.uint64)
    n_entries = offset_arr.shape[0] - 1
    masks = np.zeros(n_entries, dtype=np.uint64)
    for entry in range(n_entries):
        start = offset_arr[entry]
        stop = offset_arr[entry + 1]
        if start < 0 or stop < start or stop > family_arr.shape[0]:
            return None
        mask = 0
        for pos in range(start, stop):
            family = family_arr[pos]
            if family < 0 or family >= bit_arr.shape[0]:
                return None
            mask |= bit_arr[family]
        masks[entry] = mask
    return masks


def pack_real_array_pool(object arrays):
    """Pack variable-rank real arrays into one contiguous C++ arena."""

    cdef list packed = []
    cdef object value
    cdef cnp.ndarray arr
    cdef Py_ssize_t n_arrays
    cdef Py_ssize_t total_size = 0
    cdef Py_ssize_t total_ndim = 0
    cdef Py_ssize_t index
    cdef Py_ssize_t dim
    cdef Py_ssize_t data_cursor = 0
    cdef Py_ssize_t shape_cursor = 0
    cdef Py_ssize_t array_size
    cdef cnp.ndarray[cnp.double_t, ndim=1] data
    cdef cnp.ndarray[cnp.int64_t, ndim=1] offsets
    cdef cnp.ndarray[cnp.int64_t, ndim=1] shape_offsets
    cdef cnp.ndarray[cnp.int64_t, ndim=1] shapes

    for value in arrays:
        if np.iscomplexobj(value):
            return None
        arr = np.ascontiguousarray(value, dtype=np.float64)
        packed.append(arr)
        total_size += arr.size
        total_ndim += arr.ndim
    n_arrays = len(packed)
    data = np.empty(total_size, dtype=np.float64)
    offsets = np.empty(n_arrays + 1, dtype=np.int64)
    shape_offsets = np.empty(n_arrays + 1, dtype=np.int64)
    shapes = np.empty(total_ndim, dtype=np.int64)
    offsets[0] = 0
    shape_offsets[0] = 0
    for index in range(n_arrays):
        arr = packed[index]
        array_size = arr.size
        if array_size:
            memcpy(
                <char *>cnp.PyArray_DATA(data) + data_cursor * sizeof(double),
                cnp.PyArray_DATA(arr),
                array_size * sizeof(double),
            )
        data_cursor += array_size
        offsets[index + 1] = data_cursor
        for dim in range(arr.ndim):
            shapes[shape_cursor] = arr.shape[dim]
            shape_cursor += 1
        shape_offsets[index + 1] = shape_cursor
    return data, offsets, shape_offsets, shapes


def factorize_rank_coupled_left(object E, object W):
    """
    Build a left qchem factor block ``L[l,k,w,a,b]`` from ``E[x,l,k]``
    and ``W[x,w,a,b]``.
    """

    if np.iscomplexobj(E) or np.iscomplexobj(W):
        return _factorize_rank_coupled_left_complex(E, W)
    return _factorize_rank_coupled_left_real(E, W)


def factorize_rank_coupled_left_real(object E, object W):
    """Real-valued fast path for ``factorize_rank_coupled_left``."""

    return _factorize_rank_coupled_left_real(E, W)


def factorize_rank_coupled_right(object W, object F):
    """
    Build a right qchem factor block ``R[w,q,r,d,c]`` from ``W[w,y,d,c]``
    and ``F[y,q,r]``.
    """

    if np.iscomplexobj(W) or np.iscomplexobj(F):
        return _factorize_rank_coupled_right_complex(W, F)
    return _factorize_rank_coupled_right_real(W, F)


def factorize_rank_coupled_right_real(object W, object F):
    """Real-valued fast path for ``factorize_rank_coupled_right``."""

    return _factorize_rank_coupled_right_real(W, F)


def factorize_rank_coupled_real_pairs(
    object boundary_blocks,
    object w_blocks,
    object boundary_ids,
    object w_ids,
    bint left_representation,
):
    """Factorize a packed route's unique real boundary/W pairs in one call."""

    cdef cnp.ndarray[cnp.int64_t, ndim=1] boundary_index = np.ascontiguousarray(
        boundary_ids,
        dtype=np.int64,
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] w_index = np.ascontiguousarray(
        w_ids,
        dtype=np.int64,
    )
    cdef Py_ssize_t index
    cdef object boundary
    cdef object w_block
    cdef list out = []

    if boundary_index.shape[0] != w_index.shape[0]:
        return None
    for boundary in boundary_blocks:
        if np.iscomplexobj(boundary):
            return None
    for w_block in w_blocks:
        if np.iscomplexobj(w_block):
            return None
    for index in range(boundary_index.shape[0]):
        boundary = boundary_blocks[boundary_index[index]]
        w_block = w_blocks[w_index[index]]
        if left_representation:
            out.append(_factorize_rank_coupled_left_real(boundary, w_block))
        else:
            out.append(_factorize_rank_coupled_right_real(w_block, boundary))
    return out


def factorize_rank_coupled_real_pairs_packed(
    object boundary_blocks,
    object w_blocks,
    object boundary_ids,
    object w_ids,
    bint left_representation,
):
    """
    Factorize unique real boundary/W pairs directly into one packed arena.

    Unlike ``factorize_rank_coupled_real_pairs``, this never retains a Python
    list containing every dense factor alongside the final packed copy.
    """

    cdef cnp.ndarray[cnp.int64_t, ndim=1] boundary_index = np.ascontiguousarray(
        boundary_ids,
        dtype=np.int64,
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] w_index = np.ascontiguousarray(
        w_ids,
        dtype=np.int64,
    )
    cdef cnp.ndarray[cnp.double_t, ndim=1] data
    cdef cnp.ndarray[cnp.int64_t, ndim=1] offsets
    cdef cnp.ndarray[cnp.int64_t, ndim=1] shape_offsets
    cdef cnp.ndarray[cnp.int64_t, ndim=1] shapes
    cdef cnp.ndarray boundary_arr
    cdef cnp.ndarray w_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] boundary3
    cdef cnp.ndarray[cnp.double_t, ndim=4] w4
    cdef Py_ssize_t n_pairs = boundary_index.shape[0]
    cdef Py_ssize_t index
    cdef Py_ssize_t dim
    cdef Py_ssize_t size
    cdef Py_ssize_t total_size = 0
    cdef Py_ssize_t out_offset
    cdef Py_ssize_t x_dim
    cdef Py_ssize_t l_dim
    cdef Py_ssize_t k_dim
    cdef Py_ssize_t w_dim
    cdef Py_ssize_t a_dim
    cdef Py_ssize_t b_dim
    cdef Py_ssize_t y_dim
    cdef Py_ssize_t q_dim
    cdef Py_ssize_t r_dim
    cdef Py_ssize_t d_dim
    cdef Py_ssize_t c_dim
    cdef Py_ssize_t x
    cdef Py_ssize_t l
    cdef Py_ssize_t k
    cdef Py_ssize_t w
    cdef Py_ssize_t a
    cdef Py_ssize_t b
    cdef Py_ssize_t y
    cdef Py_ssize_t q
    cdef Py_ssize_t r
    cdef Py_ssize_t d
    cdef Py_ssize_t c
    cdef double boundary_value
    cdef double w_value
    cdef tuple shape

    if n_pairs != w_index.shape[0]:
        return None
    offsets = np.empty(n_pairs + 1, dtype=np.int64)
    shape_offsets = np.arange(0, 5 * (n_pairs + 1), 5, dtype=np.int64)
    shapes = np.empty(5 * n_pairs, dtype=np.int64)
    offsets[0] = 0
    for index in range(n_pairs):
        boundary_arr = np.asarray(boundary_blocks[boundary_index[index]])
        w_arr = np.asarray(w_blocks[w_index[index]])
        if (
            np.iscomplexobj(boundary_arr)
            or np.iscomplexobj(w_arr)
            or boundary_arr.ndim != 3
            or w_arr.ndim != 4
        ):
            return None
        if left_representation:
            if boundary_arr.shape[0] != w_arr.shape[0]:
                return None
            shape = (
                boundary_arr.shape[1],
                boundary_arr.shape[2],
                w_arr.shape[1],
                w_arr.shape[2],
                w_arr.shape[3],
            )
        else:
            if w_arr.shape[1] != boundary_arr.shape[0]:
                return None
            shape = (
                w_arr.shape[0],
                boundary_arr.shape[1],
                boundary_arr.shape[2],
                w_arr.shape[2],
                w_arr.shape[3],
            )
        size = 1
        for dim in range(5):
            shapes[5 * index + dim] = shape[dim]
            size *= shape[dim]
        total_size += size
        offsets[index + 1] = total_size

    if total_size * sizeof(double) >= 1024 * 1024:
        data = np.ndarray(
            total_size,
            dtype=np.float64,
            buffer=_mmap.mmap(-1, total_size * sizeof(double)),
        )
    else:
        data = np.empty(total_size, dtype=np.float64)
    data[:] = 0.0
    for index in range(n_pairs):
        boundary3 = np.ascontiguousarray(
            boundary_blocks[boundary_index[index]],
            dtype=np.float64,
        )
        w4 = np.ascontiguousarray(
            w_blocks[w_index[index]],
            dtype=np.float64,
        )
        out_offset = offsets[index]
        if left_representation:
            x_dim = boundary3.shape[0]
            l_dim = boundary3.shape[1]
            k_dim = boundary3.shape[2]
            w_dim = w4.shape[1]
            a_dim = w4.shape[2]
            b_dim = w4.shape[3]
            for x in range(x_dim):
                for l in range(l_dim):
                    for k in range(k_dim):
                        boundary_value = boundary3[x, l, k]
                        if boundary_value == 0.0:
                            continue
                        for w in range(w_dim):
                            for a in range(a_dim):
                                for b in range(b_dim):
                                    w_value = w4[x, w, a, b]
                                    if w_value != 0.0:
                                        data[
                                            out_offset
                                            + ((((l * k_dim + k) * w_dim + w) * a_dim + a) * b_dim + b)
                                        ] += boundary_value * w_value
        else:
            w_dim = w4.shape[0]
            y_dim = w4.shape[1]
            d_dim = w4.shape[2]
            c_dim = w4.shape[3]
            q_dim = boundary3.shape[1]
            r_dim = boundary3.shape[2]
            for w in range(w_dim):
                for y in range(y_dim):
                    for d in range(d_dim):
                        for c in range(c_dim):
                            w_value = w4[w, y, d, c]
                            if w_value == 0.0:
                                continue
                            for q in range(q_dim):
                                for r in range(r_dim):
                                    boundary_value = boundary3[y, q, r]
                                    if boundary_value != 0.0:
                                        data[
                                            out_offset
                                            + ((((w * q_dim + q) * r_dim + r) * d_dim + d) * c_dim + c)
                                        ] += w_value * boundary_value
    return data, offsets, shape_offsets, shapes


cdef object _factorize_rank_coupled_left_real(object E, object W):
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t l_dim = E_arr.shape[1]
    cdef Py_ssize_t k_dim = E_arr.shape[2]
    cdef Py_ssize_t w_dim = W_arr.shape[1]
    cdef Py_ssize_t a_dim = W_arr.shape[2]
    cdef Py_ssize_t b_dim = W_arr.shape[3]
    cdef cnp.ndarray[cnp.double_t, ndim=5] out
    cdef Py_ssize_t x, l, k, w, a, b
    cdef double e_val
    cdef double w_val

    if W_arr.shape[0] != x_dim:
        return None
    out = np.zeros((l_dim, k_dim, w_dim, a_dim, b_dim), dtype=np.float64)
    for x in range(x_dim):
        for l in range(l_dim):
            for k in range(k_dim):
                e_val = E_arr[x, l, k]
                if e_val == 0.0:
                    continue
                for w in range(w_dim):
                    for a in range(a_dim):
                        for b in range(b_dim):
                            w_val = W_arr[x, w, a, b]
                            if w_val != 0.0:
                                out[l, k, w, a, b] += e_val * w_val
    return out


cdef object _factorize_rank_coupled_right_real(object W, object F):
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.float64)
    cdef Py_ssize_t w_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t d_dim = W_arr.shape[2]
    cdef Py_ssize_t c_dim = W_arr.shape[3]
    cdef Py_ssize_t q_dim = F_arr.shape[1]
    cdef Py_ssize_t r_dim = F_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=5] out
    cdef Py_ssize_t w, y, q, r, d, c
    cdef double f_val
    cdef double w_val

    if F_arr.shape[0] != y_dim:
        return None
    out = np.zeros((w_dim, q_dim, r_dim, d_dim, c_dim), dtype=np.float64)
    for w in range(w_dim):
        for y in range(y_dim):
            for d in range(d_dim):
                for c in range(c_dim):
                    w_val = W_arr[w, y, d, c]
                    if w_val == 0.0:
                        continue
                    for q in range(q_dim):
                        for r in range(r_dim):
                            f_val = F_arr[y, q, r]
                            if f_val != 0.0:
                                out[w, q, r, d, c] += w_val * f_val
    return out


cdef object _factorize_rank_coupled_left_complex(object E, object W):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t l_dim = E_arr.shape[1]
    cdef Py_ssize_t k_dim = E_arr.shape[2]
    cdef Py_ssize_t w_dim = W_arr.shape[1]
    cdef Py_ssize_t a_dim = W_arr.shape[2]
    cdef Py_ssize_t b_dim = W_arr.shape[3]
    cdef cnp.ndarray[cnp.complex128_t, ndim=5] out
    cdef Py_ssize_t x, l, k, w, a, b
    cdef double complex e_val
    cdef double complex w_val

    if W_arr.shape[0] != x_dim:
        return None
    out = np.zeros((l_dim, k_dim, w_dim, a_dim, b_dim), dtype=np.complex128)
    for x in range(x_dim):
        for l in range(l_dim):
            for k in range(k_dim):
                e_val = E_arr[x, l, k]
                if e_val == 0.0:
                    continue
                for w in range(w_dim):
                    for a in range(a_dim):
                        for b in range(b_dim):
                            w_val = W_arr[x, w, a, b]
                            if w_val != 0.0:
                                out[l, k, w, a, b] += e_val * w_val
    return out


cdef object _factorize_rank_coupled_right_complex(object W, object F):
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef Py_ssize_t w_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t d_dim = W_arr.shape[2]
    cdef Py_ssize_t c_dim = W_arr.shape[3]
    cdef Py_ssize_t q_dim = F_arr.shape[1]
    cdef Py_ssize_t r_dim = F_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=5] out
    cdef Py_ssize_t w, y, q, r, d, c
    cdef double complex f_val
    cdef double complex w_val

    if F_arr.shape[0] != y_dim:
        return None
    out = np.zeros((w_dim, q_dim, r_dim, d_dim, c_dim), dtype=np.complex128)
    for w in range(w_dim):
        for y in range(y_dim):
            for d in range(d_dim):
                for c in range(c_dim):
                    w_val = W_arr[w, y, d, c]
                    if w_val == 0.0:
                        continue
                    for q in range(q_dim):
                        for r in range(r_dim):
                            f_val = F_arr[y, q, r]
                            if f_val != 0.0:
                                out[w, q, r, d, c] += w_val * f_val
    return out


def build_su2_qchem_factor_matches(
    object basis_left_ids,
    object basis_p1_ids,
    object basis_p2_ids,
    object basis_right_ids,
    object left_key_map,
    object right_key_map,
    object out_map,
    object left_entry_offsets,
    object left_out_boundary_ids,
    object left_out_physical_ids,
    object left_middle_ids,
    object right_entry_offsets,
    object right_out_boundary_ids,
    object right_out_physical_ids,
    object right_middle_ids,
):
    """
    Build packed SU(2) qchem factor matches from integer table metadata.

    Returns ``(input_indices, output_indices, left_entry_indices,
    right_entry_indices)`` as int64 arrays.  Python owns sector encoding and
    dense lookup-table construction; this helper keeps the heavy nested
    left/right schedule matching out of Python.
    """

    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_l = np.ascontiguousarray(basis_left_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_p1 = np.ascontiguousarray(basis_p1_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_p2 = np.ascontiguousarray(basis_p2_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] b_r = np.ascontiguousarray(basis_right_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] l_key = np.ascontiguousarray(left_key_map, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] r_key = np.ascontiguousarray(right_key_map, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=4] out_lookup = np.ascontiguousarray(out_map, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_offsets = np.ascontiguousarray(left_entry_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_out_b = np.ascontiguousarray(left_out_boundary_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_out_p = np.ascontiguousarray(left_out_physical_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_mid = np.ascontiguousarray(left_middle_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_offsets = np.ascontiguousarray(right_entry_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_out_b = np.ascontiguousarray(right_out_boundary_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_out_p = np.ascontiguousarray(right_out_physical_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_mid = np.ascontiguousarray(right_middle_ids, dtype=np.int64)
    cdef Py_ssize_t n_basis = b_l.shape[0]
    cdef Py_ssize_t in_idx
    cdef Py_ssize_t lrow
    cdef Py_ssize_t rrow
    cdef Py_ssize_t lstart
    cdef Py_ssize_t lstop
    cdef Py_ssize_t rstart
    cdef Py_ssize_t rstop
    cdef Py_ssize_t lidx
    cdef Py_ssize_t ridx
    cdef long out_idx
    cdef list in_indices = []
    cdef list out_indices = []
    cdef list left_indices = []
    cdef list right_indices = []

    for in_idx in range(n_basis):
        if b_l[in_idx] < 0 or b_p1[in_idx] < 0 or b_p2[in_idx] < 0 or b_r[in_idx] < 0:
            continue
        lrow = l_key[b_l[in_idx], b_p1[in_idx]]
        if lrow < 0:
            continue
        rrow = r_key[b_r[in_idx], b_p2[in_idx]]
        if rrow < 0:
            continue
        lstart = l_offsets[lrow]
        lstop = l_offsets[lrow + 1]
        rstart = r_offsets[rrow]
        rstop = r_offsets[rrow + 1]
        for lidx in range(lstart, lstop):
            for ridx in range(rstart, rstop):
                if l_mid[lidx] != r_mid[ridx]:
                    continue
                out_idx = out_lookup[
                    l_out_b[lidx],
                    l_out_p[lidx],
                    r_out_p[ridx],
                    r_out_b[ridx],
                ]
                if out_idx < 0:
                    continue
                in_indices.append(in_idx)
                out_indices.append(out_idx)
                left_indices.append(lidx)
                right_indices.append(ridx)
    return (
        np.asarray(in_indices, dtype=np.int64),
        np.asarray(out_indices, dtype=np.int64),
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
    )


def build_su2_qchem_parent_blocks_from_matches(
    object basis_shapes,
    object entry_comp_ids,
    object entry_comp_starts,
    object component_dims,
    object in_indices,
    object out_indices,
    object left_indices,
    object right_indices,
    object left_factor_data,
    object left_factor_offsets,
    object left_factor_shape_offsets,
    object left_factor_shapes,
    object left_factor_indices,
    object right_factor_data,
    object right_factor_offsets,
    object right_factor_shape_offsets,
    object right_factor_shapes,
    object right_factor_indices,
):
    """
    Assemble component parent blocks from packed SU(2) qchem matches.

    This keeps the high-volume factor contractions and block updates in Cython.
    Factor payloads are consumed from raw packed arrays owned by the Python
    packed factor tables.
    """

    cdef cnp.ndarray[cnp.int64_t, ndim=2] shapes = np.ascontiguousarray(basis_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] comps = np.ascontiguousarray(entry_comp_ids, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] starts = np.ascontiguousarray(entry_comp_starts, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] comp_dims = np.ascontiguousarray(component_dims, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_arr = np.ascontiguousarray(in_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_arr = np.ascontiguousarray(out_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_arr = np.ascontiguousarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_arr = np.ascontiguousarray(right_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] l_data = np.ascontiguousarray(left_factor_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_offsets = np.ascontiguousarray(left_factor_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_shape_offsets = np.ascontiguousarray(left_factor_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_shapes = np.ascontiguousarray(left_factor_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] l_factor_indices = np.ascontiguousarray(left_factor_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] r_data = np.ascontiguousarray(right_factor_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_offsets = np.ascontiguousarray(right_factor_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_shape_offsets = np.ascontiguousarray(right_factor_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_shapes = np.ascontiguousarray(right_factor_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] r_factor_indices = np.ascontiguousarray(right_factor_indices, dtype=np.int64)
    cdef Py_ssize_t n_matches = in_arr.shape[0]
    cdef Py_ssize_t match_idx
    cdef Py_ssize_t in_idx
    cdef Py_ssize_t out_idx
    cdef Py_ssize_t left_entry
    cdef Py_ssize_t right_entry
    cdef Py_ssize_t left_factor
    cdef Py_ssize_t right_factor
    cdef Py_ssize_t left_base
    cdef Py_ssize_t right_base
    cdef Py_ssize_t left_shape_base
    cdef Py_ssize_t right_shape_base
    cdef int in_comp
    cdef int out_comp
    cdef int kdim
    cdef int bdim
    cdef int cdim
    cdef int rdim
    cdef int ldim
    cdef int adim
    cdef int ddim
    cdef int qdim
    cdef int wdim
    cdef int l, a, d, q, k, b, c, r, w
    cdef Py_ssize_t row
    cdef Py_ssize_t col
    cdef Py_ssize_t out_start
    cdef Py_ssize_t in_start
    cdef double val
    cdef object key
    cdef object block_obj
    cdef dict blocks = {}
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] block
    cdef Py_ssize_t left_pos
    cdef Py_ssize_t right_pos

    for match_idx in range(n_matches):
        in_idx = in_arr[match_idx]
        out_idx = out_arr[match_idx]
        in_comp = <int>comps[in_idx]
        out_comp = <int>comps[out_idx]
        if in_comp < 0 or out_comp < 0:
            return None
        key = (in_comp, out_comp)
        block_obj = blocks.get(key)
        if block_obj is None:
            block_obj = np.zeros(
                (int(comp_dims[out_comp]), int(comp_dims[in_comp])),
                dtype=np.complex128,
            )
            blocks[key] = block_obj
        block = block_obj

        left_entry = left_arr[match_idx]
        right_entry = right_arr[match_idx]
        if left_entry < 0 or right_entry < 0:
            return None
        left_factor = l_factor_indices[left_entry]
        right_factor = r_factor_indices[right_entry]
        if left_factor < 0 or right_factor < 0:
            return None
        left_base = l_offsets[left_factor]
        right_base = r_offsets[right_factor]
        left_shape_base = l_shape_offsets[left_factor]
        right_shape_base = r_shape_offsets[right_factor]
        if l_shape_offsets[left_factor + 1] - left_shape_base != 5:
            return None
        if r_shape_offsets[right_factor + 1] - right_shape_base != 5:
            return None
        ldim = <int>l_shapes[left_shape_base]
        kdim = <int>l_shapes[left_shape_base + 1]
        wdim = <int>l_shapes[left_shape_base + 2]
        adim = <int>l_shapes[left_shape_base + 3]
        bdim = <int>l_shapes[left_shape_base + 4]
        if <int>r_shapes[right_shape_base] != wdim:
            return None
        qdim = <int>r_shapes[right_shape_base + 1]
        rdim = <int>r_shapes[right_shape_base + 2]
        ddim = <int>r_shapes[right_shape_base + 3]
        cdim = <int>r_shapes[right_shape_base + 4]
        if (
            shapes[in_idx, 0] != kdim
            or shapes[in_idx, 1] != bdim
            or shapes[in_idx, 2] != cdim
            or shapes[in_idx, 3] != rdim
            or shapes[out_idx, 0] != ldim
            or shapes[out_idx, 1] != adim
            or shapes[out_idx, 2] != ddim
            or shapes[out_idx, 3] != qdim
        ):
            return None
        in_start = starts[in_idx]
        out_start = starts[out_idx]
        for l in range(ldim):
            for a in range(adim):
                for d in range(ddim):
                    for q in range(qdim):
                        row = out_start + (((l * adim + a) * ddim + d) * qdim + q)
                        for k in range(kdim):
                            for b in range(bdim):
                                for c in range(cdim):
                                    for r in range(rdim):
                                        val = 0.0
                                        for w in range(wdim):
                                            left_pos = left_base + (((l * kdim + k) * wdim + w) * adim + a) * bdim + b
                                            right_pos = right_base + (((w * qdim + q) * rdim + r) * ddim + d) * cdim + c
                                            val += l_data[left_pos] * r_data[right_pos]
                                        if val != 0.0:
                                            col = in_start + (((k * bdim + b) * cdim + c) * rdim + r)
                                            block[row, col] += val
    return tuple(
        (int(key[0]), int(key[1]), np.ascontiguousarray(block))
        for key, block in sorted(blocks.items())
    )


def build_component_parent_block_layout(
    object basis_offsets,
    object entry_comp_ids,
    object entry_comp_starts,
    object entry_sizes,
    object in_indices,
    object out_indices,
    object left_indices,
    object right_indices,
    object left_factor_indices,
    object left_factor_shape_offsets,
    object left_factor_shapes,
    object right_factor_indices,
    object right_factor_shape_offsets,
    object right_factor_shapes,
):
    """
    Group packed qchem matches into component-parent block schedule rows.

    The returned layout contains only reduced-sector metadata.  Numerical
    contraction remains in the BLAS-backed reduced parent-block builder.
    """

    cdef object offsets = np.ascontiguousarray(
        basis_offsets, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] comps = np.ascontiguousarray(
        entry_comp_ids, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] starts = np.ascontiguousarray(
        entry_comp_starts, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sizes = np.ascontiguousarray(
        entry_sizes, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_arr = np.ascontiguousarray(
        in_indices, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_arr = np.ascontiguousarray(
        out_indices, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_arr = np.ascontiguousarray(
        left_indices, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_arr = np.ascontiguousarray(
        right_indices, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_factors = np.ascontiguousarray(
        left_factor_indices, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_shape_offsets = np.ascontiguousarray(
        left_factor_shape_offsets, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_shapes = np.ascontiguousarray(
        left_factor_shapes, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_factors = np.ascontiguousarray(
        right_factor_indices, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_shape_offsets = np.ascontiguousarray(
        right_factor_shape_offsets, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_shapes = np.ascontiguousarray(
        right_factor_shapes, dtype=np.int64
    )
    cdef Py_ssize_t n_matches = in_arr.shape[0]
    cdef Py_ssize_t idx
    cdef Py_ssize_t in_idx
    cdef Py_ssize_t out_idx
    cdef Py_ssize_t left_entry
    cdef Py_ssize_t right_entry
    cdef Py_ssize_t left_factor
    cdef Py_ssize_t right_factor
    cdef Py_ssize_t left_shape_start
    cdef Py_ssize_t right_shape_start
    cdef int in_comp
    cdef int out_comp
    cdef object key
    cdef object bucket
    cdef dict grouped = {}
    cdef list keys
    cdef list schedule = []

    if (
        out_arr.shape[0] != n_matches
        or left_arr.shape[0] != n_matches
        or right_arr.shape[0] != n_matches
    ):
        return None
    if (
        offsets.shape[0] != comps.shape[0]
        or starts.shape[0] != comps.shape[0]
        or sizes.shape[0] != comps.shape[0]
    ):
        return None

    for idx in range(n_matches):
        in_idx = in_arr[idx]
        out_idx = out_arr[idx]
        if (
            in_idx < 0
            or out_idx < 0
            or in_idx >= comps.shape[0]
            or out_idx >= comps.shape[0]
        ):
            return None
        in_comp = <int>comps[in_idx]
        out_comp = <int>comps[out_idx]
        if in_comp < 0 or out_comp < 0:
            return None
        left_entry = left_arr[idx]
        right_entry = right_arr[idx]
        if (
            left_entry < 0
            or right_entry < 0
            or left_entry >= left_factors.shape[0]
            or right_entry >= right_factors.shape[0]
        ):
            return None
        left_factor = left_factors[left_entry]
        right_factor = right_factors[right_entry]
        if (
            left_factor < 0
            or right_factor < 0
            or left_factor + 1 >= left_shape_offsets.shape[0]
            or right_factor + 1 >= right_shape_offsets.shape[0]
        ):
            return None
        left_shape_start = left_shape_offsets[left_factor]
        right_shape_start = right_shape_offsets[right_factor]
        if (
            left_shape_offsets[left_factor + 1] - left_shape_start != 5
            or right_shape_offsets[right_factor + 1] - right_shape_start != 5
        ):
            return None
        key = (
            int(in_idx),
            int(out_idx),
            int(in_comp),
            int(out_comp),
            int(starts[in_idx]),
            int(starts[in_idx] + sizes[in_idx]),
            int(starts[out_idx]),
            int(starts[out_idx] + sizes[out_idx]),
            (
                int(left_shapes[left_shape_start]),
                int(left_shapes[left_shape_start + 1]),
                int(left_shapes[left_shape_start + 2]),
                int(left_shapes[left_shape_start + 3]),
                int(left_shapes[left_shape_start + 4]),
            ),
            (
                int(right_shapes[right_shape_start]),
                int(right_shapes[right_shape_start + 1]),
                int(right_shapes[right_shape_start + 2]),
                int(right_shapes[right_shape_start + 3]),
                int(right_shapes[right_shape_start + 4]),
            ),
        )
        bucket = grouped.get(key)
        if bucket is None:
            bucket = ([], [])
            grouped[key] = bucket
        bucket[0].append(int(left_entry))
        bucket[1].append(int(right_entry))

    keys = list(grouped)
    keys.sort(key=lambda item: (int(item[2]), int(item[3]), int(offsets[int(item[1])])))
    for key in keys:
        bucket = grouped[key]
        schedule.append(
            (
                int(key[0]),
                int(key[1]),
                int(key[2]),
                int(key[3]),
                int(key[4]),
                int(key[5]),
                int(key[6]),
                int(key[7]),
                tuple(bucket[0]),
                tuple(bucket[1]),
            )
        )
    return tuple(schedule)


def pack_component_parent_factor_group(
    object group,
    object role,
    object factor_data,
    object factor_offsets,
    object factor_shape_offsets,
    object factor_shapes,
    object factor_indices,
):
    """
    Pack one same-shape qchem factor group for the parent-block GEMM.
    """

    cdef cnp.ndarray[cnp.int64_t, ndim=1] group_arr = np.ascontiguousarray(
        group, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.double_t, ndim=1] data = np.ascontiguousarray(
        factor_data, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] offsets = np.ascontiguousarray(
        factor_offsets, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] shape_offsets = np.ascontiguousarray(
        factor_shape_offsets, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] shapes = np.ascontiguousarray(
        factor_shapes, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1] indices = np.ascontiguousarray(
        factor_indices, dtype=np.int64
    )
    cdef Py_ssize_t tdim = group_arr.shape[0]
    cdef Py_ssize_t entry_idx
    cdef Py_ssize_t factor_idx
    cdef Py_ssize_t shape_start
    cdef Py_ssize_t base
    cdef Py_ssize_t factor_size
    cdef Py_ssize_t t
    cdef Py_ssize_t pos
    cdef int d0
    cdef int d1
    cdef int d2
    cdef int d3
    cdef int d4
    cdef int l
    cdef int k
    cdef int w
    cdef int a
    cdef int b
    cdef int q
    cdef int r
    cdef int d
    cdef int c
    cdef Py_ssize_t row
    cdef Py_ssize_t col
    cdef cnp.ndarray[cnp.double_t, ndim=1] stack_flat
    cdef cnp.ndarray[cnp.double_t, ndim=2] mat
    cdef object role_name = str(role)
    cdef object dims

    if tdim <= 0 or role_name not in {"left", "right"}:
        return None
    entry_idx = group_arr[0]
    if entry_idx < 0 or entry_idx >= indices.shape[0]:
        return None
    factor_idx = indices[entry_idx]
    if factor_idx < 0 or factor_idx + 1 >= shape_offsets.shape[0]:
        return None
    shape_start = shape_offsets[factor_idx]
    if shape_offsets[factor_idx + 1] - shape_start != 5:
        return None
    d0 = <int>shapes[shape_start]
    d1 = <int>shapes[shape_start + 1]
    d2 = <int>shapes[shape_start + 2]
    d3 = <int>shapes[shape_start + 3]
    d4 = <int>shapes[shape_start + 4]
    factor_size = <Py_ssize_t>d0 * d1 * d2 * d3 * d4
    stack_flat = np.empty(tdim * factor_size, dtype=np.float64)

    for t in range(tdim):
        entry_idx = group_arr[t]
        if entry_idx < 0 or entry_idx >= indices.shape[0]:
            return None
        factor_idx = indices[entry_idx]
        if factor_idx < 0 or factor_idx + 1 >= offsets.shape[0]:
            return None
        shape_start = shape_offsets[factor_idx]
        if (
            shape_offsets[factor_idx + 1] - shape_start != 5
            or shapes[shape_start] != d0
            or shapes[shape_start + 1] != d1
            or shapes[shape_start + 2] != d2
            or shapes[shape_start + 3] != d3
            or shapes[shape_start + 4] != d4
        ):
            return None
        base = offsets[factor_idx]
        if offsets[factor_idx + 1] - base != factor_size:
            return None
        for pos in range(factor_size):
            stack_flat[t * factor_size + pos] = data[base + pos]

    if role_name == "left":
        # factor[t,l,k,w,a,b] -> matrix[l,a,k,b ; t,w]
        mat = np.empty((d0 * d3 * d1 * d4, tdim * d2), dtype=np.float64)
        for t in range(tdim):
            base = t * factor_size
            for l in range(d0):
                for k in range(d1):
                    for w in range(d2):
                        for a in range(d3):
                            for b in range(d4):
                                pos = ((((l * d1 + k) * d2 + w) * d3 + a) * d4 + b)
                                row = (((l * d3 + a) * d1 + k) * d4 + b)
                                col = t * d2 + w
                                mat[row, col] = stack_flat[base + pos]
        dims = (int(tdim), d0, d1, d2, d3, d4)
    else:
        # factor[t,w,q,r,d,c] -> matrix[t,w ; d,q,c,r]
        mat = np.empty((tdim * d0, d3 * d1 * d4 * d2), dtype=np.float64)
        for t in range(tdim):
            base = t * factor_size
            for w in range(d0):
                for q in range(d1):
                    for r in range(d2):
                        for d in range(d3):
                            for c in range(d4):
                                pos = ((((w * d1 + q) * d2 + r) * d3 + d) * d4 + c)
                                row = t * d0 + w
                                col = (((d * d1 + q) * d4 + c) * d2 + r)
                                mat[row, col] = stack_flat[base + pos]
        dims = (int(tdim), d0, d1, d2, d3, d4)
    return (
        stack_flat.reshape((int(tdim), d0, d1, d2, d3, d4)),
        (mat, dims),
    )


def contract_rank_coupled_left_scalar_channel(object E, object A, object W, object B):
    """
    Contract ``x=1, y=1`` rank-coupled left boundary update blocks.

    Shapes are ``E[1,i,j]``, ``A[i,p,r]``, ``W[1,1,p,q]``,
    ``B[j,q,s]``; the result has shape ``(1,r,s)``.
    """

    if np.iscomplexobj(E) or np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(B):
        return _contract_rank_coupled_left_scalar_channel_complex(E, A, W, B)
    return _contract_rank_coupled_left_scalar_channel_real(E, A, W, B)


def contract_rank_coupled_right_scalar_channel(object A, object W, object F, object B):
    """
    Contract ``x=1, y=1`` rank-coupled right boundary update blocks.

    Shapes are ``A[i,p,r]``, ``W[1,1,p,q]``, ``F[1,r,s]``,
    ``B[j,q,s]``; the result has shape ``(1,i,j)``.
    """

    if np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(F) or np.iscomplexobj(B):
        return _contract_rank_coupled_right_scalar_channel_complex(A, W, F, B)
    return _contract_rank_coupled_right_scalar_channel_real(A, W, F, B)


def contract_rank_coupled_left_general(object E, object A, object W, object B):
    """
    Contract a small rank-coupled left boundary update block.

    Shapes are ``E[x,i,j]``, ``A[i,p,r]``, ``W[x,y,p,q]``,
    ``B[j,q,s]``; the result has shape ``(y,r,s)``.  This is the packed
    qchem sweep's small-block replacement for the three Python-dispatched
    ``einsum`` calls used by the reference path.
    """

    if np.iscomplexobj(E) or np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(B):
        return _contract_rank_coupled_left_general_complex(E, A, W, B)
    return _contract_rank_coupled_left_general_real(E, A, W, B)


def contract_rank_coupled_right_general(object A, object W, object F, object B):
    """
    Contract a small rank-coupled right boundary update block.

    Shapes are ``A[i,p,r]``, ``W[x,y,p,q]``, ``F[y,r,s]``,
    ``B[j,q,s]``; the result has shape ``(x,i,j)``.
    """

    if np.iscomplexobj(A) or np.iscomplexobj(W) or np.iscomplexobj(F) or np.iscomplexobj(B):
        return _contract_rank_coupled_right_general_complex(A, W, F, B)
    return _contract_rank_coupled_right_general_real(A, W, F, B)


def accumulate_rank_coupled_left_terms(
    object target_blocks,
    object e_blocks,
    object A,
    object B,
    object reduced_terms,
    long max_work,
):
    """
    Accumulate a batch of small left-boundary rank-coupled terms in place.

    ``reduced_terms`` contains ``(left_channel, right_channel, W_block)``
    entries.  The function returns ``True`` when the batch was handled by the
    real-valued Cython path and ``False`` when the Python reference path should
    be used instead.
    """

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef object term
    cdef object W_obj
    cdef object item
    cdef list validated = []
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        return False
    A_arr = np.ascontiguousarray(A, dtype=np.float64)
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    for term in reduced_terms:
        left_idx = int(term[0])
        right_idx = int(term[1])
        if left_idx < 0 or right_idx < 0:
            return False
        if left_idx >= len(e_blocks) or right_idx >= len(target_blocks):
            continue
        W_obj = term[2]
        if (
            np.iscomplexobj(W_obj)
            or np.iscomplexobj(e_blocks[left_idx])
            or np.iscomplexobj(target_blocks[right_idx])
        ):
            return False
        E_arr = np.ascontiguousarray(e_blocks[left_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[right_idx], dtype=np.float64)
        work = (
            <long>E_arr.shape[0]
            * <long>W_arr.shape[1]
            * <long>E_arr.shape[1]
            * <long>E_arr.shape[2]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>B_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if A_arr.shape[0] != E_arr.shape[1] or W_arr.shape[0] != E_arr.shape[0] or W_arr.shape[2] != A_arr.shape[1]:
            return False
        if B_arr.shape[0] != E_arr.shape[2] or B_arr.shape[1] != W_arr.shape[3]:
            return False
        if out_arr.shape[0] != W_arr.shape[1] or out_arr.shape[1] != A_arr.shape[2] or out_arr.shape[2] != B_arr.shape[2]:
            return False
        validated.append((left_idx, right_idx, W_obj))
    for item in validated:
        left_idx = int(item[0])
        right_idx = int(item[1])
        W_obj = item[2]
        E_arr = np.ascontiguousarray(e_blocks[left_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[right_idx], dtype=np.float64)
        if not _accumulate_rank_coupled_left_general_real(E_arr, A_arr, W_arr, B_arr, out_arr):
            return False
    return True


def accumulate_rank_coupled_right_terms(
    object target_blocks,
    object A,
    object B,
    object f_blocks,
    object reduced_terms,
    long max_work,
):
    """
    Accumulate a batch of small right-boundary rank-coupled terms in place.
    """

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef object term
    cdef object W_obj
    cdef object item
    cdef list validated = []
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        return False
    A_arr = np.ascontiguousarray(A, dtype=np.float64)
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    for term in reduced_terms:
        left_idx = int(term[0])
        right_idx = int(term[1])
        if left_idx < 0 or right_idx < 0:
            return False
        if right_idx >= len(f_blocks) or left_idx >= len(target_blocks):
            continue
        W_obj = term[2]
        if (
            np.iscomplexobj(W_obj)
            or np.iscomplexobj(f_blocks[right_idx])
            or np.iscomplexobj(target_blocks[left_idx])
        ):
            return False
        F_arr = np.ascontiguousarray(f_blocks[right_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[left_idx], dtype=np.float64)
        work = (
            <long>W_arr.shape[0]
            * <long>F_arr.shape[0]
            * <long>A_arr.shape[0]
            * <long>B_arr.shape[0]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>F_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if W_arr.shape[2] != A_arr.shape[1] or F_arr.shape[0] != W_arr.shape[1] or F_arr.shape[1] != A_arr.shape[2]:
            return False
        if B_arr.shape[1] != W_arr.shape[3] or B_arr.shape[2] != F_arr.shape[2]:
            return False
        if out_arr.shape[0] != W_arr.shape[0] or out_arr.shape[1] != A_arr.shape[0] or out_arr.shape[2] != B_arr.shape[0]:
            return False
        validated.append((left_idx, right_idx, W_obj))
    for item in validated:
        left_idx = int(item[0])
        right_idx = int(item[1])
        W_obj = item[2]
        F_arr = np.ascontiguousarray(f_blocks[right_idx], dtype=np.float64)
        W_arr = np.ascontiguousarray(W_obj, dtype=np.float64)
        out_arr = np.asarray(target_blocks[left_idx], dtype=np.float64)
        if not _accumulate_rank_coupled_right_general_real(A_arr, W_arr, F_arr, B_arr, out_arr):
            return False
    return True


def accumulate_rank_coupled_left_real_terms(
    object target_blocks,
    object e_blocks,
    object A,
    object B,
    object left_indices,
    object right_indices,
    object w_blocks,
    long max_work,
):
    """
    Accumulate prevalidated real left-boundary rank-coupled terms in place.

    This entry point is used by the qchem sweep after Python has already
    selected real-valued contiguous blocks.  It intentionally avoids the
    per-term ``np.iscomplexobj`` and conversion work in the generic fallback.
    """

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_arr = np.asarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_arr = np.asarray(right_indices, dtype=np.int64)
    cdef Py_ssize_t n_terms = left_arr.shape[0]
    cdef Py_ssize_t t
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if right_arr.shape[0] != n_terms or len(w_blocks) != n_terms:
        return False
    try:
        A_arr = A
        B_arr = B
    except (TypeError, ValueError):
        return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if left_idx < 0 or right_idx < 0:
            return False
        if left_idx >= len(e_blocks) or right_idx >= len(target_blocks):
            continue
        try:
            E_arr = e_blocks[left_idx]
            W_arr = w_blocks[t]
            out_arr = target_blocks[right_idx]
        except (TypeError, ValueError):
            return False
        work = (
            <long>E_arr.shape[0]
            * <long>W_arr.shape[1]
            * <long>E_arr.shape[1]
            * <long>E_arr.shape[2]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>B_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if A_arr.shape[0] != E_arr.shape[1] or W_arr.shape[0] != E_arr.shape[0] or W_arr.shape[2] != A_arr.shape[1]:
            return False
        if B_arr.shape[0] != E_arr.shape[2] or B_arr.shape[1] != W_arr.shape[3]:
            return False
        if out_arr.shape[0] != W_arr.shape[1] or out_arr.shape[1] != A_arr.shape[2] or out_arr.shape[2] != B_arr.shape[2]:
            return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if left_idx >= len(e_blocks) or right_idx >= len(target_blocks):
            continue
        E_arr = e_blocks[left_idx]
        W_arr = w_blocks[t]
        out_arr = target_blocks[right_idx]
        if not _accumulate_rank_coupled_left_general_real(E_arr, A_arr, W_arr, B_arr, out_arr):
            return False
    return True


def accumulate_rank_coupled_right_real_terms(
    object target_blocks,
    object A,
    object B,
    object f_blocks,
    object left_indices,
    object right_indices,
    object w_blocks,
    long max_work,
):
    """Accumulate prevalidated real right-boundary rank-coupled terms."""

    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr
    cdef cnp.ndarray[cnp.double_t, ndim=3] out_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] left_arr = np.asarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] right_arr = np.asarray(right_indices, dtype=np.int64)
    cdef Py_ssize_t n_terms = left_arr.shape[0]
    cdef Py_ssize_t t
    cdef Py_ssize_t left_idx
    cdef Py_ssize_t right_idx
    cdef long work

    if right_arr.shape[0] != n_terms or len(w_blocks) != n_terms:
        return False
    try:
        A_arr = A
        B_arr = B
    except (TypeError, ValueError):
        return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if left_idx < 0 or right_idx < 0:
            return False
        if right_idx >= len(f_blocks) or left_idx >= len(target_blocks):
            continue
        try:
            F_arr = f_blocks[right_idx]
            W_arr = w_blocks[t]
            out_arr = target_blocks[left_idx]
        except (TypeError, ValueError):
            return False
        work = (
            <long>W_arr.shape[0]
            * <long>F_arr.shape[0]
            * <long>A_arr.shape[0]
            * <long>B_arr.shape[0]
            * <long>A_arr.shape[1]
            * <long>W_arr.shape[3]
            * <long>A_arr.shape[2]
            * <long>F_arr.shape[2]
        )
        if max_work > 0 and work > max_work:
            return False
        if W_arr.shape[2] != A_arr.shape[1] or F_arr.shape[0] != W_arr.shape[1] or F_arr.shape[1] != A_arr.shape[2]:
            return False
        if B_arr.shape[1] != W_arr.shape[3] or B_arr.shape[2] != F_arr.shape[2]:
            return False
        if out_arr.shape[0] != W_arr.shape[0] or out_arr.shape[1] != A_arr.shape[0] or out_arr.shape[2] != B_arr.shape[0]:
            return False
    for t in range(n_terms):
        left_idx = <Py_ssize_t>left_arr[t]
        right_idx = <Py_ssize_t>right_arr[t]
        if right_idx >= len(f_blocks) or left_idx >= len(target_blocks):
            continue
        F_arr = f_blocks[right_idx]
        W_arr = w_blocks[t]
        out_arr = target_blocks[left_idx]
        if not _accumulate_rank_coupled_right_general_real(A_arr, W_arr, F_arr, B_arr, out_arr):
            return False
    return True


cdef bint _apply_parent_block_batch_impl(
    object blocks,
    object in_comps,
    object out_comps,
    object parent_inputs,
    object parent_outputs,
    bint apply,
):
    """
    Apply same-shape parent component blocks without per-matvec stack buffers.

    ``blocks[n]`` maps ``parent_inputs[in_comps[n]]`` into
    ``parent_outputs[out_comps[n]]``.  This is the hot path for the packed SU2
    qchem action when direct parent blocks are present.
    """

    cdef cnp.ndarray[cnp.complex128_t, ndim=3] block_arr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_arr = np.asarray(in_comps, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_arr = np.asarray(out_comps, dtype=np.int64)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] inp
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] out
    cdef Py_ssize_t n_terms
    cdef Py_ssize_t t
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t rows
    cdef Py_ssize_t cols
    cdef Py_ssize_t in_idx
    cdef Py_ssize_t out_idx
    cdef double complex acc

    try:
        block_arr = blocks
    except (TypeError, ValueError):
        return False
    n_terms = block_arr.shape[0]
    if in_arr.shape[0] != n_terms or out_arr.shape[0] != n_terms:
        return False
    rows = block_arr.shape[1]
    cols = block_arr.shape[2]
    for t in range(n_terms):
        in_idx = <Py_ssize_t>in_arr[t]
        out_idx = <Py_ssize_t>out_arr[t]
        if in_idx < 0 or out_idx < 0:
            return False
        if in_idx >= len(parent_inputs) or out_idx >= len(parent_outputs):
            return False
        try:
            inp = parent_inputs[in_idx]
            out = parent_outputs[out_idx]
        except (TypeError, ValueError):
            return False
        if inp.shape[0] != cols or out.shape[0] != rows:
            return False
        if not apply:
            continue
        for i in range(rows):
            acc = 0.0
            for j in range(cols):
                acc += block_arr[t, i, j] * inp[j]
            out[i] += acc
    return True


def apply_parent_block_batch(
    object blocks,
    object in_comps,
    object out_comps,
    object parent_inputs,
    object parent_outputs,
):
    """
    Apply one same-shape parent component block batch.

    Returns ``False`` before mutating outputs when the layout is unsupported.
    """

    if not _apply_parent_block_batch_impl(
        blocks,
        in_comps,
        out_comps,
        parent_inputs,
        parent_outputs,
        False,
    ):
        return False
    return _apply_parent_block_batch_impl(
        blocks,
        in_comps,
        out_comps,
        parent_inputs,
        parent_outputs,
        True,
    )


def apply_parent_block_batches(
    object batches,
    object parent_inputs,
    object parent_outputs,
):
    """
    Apply all parent-block batches in one Cython call.

    ``batches`` is the tuple of Python ``_ParentBlockBatch`` objects owned by
    ``SU2LocalAction``.  Validation is separated from mutation so Python can
    safely fall back to its reference path when any batch is unsupported.
    """

    cdef object batch
    for batch in batches:
        if not _apply_parent_block_batch_impl(
            batch.blocks,
            batch.in_comps,
            batch.out_comps,
            parent_inputs,
            parent_outputs,
            False,
        ):
            return False
    for batch in batches:
        if not _apply_parent_block_batch_impl(
            batch.blocks,
            batch.in_comps,
            batch.out_comps,
            parent_inputs,
            parent_outputs,
            True,
        ):
            return False
    return True


cdef object _contract_rank_coupled_left_scalar_channel_real(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((1, r_dim, s_dim), dtype=np.float64)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double e_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for j in range(j_dim):
                        e_val = E_arr[0, i, j] * a_val
                        if e_val == 0.0:
                            continue
                        for s in range(s_dim):
                            out[0, r, s] += e_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_right_scalar_channel_real(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out = np.zeros((1, i_dim, j_dim), dtype=np.float64)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double f_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for s in range(s_dim):
                        f_val = F_arr[0, r, s] * a_val
                        if f_val == 0.0:
                            continue
                        for j in range(j_dim):
                            out[0, i, j] += f_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_left_general_real(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double e_val

    if A_arr.shape[0] != i_dim or W_arr.shape[0] != x_dim or W_arr.shape[2] != p_dim:
        return None
    if B_arr.shape[0] != j_dim or B_arr.shape[1] != q_dim:
        return None
    out = np.zeros((y_dim, r_dim, s_dim), dtype=np.float64)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for j in range(j_dim):
                                e_val = E_arr[x, i, j] * a_val
                                if e_val == 0.0:
                                    continue
                                for s in range(s_dim):
                                    out[y, r, s] += e_val * B_arr[j, q, s]
    return out


cdef bint _accumulate_rank_coupled_left_general_real(
    cnp.ndarray[cnp.double_t, ndim=3] E_arr,
    cnp.ndarray[cnp.double_t, ndim=3] A_arr,
    cnp.ndarray[cnp.double_t, ndim=4] W_arr,
    cnp.ndarray[cnp.double_t, ndim=3] B_arr,
    cnp.ndarray[cnp.double_t, ndim=3] out,
):
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double e_val

    if A_arr.shape[0] != i_dim or W_arr.shape[0] != x_dim or W_arr.shape[2] != p_dim:
        return False
    if B_arr.shape[0] != j_dim or B_arr.shape[1] != q_dim:
        return False
    if out.shape[0] != y_dim or out.shape[1] != r_dim or out.shape[2] != s_dim:
        return False
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for j in range(j_dim):
                                e_val = E_arr[x, i, j] * a_val
                                if e_val == 0.0:
                                    continue
                                for s in range(s_dim):
                                    out[y, r, s] += e_val * B_arr[j, q, s]
    return True


cdef object _contract_rank_coupled_right_general_real(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.double_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.float64)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = F_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double f_val

    if W_arr.shape[2] != p_dim or F_arr.shape[0] != y_dim or F_arr.shape[1] != r_dim:
        return None
    if B_arr.shape[1] != q_dim or B_arr.shape[2] != s_dim:
        return None
    out = np.zeros((x_dim, i_dim, j_dim), dtype=np.float64)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for s in range(s_dim):
                                f_val = F_arr[y, r, s] * a_val
                                if f_val == 0.0:
                                    continue
                                for j in range(j_dim):
                                    out[x, i, j] += f_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_left_scalar_channel_complex(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out = np.zeros((1, r_dim, s_dim), dtype=np.complex128)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex e_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for j in range(j_dim):
                        e_val = E_arr[0, i, j] * a_val
                        if e_val == 0.0:
                            continue
                        for s in range(s_dim):
                            out[0, r, s] += e_val * B_arr[j, q, s]
    return out


cdef bint _accumulate_rank_coupled_right_general_real(
    cnp.ndarray[cnp.double_t, ndim=3] A_arr,
    cnp.ndarray[cnp.double_t, ndim=4] W_arr,
    cnp.ndarray[cnp.double_t, ndim=3] F_arr,
    cnp.ndarray[cnp.double_t, ndim=3] B_arr,
    cnp.ndarray[cnp.double_t, ndim=3] out,
):
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = F_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double coeff
    cdef double a_val
    cdef double f_val

    if W_arr.shape[2] != p_dim or F_arr.shape[0] != y_dim or F_arr.shape[1] != r_dim:
        return False
    if B_arr.shape[1] != q_dim or B_arr.shape[2] != s_dim:
        return False
    if out.shape[0] != x_dim or out.shape[1] != i_dim or out.shape[2] != j_dim:
        return False
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for s in range(s_dim):
                                f_val = F_arr[y, r, s] * a_val
                                if f_val == 0.0:
                                    continue
                                for j in range(j_dim):
                                    out[x, i, j] += f_val * B_arr[j, q, s]
    return True


cdef object _contract_rank_coupled_left_general_complex(object E, object A, object W, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] E_arr = np.ascontiguousarray(E, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t x_dim = E_arr.shape[0]
    cdef Py_ssize_t i_dim = E_arr.shape[1]
    cdef Py_ssize_t j_dim = E_arr.shape[2]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex e_val

    if A_arr.shape[0] != i_dim or W_arr.shape[0] != x_dim or W_arr.shape[2] != p_dim:
        return None
    if B_arr.shape[0] != j_dim or B_arr.shape[1] != q_dim:
        return None
    out = np.zeros((y_dim, r_dim, s_dim), dtype=np.complex128)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for j in range(j_dim):
                                e_val = E_arr[x, i, j] * a_val
                                if e_val == 0.0:
                                    continue
                                for s in range(s_dim):
                                    out[y, r, s] += e_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_right_general_complex(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t x_dim = W_arr.shape[0]
    cdef Py_ssize_t y_dim = W_arr.shape[1]
    cdef Py_ssize_t q_dim = W_arr.shape[3]
    cdef Py_ssize_t s_dim = F_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    cdef Py_ssize_t x, y, i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex f_val

    if W_arr.shape[2] != p_dim or F_arr.shape[0] != y_dim or F_arr.shape[1] != r_dim:
        return None
    if B_arr.shape[1] != q_dim or B_arr.shape[2] != s_dim:
        return None
    out = np.zeros((x_dim, i_dim, j_dim), dtype=np.complex128)
    for x in range(x_dim):
        for y in range(y_dim):
            for p in range(p_dim):
                for q in range(q_dim):
                    coeff = W_arr[x, y, p, q]
                    if coeff == 0.0:
                        continue
                    for i in range(i_dim):
                        for r in range(r_dim):
                            a_val = A_arr[i, p, r] * coeff
                            if a_val == 0.0:
                                continue
                            for s in range(s_dim):
                                f_val = F_arr[y, r, s] * a_val
                                if f_val == 0.0:
                                    continue
                                for j in range(j_dim):
                                    out[x, i, j] += f_val * B_arr[j, q, s]
    return out


cdef object _contract_rank_coupled_right_scalar_channel_complex(object A, object W, object F, object B):
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=4] W_arr = np.ascontiguousarray(W, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] F_arr = np.ascontiguousarray(F, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    cdef Py_ssize_t i_dim = A_arr.shape[0]
    cdef Py_ssize_t p_dim = A_arr.shape[1]
    cdef Py_ssize_t r_dim = A_arr.shape[2]
    cdef Py_ssize_t j_dim = B_arr.shape[0]
    cdef Py_ssize_t q_dim = B_arr.shape[1]
    cdef Py_ssize_t s_dim = B_arr.shape[2]
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out = np.zeros((1, i_dim, j_dim), dtype=np.complex128)
    cdef Py_ssize_t i, j, p, q, r, s
    cdef double complex coeff
    cdef double complex a_val
    cdef double complex f_val

    for p in range(p_dim):
        for q in range(q_dim):
            coeff = W_arr[0, 0, p, q]
            if coeff == 0.0:
                continue
            for i in range(i_dim):
                for r in range(r_dim):
                    a_val = A_arr[i, p, r] * coeff
                    if a_val == 0.0:
                        continue
                    for s in range(s_dim):
                        f_val = F_arr[0, r, s] * a_val
                        if f_val == 0.0:
                            continue
                        for j in range(j_dim):
                            out[0, i, j] += f_val * B_arr[j, q, s]
    return out


def apply_su2_block2_action(object plan, object x, object out):
    """
    Apply a packed SU(2) local action into ``out``.

    This entry keeps the Python ``SU2LocalAction`` object as the plan ABI while
    executing the packed local-action schedule directly.
    """

    cdef Py_ssize_t idx
    cdef object vector = np.asarray(x, dtype=complex).reshape(plan.dim)
    cdef list parent_inputs = []
    cdef list parent_outputs = []
    cdef object transform
    cdef object slc
    cdef object batch
    cdef object entries
    cdef object entry
    cdef object input_mats
    cdef object tmp
    cdef object tmp_mats
    cdef object contribs
    cdef int ldim
    cdef int adim
    cdef int ddim
    cdef int qdim
    cdef int in_comp
    cdef int out_comp
    cdef object block_in

    out[...] = 0.0
    for idx in range(len(plan.transforms)):
        transform = plan.transforms[idx]
        slc = plan.orth_slices[idx]
        parent_inputs.append(transform @ vector[slc])
        parent_outputs.append(np.zeros(plan.parent_dims[idx], dtype=complex))

    for entry in plan.parent_blocks:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        parent_outputs[out_comp] += entry[2] @ parent_inputs[in_comp]

    for batch in plan.batch_plans:
        entries = batch.entries
        input_mats = np.stack(
            [
                parent_inputs[int(entry.in_comp)][entry.in_slice]
                .reshape(entry.input_entry.shape)
                .reshape(
                    int(np.prod(entry.input_entry.shape[:2], dtype=int)),
                    int(np.prod(entry.input_entry.shape[2:], dtype=int)),
                )
                for entry in entries
            ],
            axis=0,
        )
        tmp = np.matmul(batch.left_mats, input_mats).reshape(
            (len(entries),) + tuple(batch.tmp_shape)
        )
        ldim, adim, ddim, qdim = (int(dim) for dim in batch.output_shape)
        tmp_mats = np.ascontiguousarray(
            tmp.transpose(0, 2, 4, 1, 3, 6, 5).reshape(
                len(entries),
                ldim * adim,
                -1,
            )
        )
        contribs = np.matmul(tmp_mats, batch.right_mats).reshape(
            len(entries),
            ldim * adim * ddim * qdim,
        )
        for idx, entry in enumerate(entries):
            parent_outputs[int(entry.out_comp)][entry.out_slice] += contribs[idx]

    for entry in plan.single_entries:
        in_comp = int(entry.in_comp)
        out_comp = int(entry.out_comp)
        block_in = parent_inputs[in_comp][entry.in_slice].reshape(
            entry.input_entry.shape
        )
        parent_outputs[out_comp][entry.out_slice] += entry.apply_block(block_in)

    for idx in range(len(plan.transforms)):
        transform = plan.transforms[idx]
        out[plan.orth_slices[idx]] = transform.conj().T @ parent_outputs[idx]
    return out


def build_component_parent_blocks(object plan, object component_dims):
    """
    Build component parent blocks from a component-direct factorized plan.

    This mirrors ``DirectOrthonormalFactorizedTable._build_component_parent_blocks``
    while keeping the outer loop and dictionary assembly in the extension.
    """

    cdef object blocks = {}
    cdef object entry
    cdef object term
    cdef object key
    cdef object block
    cdef object kernel
    cdef int in_comp
    cdef int out_comp
    cdef object in_slice
    cdef object out_slice

    if plan is None:
        return None
    for entry in plan:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        in_slice = entry[2]
        out_slice = entry[3]
        term = entry[4]
        key = (in_comp, out_comp)
        block = blocks.get(key)
        if block is None:
            block = np.zeros(
                (int(component_dims[out_comp]), int(component_dims[in_comp])),
                dtype=complex,
            )
            blocks[key] = block
        kernel = term.kernel_matrix(
            term.input_entry.shape,
            max_elements=max(
                int(term.input_entry.size) * int(term.output_entry.size),
                1,
            ),
        )
        if kernel is None:
            return None
        block[out_slice, in_slice] += np.asarray(kernel, dtype=complex)
    return tuple(
        (int(key[0]), int(key[1]), np.ascontiguousarray(block))
        for key, block in sorted(blocks.items())
    )


def project_component_orthonormal_blocks(
    object parent_blocks,
    object transforms,
    long max_elements,
):
    """
    Project parent component blocks into orthonormal component coordinates.
    """

    cdef object entry
    cdef object X_in
    cdef object X_out
    cdef object parent_block
    cdef object transformed
    cdef list out = []
    cdef long total_elements = 0
    cdef int in_comp
    cdef int out_comp

    if parent_blocks is None:
        return None
    for entry in parent_blocks:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        total_elements += (
            int(transforms[out_comp].shape[1])
            * int(transforms[in_comp].shape[1])
        )
        if total_elements > max_elements:
            return None
    for entry in parent_blocks:
        in_comp = int(entry[0])
        out_comp = int(entry[1])
        parent_block = entry[2]
        X_in = np.asarray(transforms[in_comp], dtype=complex)
        X_out = np.asarray(transforms[out_comp], dtype=complex)
        transformed = X_out.conj().T @ np.asarray(parent_block, dtype=complex) @ X_in
        if np.linalg.norm(transformed.reshape(-1)) > 1.0e-15:
            out.append((in_comp, out_comp, np.ascontiguousarray(transformed)))
    return tuple(out)


def apply_packed_qchem_factor_routes(
    in_indices,
    out_indices,
    left_indices,
    right_indices,
    basis_offsets,
    basis_shapes,
    left_factor_indices,
    left_offsets,
    left_shape_offsets,
    left_shapes,
    left_data,
    right_factor_indices,
    right_offsets,
    right_shape_offsets,
    right_shapes,
    right_data,
    vector,
    total_dim,
):
    """Apply packed real SU(2) factor routes with bounded C++ workspace."""

    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_arr = np.ascontiguousarray(in_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_arr = np.ascontiguousarray(out_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] li_arr = np.ascontiguousarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ri_arr = np.ascontiguousarray(right_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] bo_arr = np.ascontiguousarray(basis_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] bs_arr = np.ascontiguousarray(basis_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lfi_arr = np.ascontiguousarray(left_factor_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lo_arr = np.ascontiguousarray(left_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lso_arr = np.ascontiguousarray(left_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ls_arr = np.ascontiguousarray(left_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ld_arr = np.ascontiguousarray(left_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rfi_arr = np.ascontiguousarray(right_factor_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ro_arr = np.ascontiguousarray(right_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rso_arr = np.ascontiguousarray(right_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rs_arr = np.ascontiguousarray(right_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rd_arr = np.ascontiguousarray(right_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] vec_arr = np.ascontiguousarray(vector, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] result = np.zeros(int(total_dim), dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] tmp
    cdef Py_ssize_t n_routes = in_arr.shape[0]
    cdef Py_ssize_t route, lentry, rentry, lpool, rpool, lshape, rshape
    cdef Py_ssize_t L, K, W, A, B, Q, R, D, C
    cdef Py_ssize_t l, k, w, a, b, q, r, d, c
    cdef Py_ssize_t in_idx, out_idx, in_offset, out_offset
    cdef Py_ssize_t left_offset, right_offset, tmp_idx, input_pos, output_pos
    cdef Py_ssize_t left_pos, right_pos, max_tmp = 1
    cdef double complex acc

    if not (
        out_arr.shape[0] == n_routes
        and li_arr.shape[0] == n_routes
        and ri_arr.shape[0] == n_routes
        and bs_arr.shape[1] == 4
    ):
        raise ValueError("Packed SU2 factor-route arrays have incompatible shapes.")
    if np.iscomplexobj(left_data) or np.iscomplexobj(right_data):
        raise TypeError("C++ packed SU2 factor routes currently require real factors.")
    for route in range(n_routes):
        lentry = li_arr[route]
        rentry = ri_arr[route]
        lpool = lfi_arr[lentry]
        rpool = rfi_arr[rentry]
        lshape = lso_arr[lpool]
        rshape = rso_arr[rpool]
        if lso_arr[lpool + 1] - lshape != 5 or rso_arr[rpool + 1] - rshape != 5:
            raise ValueError("Packed SU2 factor routes require rank-5 factor entries.")
        L = ls_arr[lshape]
        W = ls_arr[lshape + 2]
        A = ls_arr[lshape + 3]
        in_idx = in_arr[route]
        C = bs_arr[in_idx, 2]
        R = bs_arr[in_idx, 3]
        max_tmp = max(max_tmp, L * W * A * C * R)
    tmp = np.empty(max_tmp, dtype=np.complex128)

    for route in range(n_routes):
        in_idx = in_arr[route]
        out_idx = out_arr[route]
        lentry = li_arr[route]
        rentry = ri_arr[route]
        lpool = lfi_arr[lentry]
        rpool = rfi_arr[rentry]
        lshape = lso_arr[lpool]
        rshape = rso_arr[rpool]
        L = ls_arr[lshape]
        K = ls_arr[lshape + 1]
        W = ls_arr[lshape + 2]
        A = ls_arr[lshape + 3]
        B = ls_arr[lshape + 4]
        Q = rs_arr[rshape + 1]
        R = rs_arr[rshape + 2]
        D = rs_arr[rshape + 3]
        C = rs_arr[rshape + 4]
        in_offset = bo_arr[in_idx]
        out_offset = bo_arr[out_idx]
        left_offset = lo_arr[lpool]
        right_offset = ro_arr[rpool]
        for l in range(L):
            for w in range(W):
                for a in range(A):
                    for c in range(C):
                        for r in range(R):
                            acc = 0.0
                            for k in range(K):
                                for b in range(B):
                                    left_pos = left_offset + ((((l * K + k) * W + w) * A + a) * B + b)
                                    input_pos = in_offset + (((k * B + b) * C + c) * R + r)
                                    acc += ld_arr[left_pos] * vec_arr[input_pos]
                            tmp_idx = ((((l * W + w) * A + a) * C + c) * R + r)
                            tmp[tmp_idx] = acc
        for l in range(L):
            for a in range(A):
                for d in range(D):
                    for q in range(Q):
                        acc = 0.0
                        for w in range(W):
                            for r in range(R):
                                for c in range(C):
                                    tmp_idx = ((((l * W + w) * A + a) * C + c) * R + r)
                                    right_pos = right_offset + ((((w * Q + q) * R + r) * D + d) * C + c)
                                    acc += tmp[tmp_idx] * rd_arr[right_pos]
                        output_pos = out_offset + (((l * A + a) * D + d) * Q + q)
                        result[output_pos] += acc
    return result


def diagonal_packed_qchem_factor_routes(
    in_indices,
    out_indices,
    left_indices,
    right_indices,
    basis_offsets,
    basis_shapes,
    left_factor_indices,
    left_offsets,
    left_shape_offsets,
    left_shapes,
    left_data,
    right_factor_indices,
    right_offsets,
    right_shape_offsets,
    right_shapes,
    right_data,
    total_dim,
):
    """Build only the diagonal of packed real SU(2) factor routes."""

    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_arr = np.ascontiguousarray(in_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_arr = np.ascontiguousarray(out_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] li_arr = np.ascontiguousarray(left_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ri_arr = np.ascontiguousarray(right_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] bo_arr = np.ascontiguousarray(basis_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2] bs_arr = np.ascontiguousarray(basis_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lfi_arr = np.ascontiguousarray(left_factor_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lo_arr = np.ascontiguousarray(left_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lso_arr = np.ascontiguousarray(left_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ls_arr = np.ascontiguousarray(left_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ld_arr = np.ascontiguousarray(left_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rfi_arr = np.ascontiguousarray(right_factor_indices, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ro_arr = np.ascontiguousarray(right_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rso_arr = np.ascontiguousarray(right_shape_offsets, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] rs_arr = np.ascontiguousarray(right_shapes, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rd_arr = np.ascontiguousarray(right_data, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.zeros(int(total_dim), dtype=np.float64)
    cdef Py_ssize_t route, entry, lentry, rentry, lpool, rpool, lshape, rshape
    cdef Py_ssize_t K, B, C, R, W, k, b, c, r, w
    cdef Py_ssize_t offset, left_offset, right_offset, left_pos, right_pos, pos
    cdef double acc

    if np.iscomplexobj(left_data) or np.iscomplexobj(right_data):
        raise TypeError("C++ packed SU2 factor diagonals currently require real factors.")
    for route in range(in_arr.shape[0]):
        if in_arr[route] != out_arr[route]:
            continue
        entry = in_arr[route]
        lentry = li_arr[route]
        rentry = ri_arr[route]
        lpool = lfi_arr[lentry]
        rpool = rfi_arr[rentry]
        lshape = lso_arr[lpool]
        rshape = rso_arr[rpool]
        K = bs_arr[entry, 0]
        B = bs_arr[entry, 1]
        C = bs_arr[entry, 2]
        R = bs_arr[entry, 3]
        if not (
            ls_arr[lshape] == K
            and ls_arr[lshape + 1] == K
            and ls_arr[lshape + 3] == B
            and ls_arr[lshape + 4] == B
            and rs_arr[rshape + 1] == R
            and rs_arr[rshape + 2] == R
            and rs_arr[rshape + 3] == C
            and rs_arr[rshape + 4] == C
        ):
            raise ValueError("Packed SU2 diagonal route has incompatible sector shapes.")
        W = ls_arr[lshape + 2]
        if rs_arr[rshape] != W:
            raise ValueError("Packed SU2 diagonal route has incompatible MPO dimensions.")
        offset = bo_arr[entry]
        left_offset = lo_arr[lpool]
        right_offset = ro_arr[rpool]
        for k in range(K):
            for b in range(B):
                for c in range(C):
                    for r in range(R):
                        acc = 0.0
                        for w in range(W):
                            left_pos = left_offset + ((((k * K + k) * W + w) * B + b) * B + b)
                            right_pos = right_offset + ((((w * R + r) * R + r) * C + c) * C + c)
                            acc += ld_arr[left_pos] * rd_arr[right_pos]
                        pos = offset + (((k * B + b) * C + c) * R + r)
                        result[pos] += acc
    return result
