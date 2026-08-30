"""Moving-environment ownership, caching, and compiled execution."""

from ._mps_common import *
from . import _abelian_local_engine as _abelian_local_engine
from ._abelian_local_engine import *
from ._mps_state import (
    DenseLocalProblem,
    coarse_grain_MPO,
    coarse_grain_MPS,
    contract_from_left,
    contract_from_right,
)


class MovingEnvironmentCompiledBackend:
    """Compiled/table backend state owned by :class:`MovingEnvironment`."""

    def __init__(self, environment):
        self.environment = environment

    def use_cpp_raw_grouped_renormalized_table(self):
        grouped_enabled_opt = MovingEnvironment._option_value(
            self.environment.matvec_options,
            "moving_environment_cpp_grouped_renormalized_table",
            None,
        )
        if grouped_enabled_opt is None:
            grouped_enabled = bool(
                MovingEnvironment._option_value(
                    self.environment.matvec_options,
                    "moving_environment_cpp_davidson",
                    False,
                )
                or MovingEnvironment._option_value(
                    self.environment.matvec_options,
                    "moving_environment_cpp_matvec",
                    False,
                )
            )
        else:
            grouped_enabled = bool(grouped_enabled_opt)
        if not grouped_enabled:
            return False
        if (
            _cpp_davidson is None
            or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        ):
            return False
        dense_cls = getattr(_cpp_davidson, "GroupedRenormalizedTable", None)
        if dense_cls is None or getattr(dense_cls, "from_raw", None) is None:
            return False
        if bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_grouped_factorized_table",
                False,
            )
        ):
            return False
        return bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_grouped_raw_table",
                True,
            )
        )

    def use_cpp_raw_payload_builder(self):
        if not self.use_cpp_raw_grouped_renormalized_table():
            return False
        if (
            _cpp_davidson is None
            or getattr(_cpp_davidson, "RawPayloadBuilder", None) is None
        ):
            return False
        dense_cls = getattr(_cpp_davidson, "GroupedRenormalizedTable", None)
        if dense_cls is None or getattr(dense_cls, "from_raw_builder", None) is None:
            return False
        return bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_raw_payload_builder",
                True,
            )
        )

    def use_cpp_raw_route_plan(self):
        backend_mode = str(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "generator_table_packed_route_table",
                "auto",
            )
        ).strip().lower()
        stats = self.environment.moving_profile_stats
        if backend_mode in {"off", "false", "0", "none"}:
            stats["cpp_raw_route_plan_backend_actual"] = "off"
            stats["cpp_raw_route_plan_fallback_reason"] = "disabled"
            return False
        if backend_mode in {"python", "reference"}:
            stats["cpp_raw_route_plan_backend_actual"] = "python"
            stats["cpp_raw_route_plan_fallback_reason"] = "requested_python"
            return False
        if backend_mode == "auto":
            stats["cpp_raw_route_plan_backend_actual"] = "python"
            stats["cpp_raw_route_plan_fallback_reason"] = (
                "auto_direct_builder_pending_validation"
            )
            return False
        if backend_mode in {"route_plan", "route-plan", "raw_route_plan", "raw-route-plan"}:
            backend_mode = "route_plan"
        else:
            stats["cpp_raw_route_plan_backend_actual"] = "off"
            stats["cpp_raw_route_plan_fallback_reason"] = "route_plan_not_requested"
            return False
        if not self.use_cpp_raw_payload_builder():
            if backend_mode == "cython":
                raise RuntimeError(
                    "generator_table_packed_route_table='cython' requires the "
                    "compiled grouped raw payload backend"
                )
            stats["cpp_raw_route_plan_backend_actual"] = "python"
            stats["cpp_raw_route_plan_fallback_reason"] = (
                "compiled_raw_payload_unavailable"
            )
            return False
        if (
            _cpp_davidson is None
            or getattr(_cpp_davidson, "RawRoutePlan", None) is None
        ):
            if backend_mode == "cython":
                raise RuntimeError(
                    "generator_table_packed_route_table='cython' requires "
                    "cpp_davidson.RawRoutePlan"
                )
            stats["cpp_raw_route_plan_backend_actual"] = "python"
            stats["cpp_raw_route_plan_fallback_reason"] = "raw_route_plan_unavailable"
            return False
        if not bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_raw_payload_stack_kernels",
                True,
            )
        ):
            if backend_mode == "cython":
                raise RuntimeError(
                    "generator_table_packed_route_table='cython' requires "
                    "moving_environment_cpp_raw_payload_stack_kernels=True"
                )
            stats["cpp_raw_route_plan_backend_actual"] = "python"
            stats["cpp_raw_route_plan_fallback_reason"] = "stack_kernels_disabled"
            return False
        enabled = bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_raw_route_plan",
                True,
            )
        )
        if not enabled:
            if backend_mode == "cython":
                raise RuntimeError(
                    "generator_table_packed_route_table='cython' requires "
                    "moving_environment_cpp_raw_route_plan=True"
                )
            stats["cpp_raw_route_plan_backend_actual"] = "python"
            stats["cpp_raw_route_plan_fallback_reason"] = "raw_route_plan_disabled"
            return False
        stats["cpp_raw_route_plan_backend_actual"] = "cython"
        stats["cpp_raw_route_plan_fallback_reason"] = ""
        return True

    def use_cpp_named_raw_payload_builder(self):
        if not self.use_cpp_raw_payload_builder():
            return False
        route_cls = None if _cpp_davidson is None else getattr(
            _cpp_davidson,
            "RawRoutePlan",
            None,
        )
        if route_cls is None or getattr(route_cls, "build_named_family_payload", None) is None:
            return False
        return bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_named_raw_payload_builder",
                True,
            )
        )

    def use_cpp_named_raw_payload_plan(self):
        if not self.use_cpp_named_raw_payload_builder():
            return False
        plan_cls = None if _cpp_davidson is None else getattr(
            _cpp_davidson,
            "NamedRawPayloadPlan",
            None,
        )
        if plan_cls is None or getattr(plan_cls, "build_payload", None) is None:
            return False
        return bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_named_raw_payload_plan",
                True,
            )
        )

    def named_raw_payload_plan_cache_key(self, operator, proto, layout):
        environments = getattr(operator, "complementary_family_environments", None)
        if not environments:
            return None
        return (
            "moving_environment_cpp_named_raw_payload_plan",
            None if operator.bond is None else int(operator.bond),
            tuple(proto.dirs),
            tuple(environments.keys()),
        )

    def raw_route_plan_cache_key(self, operator, proto, layout):
        if not self.use_cpp_raw_route_plan():
            return None
        if (
            not getattr(operator, "complementary_direct_family_environments", None)
            and self.use_cpp_named_raw_payload_builder()
        ):
            return None
        if getattr(operator, "complementary_direct_family_environments", None):
            if bool(
                MovingEnvironment._option_value(
                    self.environment.matvec_options,
                    "moving_environment_cpp_raw_route_plan_rebind_layout",
                    True,
                )
            ):
                layout_token = tuple(key for key, _shape in layout)
                key_kind = "direct_generator_ref_layout_rebind"
            else:
                layout_token = tuple(layout)
                key_kind = "direct_generator_ref"
            return (
                "moving_environment_cpp_raw_route_plan",
                key_kind,
                None if operator.bond is None else int(operator.bond),
                layout_token,
                tuple(proto.dirs),
                self._direct_family_environment_route_signature(
                    getattr(
                        operator,
                        "complementary_direct_family_environments",
                        None,
                    )
                    or {}
                ),
                self._family_environment_key_signature(
                    getattr(operator, "complementary_family_environments", None)
                    or {}
                ),
            )
        route_cls = getattr(_cpp_davidson, "RawRoutePlan", None)
        if route_cls is None:
            return None
        environments = getattr(operator, "complementary_family_environments", None)
        if not environments:
            return None
        if bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_raw_route_plan_rebind_layout",
                True,
            )
        ):
            signature = self._family_environment_key_signature(environments)
            layout_token = tuple(key for key, _shape in layout)
            key_kind = "named_layout_rebind"
        else:
            signature = route_cls.family_environment_signature(environments)
            layout_token = tuple(layout)
            key_kind = "named_exact_layout"
        return (
            "moving_environment_cpp_raw_route_plan",
            key_kind,
            None if operator.bond is None else int(operator.bond),
            layout_token,
            tuple(proto.dirs),
            signature,
        )

    def _route_plan_family_names(self, route_plan):
        family_names = []
        seen = set()
        try:
            names = route_plan.family_names()
        except Exception:
            return family_names
        for name in names:
            text = str(name)
            if text in seen:
                continue
            seen.add(text)
            family_names.append(text)
        return family_names

    def _maybe_coalesce_raw_builder(self, builder, phase):
        enabled = bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_raw_payload_coalesce_exact",
                False,
            )
        )
        if not enabled or builder is None:
            return builder
        coalesce = getattr(builder, "coalesce_exact", None)
        stats = self.environment.moving_profile_stats
        if not callable(coalesce):
            stats["cpp_raw_payload_coalesce_exact_unavailable"] = int(
                stats.get("cpp_raw_payload_coalesce_exact_unavailable", 0)
            ) + 1
            return builder
        try:
            result = dict(coalesce())
        except Exception as exc:
            stats["cpp_raw_payload_coalesce_exact_failures"] = int(
                stats.get("cpp_raw_payload_coalesce_exact_failures", 0)
            ) + 1
            stats["cpp_raw_payload_coalesce_exact_last_error"] = repr(exc)
            return builder
        stats["cpp_raw_payload_coalesce_exact_calls"] = int(
            stats.get("cpp_raw_payload_coalesce_exact_calls", 0)
        ) + 1
        stats["cpp_raw_payload_coalesce_exact_seconds"] = float(
            stats.get("cpp_raw_payload_coalesce_exact_seconds", 0.0)
        ) + float(result.get("seconds", 0.0) or 0.0)
        stats["cpp_raw_payload_coalesce_exact_before"] = int(
            stats.get("cpp_raw_payload_coalesce_exact_before", 0)
        ) + int(result.get("before", 0) or 0)
        stats["cpp_raw_payload_coalesce_exact_after"] = int(
            stats.get("cpp_raw_payload_coalesce_exact_after", 0)
        ) + int(result.get("after", 0) or 0)
        stats["cpp_raw_payload_coalesce_exact_reduction"] = int(
            stats.get("cpp_raw_payload_coalesce_exact_reduction", 0)
        ) + int(result.get("reduction", 0) or 0)
        stats["cpp_raw_payload_coalesce_exact_cancelled"] = int(
            stats.get("cpp_raw_payload_coalesce_exact_cancelled", 0)
        ) + int(result.get("cancelled", 0) or 0)
        stats["cpp_raw_payload_coalesce_exact_last"] = {
            "phase": str(phase),
            "before": int(result.get("before", 0) or 0),
            "after": int(result.get("after", 0) or 0),
            "reduction": int(result.get("reduction", 0) or 0),
            "cancelled": int(result.get("cancelled", 0) or 0),
            "seconds": float(result.get("seconds", 0.0) or 0.0),
        }
        return builder

    @staticmethod
    def _raw_builder_payload_record(
        builder,
        *,
        entry_count=None,
        family_names=(),
        raw_route_plan=None,
    ):
        if entry_count is None:
            entry_count = int(builder.size())
        return {
            "raw_builder": builder,
            "left": [],
            "right": [],
            "dims": [],
            "in_starts": [],
            "out_starts": [],
            "scales": [],
            "entry_count": int(entry_count),
            "family_names": tuple(str(name) for name in family_names),
            "matvec_groups": None,
            "raw_route_plan": raw_route_plan,
        }

    @staticmethod
    def _cpp_grouped_table_payload_record(
        backend,
        *,
        entry_count,
        family_names=(),
    ):
        return {
            "cpp_grouped_table": backend,
            "raw_builder": None,
            "left": [],
            "right": [],
            "dims": [],
            "in_starts": [],
            "out_starts": [],
            "scales": [],
            "entry_count": int(entry_count),
            "family_names": tuple(str(name) for name in family_names),
            "matvec_groups": None,
            "raw_route_plan": None,
        }

    @staticmethod
    def _tensor_key_signature(tensor):
        data = getattr(tensor, "data", {}) or {}
        try:
            keys = tuple(data.keys())
        except Exception:
            return (id(tensor),)
        return tuple(sorted(keys, key=repr))

    def _family_environment_key_signature(self, environments):
        tokens = []
        for name, env in sorted((environments or {}).items(), key=lambda item: str(item[0])):
            try:
                E, W, F = env
                W0, W1 = W[0], W[1]
            except Exception:
                tokens.append((str(name), id(env)))
                continue
            tokens.append(
                (
                    str(name),
                    self._tensor_key_signature(E),
                    self._tensor_key_signature(W0),
                    self._tensor_key_signature(W1),
                    self._tensor_key_signature(F),
                )
            )
        return tuple(tokens)

    @staticmethod
    def _tensor_block_shape_signature(tensor):
        if bool(getattr(tensor, "_pyqed_packed_boundary_tensor", False)):
            try:
                return tuple(
                    (repr(key), tuple(int(v) for v in np.asarray(block).shape))
                    for key, block in zip(tensor.keys, tensor.blocks)
                )
            except Exception:
                return (id(tensor),)
        data = getattr(tensor, "data", {}) or {}
        items = []
        try:
            iterator = data.items()
        except Exception:
            return (id(tensor),)
        for key, block in iterator:
            try:
                shape = tuple(int(v) for v in np.asarray(block).shape)
            except Exception:
                shape = ()
            items.append((repr(key), shape))
        return tuple(sorted(items, key=lambda item: item[0]))

    def _direct_component_route_signature(self, component):
        if isinstance(component, AbelianPackedIdentityLocalEntry):
            return (
                "packed_identity",
                str(getattr(component, "source", "")),
                self._tensor_block_shape_signature(getattr(component, "E", None)),
                self._tensor_block_shape_signature(getattr(component, "F", None)),
            )
        if isinstance(component, AbelianPackedLocalGeneratorEntry):
            return (
                "packed_local_generator",
                str(getattr(component, "source", "")),
                self._tensor_block_shape_signature(getattr(component, "E", None)),
                self._tensor_block_shape_signature(getattr(component, "W_left", None)),
                self._tensor_block_shape_signature(getattr(component, "W_right", None)),
                self._tensor_block_shape_signature(getattr(component, "F", None)),
            )
        try:
            E, W, F = component
            return (
                "direct_component",
                self._tensor_block_shape_signature(E),
                self._tensor_block_shape_signature(W[0]),
                self._tensor_block_shape_signature(W[1]),
                self._tensor_block_shape_signature(F),
            )
        except Exception:
            return ("unknown", type(component).__name__)

    def _direct_family_environment_route_signature(self, direct_environments):
        tokens = []
        for name, entries in sorted(
            (direct_environments or {}).items(),
            key=lambda item: str(item[0]),
        ):
            entry_groups = tuple(getattr(entries, "entry_groups", ()) or ())
            group_sizes = tuple(int(len(group)) for group in entry_groups)
            group_keys = tuple(
                repr(key) for key in (getattr(entries, "group_keys", ()) or ())
            )
            tokens.append(
                (
                    str(name),
                    int(len(entries)),
                    group_sizes,
                    group_keys,
                )
            )
        return tuple(tokens)

    def collect_renormalized_operator_payload_from_route_plan(
        self,
        operator,
        route_plan,
        proto,
        layout,
    ):
        start = time.perf_counter()
        try:
            family_environments = (
                getattr(operator, "complementary_family_environments", None) or {}
            )
            if getattr(operator, "complementary_direct_family_environments", None):
                direct_environments = (
                    getattr(
                        operator,
                        "complementary_direct_family_environments",
                        None,
                    )
                    or {}
                )
                build_from_environments = getattr(
                    route_plan,
                    "build_from_environments",
                    None,
                )
                if build_from_environments is None:
                    builder = route_plan.build_from_sources(
                        family_environments,
                        True,
                    )
                else:
                    try:
                        builder = build_from_environments(
                            family_environments,
                            direct_environments,
                            True,
                            tuple(layout),
                        )
                    except TypeError:
                        builder = build_from_environments(
                            family_environments,
                            direct_environments,
                            True,
                        )
            else:
                try:
                    builder = route_plan.build_from_sources(
                        family_environments,
                        True,
                        tuple(layout),
                    )
                except TypeError:
                    builder = route_plan.build_from_sources(
                        family_environments,
                        True,
                    )
            builder = self._maybe_coalesce_raw_builder(builder, "route_plan")
            entry_count = int(builder.size())
            if entry_count <= 0:
                return None
            return self._raw_builder_payload_record(
                builder,
                entry_count=entry_count,
                family_names=self._route_plan_family_names(route_plan),
                raw_route_plan=route_plan,
            )
        finally:
            elapsed = float(time.perf_counter() - start)
            stats = self.environment.moving_profile_stats
            stats["cpp_raw_route_plan_refresh_calls"] = int(
                stats.get("cpp_raw_route_plan_refresh_calls", 0)
            ) + 1
            stats["cpp_raw_route_plan_refresh_seconds"] = float(
                stats.get("cpp_raw_route_plan_refresh_seconds", 0.0)
            ) + elapsed
            stats["cpp_raw_route_plan_refresh_last_seconds"] = elapsed

    def collect_direct_renormalized_operator_payload_cpp(self, operator, proto, layout):
        def _legacy_direct_family_entries(entries_obj):
            packed = (
                entries_obj
                if bool(getattr(entries_obj, "_pyqed_packed_direct_family_entries", False))
                else getattr(entries_obj, "entries", None)
            )
            if bool(getattr(packed, "_pyqed_packed_direct_family_entries", False)):
                if bool(
                    getattr(
                        packed,
                        "_pyqed_composite_direct_family_entries",
                        False,
                    )
                ):
                    materialized = AbelianPackedDirectFamilyEntries()
                    materialized.extend(packed)
                    packed = materialized
                cloned = AbelianPackedDirectFamilyEntries()
                cloned.extend_identity(
                    tuple(packed.identity_coeffs),
                    tuple(
                        unpack_abelian_packed_boundary_tensor(tensor)
                        for tensor in packed.identity_E
                    ),
                    tuple(
                        unpack_abelian_packed_boundary_tensor(tensor)
                        for tensor in packed.identity_F
                    ),
                    source="packed_boundary_legacy_identity",
                )
                identity_sources = list(getattr(packed, "identity_sources", ()))
                if len(identity_sources) == len(cloned.identity_coeffs):
                    cloned.identity_sources = identity_sources
                cloned.extend_local_generators(
                    tuple(packed.local_coeffs),
                    tuple(
                        unpack_abelian_packed_boundary_tensor(tensor)
                        for tensor in packed.local_E
                    ),
                    tuple(
                        unpack_abelian_packed_boundary_tensor(tensor)
                        for tensor in packed.local_W_left
                    ),
                    tuple(
                        unpack_abelian_packed_boundary_tensor(tensor)
                        for tensor in packed.local_W_right
                    ),
                    tuple(
                        unpack_abelian_packed_boundary_tensor(tensor)
                        for tensor in packed.local_F
                    ),
                    source="packed_boundary_legacy_local",
                )
                local_sources = list(getattr(packed, "local_sources", ()))
                if len(local_sources) == len(cloned.local_coeffs):
                    cloned.local_sources = local_sources
                changed = any(
                    bool(getattr(tensor, "_pyqed_packed_boundary_tensor", False))
                    for tensors in (
                        packed.identity_E,
                        packed.identity_F,
                        packed.local_E,
                        packed.local_W_left,
                        packed.local_W_right,
                        packed.local_F,
                    )
                    for tensor in tensors
                )
                return cloned, changed
            try:
                seq = tuple(entries_obj or ())
            except TypeError:
                return entries_obj, False
            cloned_entries = []
            changed = False
            for entry in seq:
                if isinstance(entry, AbelianPackedIdentityLocalEntry):
                    E = unpack_abelian_packed_boundary_tensor(entry.E)
                    F = unpack_abelian_packed_boundary_tensor(entry.F)
                    changed = changed or E is not entry.E or F is not entry.F
                    cloned_entries.append(
                        AbelianPackedIdentityLocalEntry(
                            entry.coeff,
                            E,
                            F,
                            source=entry.source,
                        )
                    )
                elif isinstance(entry, AbelianPackedLocalGeneratorEntry):
                    E = unpack_abelian_packed_boundary_tensor(entry.E)
                    W_left = unpack_abelian_packed_boundary_tensor(entry.W_left)
                    W_right = unpack_abelian_packed_boundary_tensor(entry.W_right)
                    F = unpack_abelian_packed_boundary_tensor(entry.F)
                    changed = changed or any(
                        left is not right
                        for left, right in (
                            (E, entry.E),
                            (W_left, entry.W_left),
                            (W_right, entry.W_right),
                            (F, entry.F),
                        )
                    )
                    cloned_entries.append(
                        AbelianPackedLocalGeneratorEntry(
                            entry.coeff,
                            E,
                            W_left,
                            W_right,
                            F,
                            source=entry.source,
                        )
                    )
                else:
                    cloned_entries.append(entry)
            return tuple(cloned_entries), changed

        def _legacy_direct_family_environments(environments):
            cloned = {}
            changed = False
            for family_name, entries in (environments or {}).items():
                cloned_entries, entry_changed = _legacy_direct_family_entries(entries)
                cloned[family_name] = cloned_entries
                changed = changed or entry_changed
            return cloned, changed

        def _packed_boundary_legacy_compare_supported(entries_obj):
            packed = (
                entries_obj
                if bool(getattr(entries_obj, "_pyqed_packed_direct_family_entries", False))
                else getattr(entries_obj, "entries", None)
            )
            if not bool(getattr(packed, "_pyqed_packed_direct_family_entries", False)):
                return True
            if bool(getattr(packed, "_pyqed_planned_direct_family_entries", False)):
                return False
            if bool(getattr(packed, "_pyqed_same_side_route_identity_entries", False)):
                return False
            if bool(getattr(packed, "_pyqed_composite_direct_family_entries", False)):
                return all(
                    _packed_boundary_legacy_compare_supported(part)
                    for part in tuple(getattr(packed, "parts", ()) or ())
                )
            return True

        def _legacy_boundary_compare_supported(environments):
            return all(
                _packed_boundary_legacy_compare_supported(entries)
                for entries in (environments or {}).values()
            )

        mode = str(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "generator_table_packed_route_table",
                "auto",
            )
        ).strip().lower()
        if mode in {"off", "false", "0", "none", "python", "reference"}:
            stats = self.environment.moving_profile_stats
            stats["packed_route_table_backend_actual"] = (
                "off" if mode in {"off", "false", "0", "none"} else "python"
            )
            stats["packed_route_table_fallback_reason"] = (
                "disabled" if mode in {"off", "false", "0", "none"} else "requested_python"
            )
            return None
        if mode in {"route_plan", "route-plan", "raw_route_plan", "raw-route-plan"}:
            stats = self.environment.moving_profile_stats
            stats["packed_route_table_backend_actual"] = "route_plan"
            stats["packed_route_table_fallback_reason"] = "build_or_refresh_route_plan"
            return None
        if mode in {
            "auto",
            "cython",
            "cpp",
            "native",
            "packed",
            "raw_cython",
            "raw-cython",
            "raw_cpp",
            "raw-cpp",
            "raw_cython_validate",
            "raw-cython-validate",
        }:
            validate_raw = mode in {"raw_cython_validate", "raw-cython-validate"}
            if not self.use_cpp_raw_payload_builder():
                if mode == "auto":
                    stats = self.environment.moving_profile_stats
                    stats["packed_route_table_backend_actual"] = "python"
                    stats["packed_route_table_fallback_reason"] = (
                        "compiled_raw_payload_unavailable"
                    )
                    return None
                if mode == "cython":
                    raise RuntimeError(
                        "generator_table_packed_route_table='cython' requires "
                        "the compiled grouped raw payload backend"
                    )
                return None
            fast_builder_fn = (
                None
                if _cpp_davidson is None
                else getattr(
                    _cpp_davidson,
                    "build_direct_family_payload_fastkeys",
                    None,
                )
            )
            legacy_builder_fn = (
                None
                if _cpp_davidson is None
                else getattr(_cpp_davidson, "build_direct_family_payload", None)
            )
            direct_builder_fn = fast_builder_fn or legacy_builder_fn
            route_cls = None if _cpp_davidson is None else getattr(
                _cpp_davidson,
                "RawRoutePlan",
                None,
            )
            use_planned_direct_payload = bool(
                MovingEnvironment._option_value(
                    self.environment.matvec_options,
                    "generator_table_use_planned_direct_payload",
                    True,
                )
            )
            planned_plan_cls = (
                None
                if _cpp_davidson is None or not use_planned_direct_payload
                else getattr(
                    _cpp_davidson,
                    "PlannedDirectPayloadPlan",
                    None,
                )
            )
            if direct_builder_fn is None or (
                mode in {"auto", "cython", "cpp", "native", "packed"}
                and fast_builder_fn is None
            ):
                if mode == "auto":
                    stats = self.environment.moving_profile_stats
                    stats["packed_route_table_backend_actual"] = "python"
                    stats["packed_route_table_fallback_reason"] = (
                        "raw_cython_builder_unavailable"
                    )
                    return None
                raise RuntimeError(
                    "generator_table_packed_route_table='cython' requires "
                    "build_direct_family_payload"
                )
            start = time.perf_counter()
            stats = self.environment.moving_profile_stats
            try:
                direct_environments = (
                    getattr(
                        operator,
                        "complementary_direct_family_environments",
                        None,
                    )
                    or {}
                )
                layout_tuple = tuple(layout)
                proto_data = getattr(proto, "data", {}) or {}
                planned_plan_used = False
                planned_plan_error = ""
                if direct_environments and planned_plan_cls is not None:
                    planned_plan = getattr(
                        self.environment,
                        "_pyqed_planned_direct_payload_plan",
                        None,
                    )
                    if planned_plan is None:
                        planned_plan = planned_plan_cls()
                        setattr(
                            self.environment,
                            "_pyqed_planned_direct_payload_plan",
                            planned_plan,
                        )
                    try:
                        builder = planned_plan.build_payload(
                            direct_environments,
                            proto_data,
                            layout_tuple,
                            True,
                        )
                        planned_plan_used = True
                        try:
                            stats["packed_planned_route_cpp_last_stats"] = dict(
                                planned_plan.stats()
                            )
                        except Exception:
                            pass
                    except Exception as planned_exc:
                        planned_plan_error = str(planned_exc)
                        stats["packed_planned_route_cpp_failures"] = int(
                            stats.get("packed_planned_route_cpp_failures", 0)
                        ) + 1
                        stats["packed_planned_route_cpp_fallback_reason"] = (
                            planned_plan_error
                        )
                        builder = direct_builder_fn(
                            direct_environments,
                            proto_data,
                            layout_tuple,
                            True,
                        )
                else:
                    builder = direct_builder_fn(
                        direct_environments,
                        proto_data,
                        layout_tuple,
                        True,
                    )
                stats["packed_planned_route_cpp_used"] = bool(planned_plan_used)
                if planned_plan_used:
                    stats["packed_planned_route_cpp_fallback_reason"] = ""
                elif not use_planned_direct_payload:
                    stats["packed_planned_route_cpp_fallback_reason"] = "disabled"
                elif planned_plan_error:
                    stats["packed_planned_route_cpp_fallback_reason"] = (
                        planned_plan_error
                    )
                family_names = [str(name) for name in direct_environments]
                family_environments = (
                    getattr(operator, "complementary_family_environments", None) or {}
                )
                if family_environments:
                    if route_cls is None:
                        raise RuntimeError("RawRoutePlan is unavailable for named payloads")
                    named_builder = route_cls.build_named_family_payload(
                        family_environments,
                        proto_data,
                        layout_tuple,
                        True,
                    )
                    if int(named_builder.size()) > 0:
                        builder.extend(named_builder)
                        family_names.extend(str(name) for name in family_environments)
                builder = self._maybe_coalesce_raw_builder(builder, "direct")
                entry_count = int(builder.size())
                if entry_count <= 0:
                    return None
                try:
                    builder_stats_fn = getattr(builder, "stats", None)
                    builder_stats = (
                        dict(builder_stats_fn())
                        if callable(builder_stats_fn)
                        else {}
                    )
                except Exception:
                    builder_stats = {}
                if builder_stats:
                    total_stats = stats.setdefault("packed_route_cpp_stats", {})
                    for key, value in builder_stats.items():
                        if isinstance(value, (int, np.integer)):
                            total_stats[key] = int(total_stats.get(key, 0)) + int(value)
                        elif isinstance(value, (float, np.floating)):
                            total_stats[key] = (
                                float(total_stats.get(key, 0.0)) + float(value)
                            )
                    stats["packed_route_cpp_last_stats"] = builder_stats
                if validate_raw:
                    dense_cls = None if _cpp_davidson is None else getattr(
                        _cpp_davidson,
                        "GroupedRenormalizedTable",
                        None,
                    )
                    if dense_cls is None:
                        raise RuntimeError(
                            "raw_cython_validate requires GroupedRenormalizedTable"
                        )
                    legacy_envs, legacy_changed = _legacy_direct_family_environments(
                        direct_environments
                    )
                    if legacy_changed and _legacy_boundary_compare_supported(
                        direct_environments
                    ):
                        legacy_builder = direct_builder_fn(
                            legacy_envs,
                            getattr(proto, "data", {}) or {},
                            tuple(layout),
                            True,
                        )
                        dim = int(operator._size(layout))
                        threshold = float(
                            operator._renormalized_operator_table_sparse_density_threshold
                        )
                        test_table = dense_cls.from_raw_builder(
                            builder,
                            dim,
                            threshold,
                        )
                        legacy_table = dense_cls.from_raw_builder(
                            legacy_builder,
                            dim,
                            threshold,
                        )
                        seed = (
                            6151
                            + 19 * int(0 if operator.bond is None else operator.bond)
                            + int(
                                stats.get(
                                    "packed_boundary_route_validate_calls",
                                    0,
                                )
                            )
                        )
                        rng = np.random.default_rng(seed)
                        vec = (
                            rng.standard_normal(dim)
                            + 1j * rng.standard_normal(dim)
                        ).astype(np.complex128)
                        out_test = np.asarray(
                            test_table.matvec(vec),
                            dtype=np.complex128,
                        )
                        out_legacy = np.asarray(
                            legacy_table.matvec(vec),
                            dtype=np.complex128,
                        )
                        diff = float(np.linalg.norm(out_test - out_legacy))
                        denom = max(1.0, float(np.linalg.norm(out_legacy)))
                        rel = diff / denom
                        legacy_entries = int(legacy_builder.size())
                        stats["packed_boundary_route_validate_calls"] = int(
                            stats.get("packed_boundary_route_validate_calls", 0)
                        ) + 1
                        stats["packed_boundary_route_validate_max_abs"] = max(
                            float(
                                stats.get(
                                    "packed_boundary_route_validate_max_abs",
                                    0.0,
                                )
                            ),
                            diff,
                        )
                        stats["packed_boundary_route_validate_max_rel"] = max(
                            float(
                                stats.get(
                                    "packed_boundary_route_validate_max_rel",
                                    0.0,
                                )
                            ),
                            rel,
                        )
                        stats["packed_boundary_route_validate_last"] = {
                            "packed_entries": int(entry_count),
                            "legacy_entries": int(legacy_entries),
                            "abs": diff,
                            "rel": rel,
                        }
                        if rel > 1.0e-10 and diff > 1.0e-10:
                            stats["packed_boundary_route_validate_failures"] = int(
                                stats.get(
                                    "packed_boundary_route_validate_failures",
                                    0,
                                )
                            ) + 1
                            raise RuntimeError(
                                "packed boundary route payload mismatch: "
                                f"packed_entries={entry_count} "
                                f"legacy_entries={legacy_entries} "
                                f"abs={diff:.3e} rel={rel:.3e}"
                            )
                    elif legacy_changed:
                        stats["packed_boundary_route_validate_skipped_special"] = int(
                            stats.get(
                                "packed_boundary_route_validate_skipped_special",
                                0,
                            )
                        ) + 1
                        stats["packed_boundary_route_validate_skip_reason"] = (
                            "compact_planned_or_same_side_entries"
                        )
                    ref = operator._flat_generator_family_csr_kernels(
                        proto,
                        layout,
                        build_groups=False,
                    )
                    ref_builder = None if ref is None else ref.get("raw_builder")
                    if ref_builder is None or dense_cls is None:
                        raise RuntimeError(
                            "raw_cython_validate requires a reference raw builder "
                            "and GroupedRenormalizedTable"
                        )
                    dim = int(operator._size(layout))
                    threshold = float(
                        operator._renormalized_operator_table_sparse_density_threshold
                    )
                    test_table = dense_cls.from_raw_builder(
                        builder,
                        dim,
                        threshold,
                    )
                    ref_table = dense_cls.from_raw_builder(
                        ref_builder,
                        dim,
                        threshold,
                    )
                    seed = (
                        7919
                        + 17 * int(0 if operator.bond is None else operator.bond)
                        + int(stats.get("packed_route_validate_calls", 0))
                    )
                    rng = np.random.default_rng(seed)
                    vec = (
                        rng.standard_normal(dim)
                        + 1j * rng.standard_normal(dim)
                    ).astype(np.complex128)
                    out_test = np.asarray(
                        test_table.matvec(vec),
                        dtype=np.complex128,
                    )
                    out_ref = np.asarray(
                        ref_table.matvec(vec),
                        dtype=np.complex128,
                    )
                    diff = float(np.linalg.norm(out_test - out_ref))
                    denom = max(1.0, float(np.linalg.norm(out_ref)))
                    rel = diff / denom
                    stats["packed_route_validate_calls"] = int(
                        stats.get("packed_route_validate_calls", 0)
                    ) + 1
                    stats["packed_route_validate_max_abs"] = max(
                        float(stats.get("packed_route_validate_max_abs", 0.0)),
                        diff,
                    )
                    stats["packed_route_validate_max_rel"] = max(
                        float(stats.get("packed_route_validate_max_rel", 0.0)),
                        rel,
                    )
                    if rel > 1.0e-10 and diff > 1.0e-10:
                        raise RuntimeError(
                            "raw_cython fast-key payload matvec mismatch: "
                            f"abs={diff:.3e} rel={rel:.3e}"
                        )
                stats["packed_route_table_backend_actual"] = (
                    "planned_raw_cython" if planned_plan_used else "raw_cython"
                )
                stats["packed_route_table_fallback_reason"] = ""
                stats["packed_route_entries"] = int(
                    stats.get("packed_route_entries", 0)
                ) + entry_count
                return self._raw_builder_payload_record(
                    builder,
                    entry_count=entry_count,
                    family_names=family_names,
                )
            except Exception as exc:
                stats["packed_route_table_failures"] = int(
                    stats.get("packed_route_table_failures", 0)
                ) + 1
                stats["packed_route_table_fallback_reason"] = str(exc)
                return None
            finally:
                elapsed = float(time.perf_counter() - start)
                stats["packed_route_build_seconds"] = float(
                    stats.get("packed_route_build_seconds", 0.0)
                ) + elapsed
                stats["packed_route_table_last_build_seconds"] = elapsed
        if mode not in {"dense_cython", "direct_table", "direct_cython"}:
            stats = self.environment.moving_profile_stats
            stats["packed_route_table_backend_actual"] = "python"
            stats["packed_route_table_fallback_reason"] = f"unknown_mode:{mode}"
            return None
        dense_cls = None if _cpp_davidson is None else getattr(
            _cpp_davidson,
            "GroupedRenormalizedTable",
            None,
        )
        direct_table_fn = None
        if dense_cls is not None:
            direct_table_fn = getattr(
                dense_cls,
                "from_direct_family_environments_fastkeys",
                None,
            ) or getattr(
                dense_cls,
                "from_direct_family_environments",
                None,
            )
        if direct_table_fn is None:
            if mode == "cython":
                raise RuntimeError(
                    "generator_table_packed_route_table='cython' requires "
                    "GroupedRenormalizedTable.from_direct_family_environments"
                )
            return None
        start = time.perf_counter()
        stats = self.environment.moving_profile_stats
        try:
            direct_environments = (
                getattr(
                    operator,
                    "complementary_direct_family_environments",
                    None,
                )
                or {}
            )
            family_environments = (
                getattr(operator, "complementary_family_environments", None) or {}
            )
            backend = direct_table_fn(
                direct_environments,
                family_environments,
                getattr(proto, "data", {}) or {},
                tuple(layout),
                int(operator._size(layout)),
                float(operator._renormalized_operator_table_sparse_density_threshold),
                True,
            )
            family_names = [str(name) for name in direct_environments]
            if family_environments:
                family_names.extend(str(name) for name in family_environments)
            try:
                entry_count = int(backend.n_routes())
            except Exception:
                entry_count = int(backend.n_group_channels())
            if entry_count <= 0:
                return None
            stats["packed_route_table_backend_actual"] = "cython"
            stats["packed_route_table_fallback_reason"] = ""
            stats["packed_route_entries"] = int(
                stats.get("packed_route_entries", 0)
            ) + entry_count
            stats["packed_route_groups"] = int(
                stats.get("packed_route_groups", 0)
            ) + int(backend.n_groups())
            return self._cpp_grouped_table_payload_record(
                backend,
                entry_count=entry_count,
                family_names=family_names,
            )
        except Exception as exc:
            stats["packed_route_table_failures"] = int(
                stats.get("packed_route_table_failures", 0)
            ) + 1
            stats["packed_route_table_fallback_reason"] = str(exc)
            return None
        finally:
            elapsed = float(time.perf_counter() - start)
            stats["packed_route_build_seconds"] = float(
                stats.get("packed_route_build_seconds", 0.0)
            ) + elapsed
            stats["packed_route_table_last_build_seconds"] = elapsed

    def collect_renormalized_operator_payload(self, operator, proto, layout):
        start = time.perf_counter()
        try:
            build_groups = not self.use_cpp_raw_grouped_renormalized_table()
            if getattr(operator, "complementary_direct_family_environments", None):
                cpp_collected = self.collect_direct_renormalized_operator_payload_cpp(
                    operator,
                    proto,
                    layout,
                )
                if cpp_collected is not None:
                    return cpp_collected
                return operator._flat_generator_family_csr_kernels(
                    proto,
                    layout,
                    build_groups=build_groups,
                )
            if self.use_cpp_named_raw_payload_builder():
                route_cls = getattr(_cpp_davidson, "RawRoutePlan", None)
                descriptor_names = tuple(
                    getattr(
                        self.environment,
                        "_cpp_family_mpo_descriptor_names",
                        (),
                    )
                    or ()
                )
                if (
                    descriptor_names
                    and self.use_cpp_named_raw_payload_plan()
                    and self.environment.uses_cpp_family_mpo_descriptor()
                    and operator.bond is not None
                ):
                    plan_cls = getattr(_cpp_davidson, "NamedRawPayloadPlan", None)
                    plan_key = (
                        "moving_environment_cpp_named_raw_payload_plan_descriptor",
                        None if operator.bond is None else int(operator.bond),
                        tuple(proto.dirs),
                        tuple(key for key, _shape in tuple(layout)),
                        descriptor_names,
                    )
                    plan = self.environment._named_raw_payload_plan_cache.get(
                        plan_key
                    )
                    stats = self.environment.moving_profile_stats
                    try:
                        if plan is None:
                            build_start = time.perf_counter()
                            plan = plan_cls()
                            build_elapsed = float(time.perf_counter() - build_start)
                            self.environment._named_raw_payload_plan_cache[
                                plan_key
                            ] = plan
                            stats["cpp_named_raw_payload_plan_builds"] = int(
                                stats.get("cpp_named_raw_payload_plan_builds", 0)
                            ) + 1
                            stats["cpp_named_raw_payload_plan_build_seconds"] = float(
                                stats.get(
                                    "cpp_named_raw_payload_plan_build_seconds",
                                    0.0,
                                )
                            ) + build_elapsed
                            stats[
                                "cpp_named_raw_payload_plan_last_build_seconds"
                            ] = build_elapsed
                        else:
                            stats["cpp_named_raw_payload_plan_cache_hits"] = int(
                                stats.get(
                                    "cpp_named_raw_payload_plan_cache_hits",
                                    0,
                                )
                            ) + 1
                        try:
                            before_index_rebuilds = int(
                                dict(plan.stats()).get("index_rebuilds", 0)
                            )
                        except Exception:
                            before_index_rebuilds = None
                        refresh_start = time.perf_counter()
                        builder = (
                            self.environment._build_cpp_named_payload_from_family_descriptor(
                                plan,
                                operator.bond,
                                layout,
                            )
                        )
                        refresh_elapsed = float(time.perf_counter() - refresh_start)
                        if builder is not None:
                            try:
                                plan_stats = dict(plan.stats())
                            except Exception:
                                plan_stats = {}
                            if before_index_rebuilds is not None:
                                after_index_rebuilds = int(
                                    plan_stats.get("index_rebuilds", 0)
                                )
                                index_rebuild_delta = max(
                                    0,
                                    after_index_rebuilds - before_index_rebuilds,
                                )
                                if index_rebuild_delta:
                                    stats[
                                        "cpp_named_raw_payload_plan_index_rebuilds"
                                    ] = int(
                                        stats.get(
                                            "cpp_named_raw_payload_plan_index_rebuilds",
                                            0,
                                        )
                                    ) + index_rebuild_delta
                                    stats[
                                        "cpp_named_raw_payload_plan_index_rebuild_seconds"
                                    ] = float(
                                        stats.get(
                                            "cpp_named_raw_payload_plan_index_rebuild_seconds",
                                            0.0,
                                        )
                                    ) + refresh_elapsed
                                    stats[
                                        "cpp_named_raw_payload_plan_last_index_rebuild_seconds"
                                    ] = refresh_elapsed
                            stats["cpp_named_raw_payload_plan_refresh_calls"] = int(
                                stats.get(
                                    "cpp_named_raw_payload_plan_refresh_calls",
                                    0,
                                )
                            ) + 1
                            stats["cpp_named_raw_payload_plan_refresh_seconds"] = float(
                                stats.get(
                                    "cpp_named_raw_payload_plan_refresh_seconds",
                                    0.0,
                                )
                            ) + refresh_elapsed
                            stats[
                                "cpp_named_raw_payload_plan_last_refresh_seconds"
                            ] = refresh_elapsed
                            stats[
                                "cpp_named_raw_payload_plan_backend_actual"
                            ] = "cpp_family_mpo_descriptor"
                            for key, value in plan_stats.items():
                                stats[f"cpp_named_raw_payload_plan_last_{key}"] = value
                            entry_count = int(builder.size())
                            if entry_count > 0:
                                return self._raw_builder_payload_record(
                                    builder,
                                    entry_count=entry_count,
                                    family_names=descriptor_names,
                                )
                    except Exception as exc:
                        stats["cpp_named_raw_payload_plan_failures"] = int(
                            stats.get("cpp_named_raw_payload_plan_failures", 0)
                        ) + 1
                        stats["cpp_named_raw_payload_plan_last_error"] = str(exc)
                        self.environment._named_raw_payload_plan_cache.pop(
                            plan_key,
                            None,
                        )
                environments = (
                    getattr(operator, "complementary_family_environments", None) or {}
                )
                family_names = tuple(str(name) for name in environments)
                if self.use_cpp_named_raw_payload_plan():
                    plan_cls = getattr(_cpp_davidson, "NamedRawPayloadPlan", None)
                    plan_key = self.named_raw_payload_plan_cache_key(
                        operator,
                        proto,
                        layout,
                    )
                    plan_record = (
                        None
                        if plan_key is None
                        else self.environment._named_raw_payload_plan_cache.get(plan_key)
                    )
                    if isinstance(plan_record, tuple):
                        plan = plan_record[0]
                    else:
                        plan = plan_record
                    stats = self.environment.moving_profile_stats
                    try:
                        if plan is None:
                            build_start = time.perf_counter()
                            plan = plan_cls()
                            build_elapsed = float(time.perf_counter() - build_start)
                            if plan_key is not None:
                                self.environment._named_raw_payload_plan_cache[
                                    plan_key
                                ] = plan
                            stats["cpp_named_raw_payload_plan_builds"] = int(
                                stats.get("cpp_named_raw_payload_plan_builds", 0)
                            ) + 1
                            stats["cpp_named_raw_payload_plan_build_seconds"] = float(
                                stats.get(
                                    "cpp_named_raw_payload_plan_build_seconds",
                                    0.0,
                                )
                            ) + build_elapsed
                            stats[
                                "cpp_named_raw_payload_plan_last_build_seconds"
                            ] = build_elapsed
                        else:
                            stats["cpp_named_raw_payload_plan_cache_hits"] = int(
                                stats.get(
                                    "cpp_named_raw_payload_plan_cache_hits",
                                    0,
                                )
                            ) + 1
                        if plan is not None:
                            try:
                                before_index_rebuilds = int(
                                    dict(plan.stats()).get("index_rebuilds", 0)
                                )
                            except Exception:
                                before_index_rebuilds = None
                            refresh_start = time.perf_counter()
                            if hasattr(plan, "build_payload_for_layout"):
                                builder = plan.build_payload_for_layout(
                                    environments,
                                    tuple(layout),
                                )
                            else:
                                if hasattr(plan, "refresh_indices_if_needed"):
                                    plan.refresh_indices_if_needed(environments)
                                builder = plan.build_payload(environments)
                            refresh_elapsed = float(
                                time.perf_counter() - refresh_start
                            )
                            try:
                                plan_stats = dict(plan.stats())
                            except Exception:
                                plan_stats = {}
                            if before_index_rebuilds is not None:
                                after_index_rebuilds = int(
                                    plan_stats.get("index_rebuilds", 0)
                                )
                                index_rebuild_delta = max(
                                    0,
                                    after_index_rebuilds - before_index_rebuilds,
                                )
                                if index_rebuild_delta:
                                    rebuild_elapsed = refresh_elapsed
                                    stats[
                                        "cpp_named_raw_payload_plan_index_rebuilds"
                                    ] = int(
                                        stats.get(
                                            "cpp_named_raw_payload_plan_index_rebuilds",
                                            0,
                                        )
                                    ) + index_rebuild_delta
                                    stats[
                                        "cpp_named_raw_payload_plan_index_rebuild_seconds"
                                    ] = float(
                                        stats.get(
                                            "cpp_named_raw_payload_plan_index_rebuild_seconds",
                                            0.0,
                                        )
                                    ) + rebuild_elapsed
                                    stats[
                                        "cpp_named_raw_payload_plan_last_index_rebuild_seconds"
                                    ] = rebuild_elapsed
                            stats["cpp_named_raw_payload_plan_refresh_calls"] = int(
                                stats.get(
                                    "cpp_named_raw_payload_plan_refresh_calls",
                                    0,
                                )
                            ) + 1
                            stats["cpp_named_raw_payload_plan_refresh_seconds"] = float(
                                stats.get(
                                    "cpp_named_raw_payload_plan_refresh_seconds",
                                    0.0,
                                )
                            ) + refresh_elapsed
                            stats[
                                "cpp_named_raw_payload_plan_last_refresh_seconds"
                            ] = refresh_elapsed
                            for key, value in plan_stats.items():
                                stats[f"cpp_named_raw_payload_plan_last_{key}"] = value
                            entry_count = int(builder.size())
                            if entry_count > 0:
                                return self._raw_builder_payload_record(
                                    builder,
                                    entry_count=entry_count,
                                    family_names=family_names,
                                )
                    except Exception as exc:
                        stats["cpp_named_raw_payload_plan_failures"] = int(
                            stats.get("cpp_named_raw_payload_plan_failures", 0)
                        ) + 1
                        stats["cpp_named_raw_payload_plan_last_error"] = str(exc)
                        if plan_key is not None:
                            self.environment._named_raw_payload_plan_cache.pop(
                                plan_key,
                                None,
                            )
                build_start = time.perf_counter()
                try:
                    builder = route_cls.build_named_family_payload(
                        environments,
                        getattr(proto, "data", {}) or {},
                        tuple(layout),
                        True,
                    )
                except Exception as exc:
                    stats = self.environment.moving_profile_stats
                    stats["cpp_named_raw_payload_builder_failures"] = int(
                        stats.get("cpp_named_raw_payload_builder_failures", 0)
                    ) + 1
                    stats["cpp_named_raw_payload_builder_last_error"] = str(exc)
                else:
                    elapsed = float(time.perf_counter() - build_start)
                    stats = self.environment.moving_profile_stats
                    stats["cpp_named_raw_payload_builder_calls"] = int(
                        stats.get("cpp_named_raw_payload_builder_calls", 0)
                    ) + 1
                    stats["cpp_named_raw_payload_builder_seconds"] = float(
                        stats.get("cpp_named_raw_payload_builder_seconds", 0.0)
                    ) + elapsed
                    stats["cpp_named_raw_payload_builder_last_seconds"] = elapsed
                    entry_count = int(builder.size())
                    if entry_count > 0:
                        return self._raw_builder_payload_record(
                            builder,
                            entry_count=entry_count,
                            family_names=family_names,
                        )
            return operator._flat_named_family_csr_kernels(
                proto,
                layout,
                build_groups=build_groups,
            )
        finally:
            elapsed = float(time.perf_counter() - start)
            stats = self.environment.moving_profile_stats
            stats["renormalized_operator_payload_collect_calls"] = int(
                stats.get("renormalized_operator_payload_collect_calls", 0)
            ) + 1
            stats["renormalized_operator_payload_collect_seconds"] = float(
                stats.get("renormalized_operator_payload_collect_seconds", 0.0)
            ) + elapsed
            stats["renormalized_operator_payload_collect_last_seconds"] = elapsed

    def build_renormalized_operator_table(self, operator, collected, proto, layout):
        cpp_table = self.build_cpp_grouped_renormalized_operator_table(
            operator,
            collected,
            proto,
            layout,
        )
        if cpp_table is not None:
            return cpp_table
        if collected.get("raw_builder") is not None:
            collected = operator._materialize_raw_builder_collected(collected)
        return AbelianRenormalizedOperatorActionTable(
            collected,
            operator._size(layout),
            layout,
            operator._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=operator.bond,
            source="moving_environment_renormalized_operator_table",
            boundary_family_tables=operator._boundary_family_tables(),
            max_dense_block_elements=(
                operator._renormalized_operator_table_dense_block_max_elements
            ),
            sparse_density_threshold=(
                operator._renormalized_operator_table_sparse_density_threshold
            ),
        )

    def build_cpp_grouped_renormalized_operator_table(
        self,
        operator,
        collected,
        proto,
        layout,
    ):
        enabled_opt = MovingEnvironment._option_value(
            self.environment.matvec_options,
            "moving_environment_cpp_grouped_renormalized_table",
            None,
        )
        if enabled_opt is None:
            enabled = bool(
                MovingEnvironment._option_value(
                    self.environment.matvec_options,
                    "moving_environment_cpp_davidson",
                    False,
                )
                or MovingEnvironment._option_value(
                    self.environment.matvec_options,
                    "moving_environment_cpp_matvec",
                    False,
                )
            )
        else:
            enabled = bool(enabled_opt)
        if not bool(enabled):
            return None
        if (
            _cpp_davidson is None
            or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
            or (
                getattr(_cpp_davidson, "GroupedFactorizedTable", None) is None
                and getattr(_cpp_davidson, "GroupedRenormalizedTable", None) is None
            )
        ):
            return None
        try:
            factorized_cls = getattr(_cpp_davidson, "GroupedFactorizedTable", None)
            dense_cls = getattr(_cpp_davidson, "GroupedRenormalizedTable", None)
            prebuilt = collected.get("cpp_grouped_table")
            if prebuilt is not None:
                build_start = time.perf_counter()
                backend = prebuilt
                if dense_cls is not None and not isinstance(backend, dense_cls):
                    return None
                build_seconds = float(time.perf_counter() - build_start)
                stats = self.environment.moving_profile_stats
                stats["cpp_grouped_renormalized_table_builds"] = int(
                    stats.get("cpp_grouped_renormalized_table_builds", 0)
                ) + 1
                stats["cpp_grouped_renormalized_table_build_seconds"] = float(
                    stats.get("cpp_grouped_renormalized_table_build_seconds", 0.0)
                ) + build_seconds
                stats["cpp_grouped_renormalized_table_prebuilt_builds"] = int(
                    stats.get(
                        "cpp_grouped_renormalized_table_prebuilt_builds",
                        0,
                    )
                ) + 1
                stats["cpp_grouped_renormalized_table_last_storage"] = str(
                    backend.storage()
                )
                try:
                    stats["cpp_grouped_renormalized_table_last_refresh_kind"] = str(
                        backend.last_refresh_kind()
                    )
                except Exception:
                    stats["cpp_grouped_renormalized_table_last_refresh_kind"] = (
                        "direct_route_build"
                    )
                stats["cpp_grouped_renormalized_table_last_blocks"] = int(
                    backend.n_blocks()
                )
                stats["cpp_grouped_renormalized_table_last_elements"] = int(
                    backend.block_matrix_elements()
                )
                stats["cpp_grouped_renormalized_table_last_sparse_nnz"] = int(
                    backend.block_sparse_nnz()
                )
                return MovingEnvironmentGroupedRenormalizedTable(
                    backend,
                    collected,
                    int(operator._size(layout)),
                    layout,
                    operator._qns_from_layout_with_proto(layout, proto),
                    proto.dirs[:],
                    bond=operator.bond,
                    source="moving_environment_cpp_grouped_direct_route_table",
                    boundary_family_tables=operator._boundary_family_tables(),
                )
            use_factorized = bool(
                MovingEnvironment._option_value(
                    self.environment.matvec_options,
                    "moving_environment_cpp_grouped_factorized_table",
                    False,
                )
            )
            raw_enabled = bool(
                not use_factorized
                and self.use_cpp_raw_grouped_renormalized_table()
                and (
                    collected.get("raw_builder") is not None
                    or (
                        collected.get("left")
                        and collected.get("right")
                        and "dims_array" in collected
                        and "in_starts_array" in collected
                        and "out_starts_array" in collected
                    )
                )
            )
            groups = collected.get("matvec_groups")
            if not groups and not raw_enabled:
                return None
            if raw_enabled:
                raw_builder = collected.get("raw_builder")
                if raw_builder is not None:
                    capacity = int(raw_builder.grouped_capacity())
                else:
                    dims_array = np.ascontiguousarray(
                        collected.get("dims_array"),
                        dtype=np.int64,
                    )
                    in_starts_array = np.ascontiguousarray(
                        collected.get("in_starts_array"),
                        dtype=np.int64,
                    )
                    out_starts_array = np.ascontiguousarray(
                        collected.get("out_starts_array"),
                        dtype=np.int64,
                    )
                    if (
                        dims_array.ndim != 2
                        or dims_array.shape[1] != 8
                        or in_starts_array.shape[0] != dims_array.shape[0]
                        or out_starts_array.shape[0] != dims_array.shape[0]
                    ):
                        return None
                    capacity = MovingEnvironmentGroupedRenormalizedTable._raw_group_capacity(
                        dims_array,
                        in_starts_array,
                        out_starts_array,
                    )
            else:
                dims_array = np.ascontiguousarray(
                    collected.get("group_dims_array"),
                    dtype=np.int64,
                )
                if dims_array.ndim != 2 or dims_array.shape[1] != 8:
                    return None
                in_sizes = (
                    dims_array[:, 4]
                    * dims_array[:, 5]
                    * dims_array[:, 6]
                    * dims_array[:, 7]
                )
                out_sizes = (
                    dims_array[:, 0]
                    * dims_array[:, 1]
                    * dims_array[:, 2]
                    * dims_array[:, 3]
                )
                capacity = int(np.sum(in_sizes * out_sizes, dtype=np.int64))
            cap = int(operator._renormalized_operator_table_dense_block_max_elements)
            backend_cls = factorized_cls if use_factorized and factorized_cls is not None else dense_cls
            if backend_cls is None:
                return None
            if capacity <= 0 or (
                backend_cls is dense_cls and cap > 0 and capacity > cap
            ):
                return None
            build_start = time.perf_counter()
            if raw_enabled and backend_cls is dense_cls:
                raw_builder = collected.get("raw_builder")
                if raw_builder is not None:
                    backend = backend_cls.from_raw_builder(
                        raw_builder,
                        int(operator._size(layout)),
                        float(operator._renormalized_operator_table_sparse_density_threshold),
                    )
                else:
                    (
                        raw_left,
                        raw_right,
                        raw_dims,
                        raw_in_starts,
                        raw_out_starts,
                        raw_scales,
                    ) = MovingEnvironmentGroupedRenormalizedTable._raw_payload_arrays(
                        collected
                    )
                    backend = backend_cls.from_raw(
                        raw_left,
                        raw_right,
                        raw_dims,
                        raw_in_starts,
                        raw_out_starts,
                        int(operator._size(layout)),
                        float(operator._renormalized_operator_table_sparse_density_threshold),
                        raw_scales,
                    )
            else:
                (
                    group_left,
                    group_right,
                    _dims_array,
                    _in_starts,
                    _out_starts,
                    group_scales,
                ) = MovingEnvironmentGroupedRenormalizedTable._group_payload_arrays(
                    collected
                )
                backend = backend_cls(
                    group_left,
                    group_right,
                    _dims_array,
                    _in_starts,
                    _out_starts,
                    int(operator._size(layout)),
                    float(operator._renormalized_operator_table_sparse_density_threshold),
                    group_scales,
                )
        except Exception as exc:
            stats = self.environment.moving_profile_stats
            stats["cpp_grouped_renormalized_table_failures"] = int(
                stats.get("cpp_grouped_renormalized_table_failures", 0)
            ) + 1
            stats["cpp_grouped_renormalized_table_last_error"] = str(exc)
            return None
        build_seconds = float(time.perf_counter() - build_start)
        stats = self.environment.moving_profile_stats
        stats["cpp_grouped_renormalized_table_builds"] = int(
            stats.get("cpp_grouped_renormalized_table_builds", 0)
        ) + 1
        stats["cpp_grouped_renormalized_table_build_seconds"] = float(
            stats.get("cpp_grouped_renormalized_table_build_seconds", 0.0)
        ) + build_seconds
        stats["cpp_grouped_renormalized_table_last_storage"] = str(backend.storage())
        if raw_enabled:
            stats["cpp_grouped_renormalized_table_raw_builds"] = int(
                stats.get("cpp_grouped_renormalized_table_raw_builds", 0)
            ) + 1
            raw_builder_for_stats = collected.get("raw_builder")
            if raw_builder_for_stats is not None:
                stats["cpp_grouped_renormalized_table_raw_builder_builds"] = int(
                    stats.get(
                        "cpp_grouped_renormalized_table_raw_builder_builds",
                        0,
                    )
                ) + 1
                analysis_fn = getattr(
                    dense_cls,
                    "raw_builder_hybrid_analysis",
                    None,
                )
                if analysis_fn is not None:
                    try:
                        analysis = dict(
                            analysis_fn(
                                raw_builder_for_stats,
                                int(operator._size(layout)),
                                float(
                                    operator._renormalized_operator_table_sparse_density_threshold
                                ),
                            )
                        )
                    except Exception as exc:
                        stats[
                            "cpp_grouped_renormalized_table_hybrid_analysis_error"
                        ] = str(exc)
                    else:
                        for key, value in analysis.items():
                            stat_key = (
                                "cpp_grouped_renormalized_table_hybrid_"
                                f"{key}"
                            )
                            if isinstance(value, (bool, np.bool_)):
                                stats[stat_key] = bool(value)
                            elif isinstance(value, (int, np.integer)):
                                stats[stat_key] = int(value)
                            elif isinstance(value, (float, np.floating)):
                                stats[stat_key] = float(value)
                            else:
                                stats[stat_key] = str(value)
        try:
            stats["cpp_grouped_renormalized_table_last_refresh_kind"] = str(
                backend.last_refresh_kind()
            )
        except Exception:
            stats["cpp_grouped_renormalized_table_last_refresh_kind"] = "build"
        stats["cpp_grouped_renormalized_table_last_blocks"] = int(backend.n_blocks())
        stats["cpp_grouped_renormalized_table_last_elements"] = int(
            backend.block_matrix_elements()
        )
        stats["cpp_grouped_renormalized_table_last_sparse_nnz"] = int(
            backend.block_sparse_nnz()
        )
        try:
            index_cache_stats = dict(_cpp_davidson.GroupedRenormalizedTable.index_cache_stats())
        except Exception:
            index_cache_stats = {}
        if index_cache_stats:
            stats["cpp_grouped_renormalized_table_index_cache_entries"] = int(
                index_cache_stats.get("entries", 0)
            )
            stats["cpp_grouped_renormalized_table_index_cache_hits"] = int(
                index_cache_stats.get("hits", 0)
            )
            stats["cpp_grouped_renormalized_table_index_cache_misses"] = int(
                index_cache_stats.get("misses", 0)
            )
        return MovingEnvironmentGroupedRenormalizedTable(
            backend,
            collected,
            int(operator._size(layout)),
            layout,
            operator._qns_from_layout_with_proto(layout, proto),
            proto.dirs[:],
            bond=operator.bond,
            source="moving_environment_cpp_grouped_renormalized_table",
            boundary_family_tables=operator._boundary_family_tables(),
        )

    def apply_renormalized_operator_table(self, table, vector):
        vector = np.ascontiguousarray(vector, dtype=np.complex128)
        if bool(getattr(self.environment, "use_cpp_block_matvec", False)):
            cpp_table = self.environment.cpp_renormalized_table(
                table,
                validation_vector=vector,
            )
        else:
            cpp_table = None
        if cpp_table is not None:
            start = time.perf_counter()
            try:
                out = cpp_table.matvec(vector)
            except Exception as exc:
                stats = self.environment.moving_profile_stats
                stats["cpp_block_matvec_failures"] = int(
                    stats.get("cpp_block_matvec_failures", 0)
                ) + 1
                stats["cpp_block_matvec_last_error"] = str(exc)
            else:
                elapsed = float(time.perf_counter() - start)
                stats = self.environment.moving_profile_stats
                stats["cpp_block_matvec_calls"] = int(
                    stats.get("cpp_block_matvec_calls", 0)
                ) + 1
                stats["cpp_block_matvec_seconds"] = float(
                    stats.get("cpp_block_matvec_seconds", 0.0)
                ) + elapsed
                stats["cpp_block_matvec_last_seconds"] = elapsed
                stats["cpp_renormalized_table_matvec_calls"] = int(
                    stats.get("cpp_renormalized_table_matvec_calls", 0)
                ) + 1
                stats["cpp_renormalized_table_matvec_seconds"] = float(
                    stats.get("cpp_renormalized_table_matvec_seconds", 0.0)
                ) + elapsed
                stats["cpp_renormalized_table_matvec_last_seconds"] = elapsed
                return out
        return table.matvec(vector)

    def use_cpp_environment_plan(self):
        if not self.use_cpp_environment_update():
            return False
        plan_cls = None if _cpp_davidson is None else getattr(
            _cpp_davidson,
            "AbelianEnvironmentAdvancePlan",
            None,
        )
        if plan_cls is None:
            return False
        return bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_environment_plan",
                True,
            )
        )

    def _environment_plan_key(self, direction, W, A, E_or_F, B):
        return (
            "abelian_environment_advance_plan",
            str(direction),
            self._tensor_key_signature(W),
            self._tensor_key_signature(A),
            self._tensor_key_signature(E_or_F),
            self._tensor_key_signature(B),
        )

    def _environment_plan_owner_key(self, key):
        text = repr(key)
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        slot = getattr(self.environment, "_environment_advance_slot_key", None)
        if slot is None:
            return f"environment-plan-direct:{digest}", digest
        slot_text = ":".join(str(part) for part in slot)
        return f"environment-plan-slot:{slot_text}", digest

    def _record_cpp_environment_owner_plan_stats(self):
        owner = self.environment._cpp_moving_environment
        if owner is None or not hasattr(owner, "stats"):
            return
        try:
            owner_stats = dict(owner.stats())
        except Exception as exc:
            self.environment.moving_profile_stats[
                "cpp_environment_plan_owner_last_stats_error"
            ] = str(exc)
            return
        stats = self.environment.moving_profile_stats
        mapping = {
            "environment_plan_builds": "cpp_environment_plan_builds",
            "environment_plan_build_seconds": "cpp_environment_plan_build_seconds",
            "environment_plan_cache_hits": "cpp_environment_plan_cache_hits",
            "environment_plan_advance_calls": "cpp_environment_plan_advance_calls",
            "environment_plan_advance_seconds": "cpp_environment_plan_advance_seconds",
            "environment_plan_failures": "cpp_environment_plan_failures",
            "environment_plan_last_routes": "cpp_environment_plan_last_routes",
            "environment_plan_last_blocks": "cpp_environment_plan_last_blocks",
        }
        for src, dst in mapping.items():
            if src in owner_stats:
                if src.endswith("_seconds"):
                    stats[dst] = float(owner_stats[src])
                else:
                    stats[dst] = int(owner_stats[src])
        stats["cpp_environment_plan_backend_actual"] = "cpp_moving_environment"
        stats["cpp_environment_plan_owner_records"] = int(
            owner_stats.get("environment_plan_records", 0) or 0
        )
        if "environment_plan_last_error" in owner_stats:
            stats["cpp_environment_plan_last_error"] = owner_stats[
                "environment_plan_last_error"
            ]

    def _cpp_environment_owner_plan_update(self, direction, W, A, E_or_F, B, key):
        if not bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_environment_owner_plan",
                True,
            )
        ):
            return None
        owner = self.environment._cpp_moving_environment
        if owner is None or not hasattr(owner, "environment_advance"):
            return None
        stats = self.environment.moving_profile_stats
        try:
            owner_key, signature = self._environment_plan_owner_key(key)
            keys, blocks, qns, dirs = owner.environment_advance(
                owner_key,
                str(direction),
                W,
                A,
                E_or_F,
                B,
                signature,
            )
        except Exception as exc:
            stats["cpp_environment_plan_owner_failures"] = int(
                stats.get("cpp_environment_plan_owner_failures", 0)
            ) + 1
            stats["cpp_environment_plan_owner_last_error"] = str(exc)
            return None
        self._record_cpp_environment_owner_plan_stats()
        data = OrderedDict(
            (tuple(key), np.asarray(block))
            for key, block in zip(keys, blocks)
        )
        stats["cpp_environment_plan_last_blocks"] = int(len(data))
        return AbelianEnvironmentTensorData(
            data,
            qns,
            dirs,
            copy=False,
        )

    def _cpp_environment_plan_update(self, direction, W, A, E_or_F, B):
        if not self.use_cpp_environment_plan():
            return None
        plan_cls = getattr(_cpp_davidson, "AbelianEnvironmentAdvancePlan", None)
        if plan_cls is None:
            return None
        stats = self.environment.moving_profile_stats
        key = self._environment_plan_key(direction, W, A, E_or_F, B)
        updated = self._cpp_environment_owner_plan_update(
            direction,
            W,
            A,
            E_or_F,
            B,
            key,
        )
        if updated is not None:
            return updated
        cache = self.environment._environment_advance_plan_cache
        plan = cache.get(key)
        if plan is None:
            build_start = time.perf_counter()
            try:
                if direction == "left":
                    plan = plan_cls.from_left(W, A, E_or_F, B)
                else:
                    plan = plan_cls.from_right(W, A, E_or_F, B)
            except Exception as exc:
                stats["cpp_environment_plan_failures"] = int(
                    stats.get("cpp_environment_plan_failures", 0)
                ) + 1
                stats["cpp_environment_plan_last_error"] = str(exc)
                return None
            elapsed = float(time.perf_counter() - build_start)
            cache[key] = plan
            stats["cpp_environment_plan_builds"] = int(
                stats.get("cpp_environment_plan_builds", 0)
            ) + 1
            stats["cpp_environment_plan_build_seconds"] = float(
                stats.get("cpp_environment_plan_build_seconds", 0.0)
            ) + elapsed
            try:
                stats["cpp_environment_plan_last_routes"] = int(plan.route_count())
            except Exception:
                stats["cpp_environment_plan_last_routes"] = 0
        else:
            stats["cpp_environment_plan_cache_hits"] = int(
                stats.get("cpp_environment_plan_cache_hits", 0)
            ) + 1
        stats["cpp_environment_plan_backend_actual"] = "standalone_cpp_plan"
        advance_start = time.perf_counter()
        try:
            keys, blocks, qns, dirs = plan.advance(W, A, E_or_F, B)
        except Exception as exc:
            stats["cpp_environment_plan_failures"] = int(
                stats.get("cpp_environment_plan_failures", 0)
            ) + 1
            stats["cpp_environment_plan_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - advance_start)
        stats["cpp_environment_plan_advance_calls"] = int(
            stats.get("cpp_environment_plan_advance_calls", 0)
        ) + 1
        stats["cpp_environment_plan_advance_seconds"] = float(
            stats.get("cpp_environment_plan_advance_seconds", 0.0)
        ) + elapsed
        stats["cpp_environment_plan_last_advance_seconds"] = elapsed
        data = OrderedDict(
            (tuple(key), np.asarray(block))
            for key, block in zip(keys, blocks)
        )
        stats["cpp_environment_plan_last_blocks"] = int(len(data))
        return AbelianEnvironmentTensorData(
            data,
            qns,
            dirs,
            copy=False,
        )

    def update_left_environment(self, W, A, E, B):
        if self.use_cpp_dense_environment_update():
            updated = self._cpp_dense_update_left_environment(W, A, E, B)
            if updated is not None:
                self.environment._last_environment_update_backend = "cpp_dense_environment"
                return updated
        if self.use_cpp_environment_update():
            updated = self._cpp_update_left_environment(W, A, E, B)
            if updated is not None:
                self.environment._last_environment_update_backend = "cpp_native_environment"
                return updated
        self.environment._last_environment_update_backend = "python_contract"
        return contract_from_left(W, A, E, B)

    def update_right_environment(self, W, A, F, B):
        if self.use_cpp_dense_environment_update():
            updated = self._cpp_dense_update_right_environment(W, A, F, B)
            if updated is not None:
                self.environment._last_environment_update_backend = "cpp_dense_environment"
                return updated
        if self.use_cpp_environment_update():
            updated = self._cpp_update_right_environment(W, A, F, B)
            if updated is not None:
                self.environment._last_environment_update_backend = "cpp_native_environment"
                return updated
        self.environment._last_environment_update_backend = "python_contract"
        return contract_from_right(W, A, F, B)

    def use_cpp_environment_update(self):
        if _cpp_davidson is None or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
            return False
        return bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_cpp_environment_update",
                False,
            )
        )

    def use_cpp_dense_environment_update(self):
        if _cpp_davidson is None or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False):
            return False
        if (
            getattr(_cpp_davidson, "dense_environment_update_left", None) is None
            or getattr(_cpp_davidson, "dense_environment_update_right", None) is None
        ):
            return False
        return bool(
            MovingEnvironment._option_value(
                self.environment.matvec_options,
                "moving_environment_dense_cpp_environment_update",
                False,
            )
        )

    @staticmethod
    def _dense_environment_inputs_supported(W, A, E_or_F, B):
        return (
            isinstance(W, np.ndarray)
            and isinstance(A, np.ndarray)
            and isinstance(E_or_F, np.ndarray)
            and isinstance(B, np.ndarray)
            and W.ndim == 4
            and A.ndim == 3
            and E_or_F.ndim == 3
            and B.ndim == 3
        )

    def _cpp_dense_update_left_environment(self, W, A, E, B):
        if not self._dense_environment_inputs_supported(W, A, E, B):
            return None
        stats = self.environment.moving_profile_stats
        start = time.perf_counter()
        try:
            updated = _cpp_davidson.dense_environment_update_left(
                np.asarray(W, dtype=np.complex128),
                np.asarray(A, dtype=np.complex128),
                np.asarray(E, dtype=np.complex128),
                np.asarray(B, dtype=np.complex128),
            )
        except Exception as exc:
            stats["cpp_environment_update_failures"] = int(
                stats.get("cpp_environment_update_failures", 0)
            ) + 1
            stats["dense_cpp_environment_update_failures"] = int(
                stats.get("dense_cpp_environment_update_failures", 0)
            ) + 1
            stats["cpp_environment_update_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - start)
        stats["cpp_environment_update_left_calls"] = int(
            stats.get("cpp_environment_update_left_calls", 0)
        ) + 1
        stats["cpp_environment_update_calls"] = int(
            stats.get("cpp_environment_update_calls", 0)
        ) + 1
        stats["cpp_environment_update_seconds"] = float(
            stats.get("cpp_environment_update_seconds", 0.0)
        ) + elapsed
        stats["cpp_environment_update_last_seconds"] = elapsed
        stats["cpp_environment_update_backend_actual"] = "cpp_dense_payload"
        stats["dense_cpp_environment_update_calls"] = int(
            stats.get("dense_cpp_environment_update_calls", 0)
        ) + 1
        stats["dense_cpp_environment_update_seconds"] = float(
            stats.get("dense_cpp_environment_update_seconds", 0.0)
        ) + elapsed
        return np.asarray(updated)

    def _cpp_dense_update_right_environment(self, W, A, F, B):
        if not self._dense_environment_inputs_supported(W, A, F, B):
            return None
        stats = self.environment.moving_profile_stats
        start = time.perf_counter()
        try:
            updated = _cpp_davidson.dense_environment_update_right(
                np.asarray(W, dtype=np.complex128),
                np.asarray(A, dtype=np.complex128),
                np.asarray(F, dtype=np.complex128),
                np.asarray(B, dtype=np.complex128),
            )
        except Exception as exc:
            stats["cpp_environment_update_failures"] = int(
                stats.get("cpp_environment_update_failures", 0)
            ) + 1
            stats["dense_cpp_environment_update_failures"] = int(
                stats.get("dense_cpp_environment_update_failures", 0)
            ) + 1
            stats["cpp_environment_update_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - start)
        stats["cpp_environment_update_right_calls"] = int(
            stats.get("cpp_environment_update_right_calls", 0)
        ) + 1
        stats["cpp_environment_update_calls"] = int(
            stats.get("cpp_environment_update_calls", 0)
        ) + 1
        stats["cpp_environment_update_seconds"] = float(
            stats.get("cpp_environment_update_seconds", 0.0)
        ) + elapsed
        stats["cpp_environment_update_last_seconds"] = elapsed
        stats["cpp_environment_update_backend_actual"] = "cpp_dense_payload"
        stats["dense_cpp_environment_update_calls"] = int(
            stats.get("dense_cpp_environment_update_calls", 0)
        ) + 1
        stats["dense_cpp_environment_update_seconds"] = float(
            stats.get("dense_cpp_environment_update_seconds", 0.0)
        ) + elapsed
        return np.asarray(updated)

    def _cpp_update_left_environment(self, W, A, E, B):
        stats = self.environment.moving_profile_stats
        start = time.perf_counter()
        try:
            updated = self._cpp_environment_plan_update("left", W, A, E, B)
            if updated is not None:
                stats["cpp_environment_update_backend_actual"] = (
                    "cpp_environment_plan"
                )
            else:
                updated = abelian_contract_from_left_data(
                    W,
                    A,
                    E,
                    B,
                )
                stats["cpp_environment_update_backend_actual"] = (
                    "cpp_native_payload"
                )
        except Exception as exc:
            stats["cpp_environment_update_failures"] = int(
                stats.get("cpp_environment_update_failures", 0)
            ) + 1
            stats["cpp_environment_update_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - start)
        stats["cpp_environment_update_left_calls"] = int(
            stats.get("cpp_environment_update_left_calls", 0)
        ) + 1
        stats["cpp_environment_update_calls"] = int(
            stats.get("cpp_environment_update_calls", 0)
        ) + 1
        stats["cpp_environment_update_seconds"] = float(
            stats.get("cpp_environment_update_seconds", 0.0)
        ) + elapsed
        stats["cpp_environment_update_last_seconds"] = elapsed
        return updated

    def _cpp_update_right_environment(self, W, A, F, B):
        stats = self.environment.moving_profile_stats
        start = time.perf_counter()
        try:
            updated = self._cpp_environment_plan_update("right", W, A, F, B)
            if updated is not None:
                stats["cpp_environment_update_backend_actual"] = (
                    "cpp_environment_plan"
                )
            else:
                updated = abelian_contract_from_right_data(
                    W,
                    A,
                    F,
                    B,
                )
                stats["cpp_environment_update_backend_actual"] = (
                    "cpp_native_payload"
                )
        except Exception as exc:
            stats["cpp_environment_update_failures"] = int(
                stats.get("cpp_environment_update_failures", 0)
            ) + 1
            stats["cpp_environment_update_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - start)
        stats["cpp_environment_update_right_calls"] = int(
            stats.get("cpp_environment_update_right_calls", 0)
        ) + 1
        stats["cpp_environment_update_calls"] = int(
            stats.get("cpp_environment_update_calls", 0)
        ) + 1
        stats["cpp_environment_update_seconds"] = float(
            stats.get("cpp_environment_update_seconds", 0.0)
        ) + elapsed
        stats["cpp_environment_update_last_seconds"] = elapsed
        return updated


class MovingEnvironmentGroupedRenormalizedTable(AbelianGroupedRenormalizedDataTable):
    """C++ grouped table with native-aware compatibility adapters."""

    def flatten(self, A):
        return self.flatten_data(getattr(A, "data", {}) or {})

    def unflatten(self, vec):
        return _blocktensor_from_abelian_layout_data(
            self.vector_layout,
            self.unflatten_data(vec),
        )

    def apply(self, A):
        return _tensor_from_abelian_layout_data_like(
            A,
            self.vector_layout,
            self.apply_data(getattr(A, "data", {}) or {}),
        )

    def apply_channels(self, A):
        return {}


MovingEnvironmentFlatMatvec = AbelianMovingEnvironmentFlatMatvec
MovingEnvironmentCompactBlockTable = AbelianCompactBlockDataTable


MovingEnvironmentCompactRenormalizedTable = AbelianCompactRenormalizedDataTable
MovingEnvironmentCompactPlanOperator = MovingEnvironmentCompactRenormalizedTable


class MovingEnvironment:
    """Block2-like facade for Abelian/spatial local DMRG environments.

    The first implementation intentionally wraps ``HamiltonianMultiplyU1``.  It
    centralizes local-operator construction, renormalized-table storage, and
    environment-update entry points without changing the numerical matvec path.
    """

    def __init__(
        self,
        *,
        complementary_operator_families=None,
        matvec_options=None,
    ):
        self.complementary_operator_families = complementary_operator_families
        self.matvec_options = matvec_options
        self.compiled_backend = MovingEnvironmentCompiledBackend(self)
        self.operator = None
        self._operatorless_local_problem_active = False
        self._dense_operatorless_local_problem_active = False
        self._local_profile_stats = {}
        self._dense_local_profile_stats = {}
        self._dense_operatorless_key = None
        self.bond = None
        self.left_environments = None
        self.right_environments = None
        self.complementary_left_environments = {}
        self.complementary_right_environments = {}
        self.complementary_operator_mpos = {}
        self._family_environment_descriptor = ()
        self._family_local_pair_cache = {}
        self._family_environment_cache = {}
        self._cpp_family_mpo_descriptor_key = None
        self._cpp_family_mpo_descriptor_names = ()
        self._cpp_owned_family_mpo_key = self._option_value(
            matvec_options,
            "moving_environment_cpp_owned_family_mpo_key",
            None,
        )
        owned_family_mpo_names = self._option_value(
            matvec_options,
            "moving_environment_cpp_owned_family_mpo_names",
            (),
        )
        self._cpp_owned_family_mpo_names = tuple(
            str(name) for name in tuple(owned_family_mpo_names or ())
        )
        self._cpp_qchem_family_descriptor_key = self._option_value(
            matvec_options,
            "moving_environment_cpp_qchem_family_descriptor_key",
            None,
        )
        qchem_family_descriptor_names = self._option_value(
            matvec_options,
            "moving_environment_cpp_qchem_family_descriptor_names",
            (),
        )
        self._cpp_qchem_family_descriptor_names = tuple(
            str(name) for name in tuple(qchem_family_descriptor_names or ())
        )
        self._environment_advance_slot_key = None
        self._pending_cpp_bond_environment_step = None
        self._last_cpp_bond_environment_step = None
        self._direct_family_cache_invalidator = None
        self._direct_family_revision_ref = None
        self._direct_family_cache_maps = ()
        self._owner_direct_family_environment_cache = {}
        self._owner_direct_family_prepared_payloads = {}
        self._owner_typed_direct_family_after_environment_consume = None
        self.direct_family_revision = 0
        self.use_cpp_block_matvec = bool(
            self._option_value(
                matvec_options,
                "moving_environment_cpp_matvec",
                False,
            )
        )
        self._renormalized_operator_table_cache = {}
        self._raw_route_plan_cache = {}
        self._named_raw_payload_plan_cache = {}
        self._incremental_renormalized_operator_table_cache = {}
        self._grouped_renormalized_table_bond_slots = {}
        self._grouped_renormalized_table_slot = None
        self._grouped_renormalized_table_slot_key = (
            "moving_environment_grouped_renormalized_table_slot",
            id(self),
        )
        self._compiled_flat_matvec_cache = {}
        self._compact_renormalized_table_cache = {}
        self._compact_renormalized_table_bond_slots = {}
        self._compact_plan_validation_cache = {}
        self._compact_block_table_cache = {}
        self._environment_advance_plan_cache = {}
        self._dense_cpp_sweep_workspace = None
        self._dense_cpp_sweep_bind_signatures = {}
        self._dense_cpp_sweep_w_signatures = {}
        self._dense_cpp_coarse_grained_w_cache = {}
        self._dense_cpp_coarse_grained_w_signatures = {}
        self._cpp_moving_environment = None
        existing_owner = self._option_value(
            matvec_options,
            "moving_environment_cpp_state_owner_instance",
            None,
        )
        if existing_owner is not None:
            self._cpp_moving_environment = existing_owner
        elif bool(
            self._option_value(
                matvec_options,
                "moving_environment_cpp_state_owner",
                False,
            )
        ):
            owner_cls = None if _cpp_davidson is None else getattr(
                _cpp_davidson,
                "MovingEnvironment",
                None,
            )
            if owner_cls is not None:
                try:
                    self._cpp_moving_environment = owner_cls()
                except Exception as exc:
                    self._cpp_moving_environment = None
                    self._cpp_moving_environment_error = str(exc)
        self.moving_profile_stats = {
            "local_operator_builds": 0,
            "renormalized_operator_table_builds": 0,
            "renormalized_operator_table_build_seconds": 0.0,
            "renormalized_operator_table_cache_hits": 0,
            "renormalized_operator_table_refreshes": 0,
            "renormalized_operator_table_refresh_seconds": 0.0,
            "renormalized_operator_table_structural_cache_hits": 0,
            "renormalized_operator_table_slot_reuses": 0,
            "cpp_raw_route_plan_builds": 0,
            "cpp_raw_route_plan_cache_hits": 0,
            "cpp_raw_route_plan_cache_misses": 0,
            "cpp_raw_route_plan_refresh_calls": 0,
            "cpp_raw_route_plan_refresh_seconds": 0.0,
            "cpp_raw_route_plan_refresh_failures": 0,
            "cpp_environment_update_calls": 0,
            "cpp_environment_update_left_calls": 0,
            "cpp_environment_update_right_calls": 0,
            "cpp_environment_update_seconds": 0.0,
            "cpp_environment_update_failures": 0,
            "cpp_environment_plan_builds": 0,
            "cpp_environment_plan_build_seconds": 0.0,
            "cpp_environment_plan_cache_hits": 0,
            "cpp_environment_plan_advance_calls": 0,
            "cpp_environment_plan_advance_seconds": 0.0,
            "cpp_environment_plan_failures": 0,
            "cpp_environment_plan_owner_failures": 0,
            "cpp_environment_stack_resets": 0,
            "cpp_environment_stack_pushes": 0,
            "cpp_environment_stack_pops": 0,
            "cpp_environment_stack_failures": 0,
            "cpp_sweep_environment_step_calls": 0,
            "cpp_sweep_environment_step_updates": 0,
            "cpp_sweep_environment_step_pops": 0,
            "cpp_sweep_environment_step_syncs": 0,
            "cpp_sweep_environment_step_seconds": 0.0,
            "cpp_sweep_environment_step_failures": 0,
            "cpp_sweep_environment_step_auto_calls": 0,
            "cpp_bond_step_transaction_attempts": 0,
            "cpp_bond_step_transaction_calls": 0,
            "cpp_bond_step_transaction_accepted": 0,
            "cpp_bond_step_transaction_failures": 0,
            "cpp_bond_step_transaction_environment_updates": 0,
            "cpp_bond_step_transaction_record_builds": 0,
            "cpp_bond_step_transaction_record_prepares": 0,
            "cpp_bond_step_transaction_record_consumes": 0,
            "cpp_bond_step_transaction_commits": 0,
            "cpp_bond_step_transaction_seconds": 0.0,
            "cpp_bond_step_transaction_last_seconds": 0.0,
            "cpp_bond_step_transaction_backend_actual": None,
            "cpp_bond_step_transaction_commit_backend_actual": None,
            "cpp_bond_step_transaction_last_error": None,
            "owner_bond_step_calls": 0,
            "owner_bond_step_accepts": 0,
            "owner_bond_step_failures": 0,
            "owner_bond_step_environment_moves": 0,
            "owner_bond_step_environment_fallbacks": 0,
            "owner_bond_step_payload_prepares": 0,
            "owner_bond_step_payload_prepare_seconds": 0.0,
            "owner_bond_step_payload_prepare_last_seconds": 0.0,
            "owner_bond_step_seconds": 0.0,
            "owner_bond_step_last_seconds": 0.0,
            "owner_bond_step_backend_actual": None,
            "owner_bond_step_orchestrator_actual": None,
            "owner_bond_step_last_error": None,
            "owner_typed_direct_plan_static_refresh_attempts": 0,
            "owner_typed_direct_plan_static_refresh_accepts": 0,
            "owner_typed_direct_plan_static_refresh_fallbacks": 0,
            "owner_typed_direct_plan_static_refresh_failures": 0,
            "owner_local_optimize_calls": 0,
            "owner_local_optimize_accepts": 0,
            "owner_local_optimize_rejections": 0,
            "owner_local_optimize_failures": 0,
            "owner_local_optimize_seconds": 0.0,
            "owner_local_optimize_last_seconds": 0.0,
            "owner_local_optimize_backend_actual": None,
            "owner_local_optimize_rejected_reason": None,
            "owner_local_optimize_last_error": None,
            "owner_local_optimize_site_commits": 0,
            "owner_local_optimize_guess_cache_sets": 0,
            "owner_local_optimize_direct_cache_invalidations": 0,
            "owner_local_optimize_direct_payload_key_hits": 0,
            "owner_local_optimize_commit_actual": None,
            "owner_local_optimize_solve_actual": None,
            "owner_local_optimize_update_payload_actual": None,
            "owner_site_chain_backend_actual": None,
            "owner_site_chain_gets": 0,
            "owner_site_chain_sets": 0,
            "owner_site_chain_syncs": 0,
            "owner_site_chain_sync_sites": 0,
            "owner_site_chain_deferred_half_syncs": 0,
            "owner_site_chain_deferred_schedule_syncs": 0,
            "owner_local_grouped_solve_update_calls": 0,
            "owner_local_grouped_solve_update_accepts": 0,
            "owner_local_grouped_solve_update_rejections": 0,
            "owner_local_grouped_solve_update_failures": 0,
            "owner_local_grouped_solve_update_seconds": 0.0,
            "owner_local_grouped_solve_update_last_seconds": 0.0,
            "owner_local_grouped_solve_update_backend_actual": None,
            "owner_local_grouped_solve_update_rejected_reason": None,
            "owner_local_grouped_solve_update_last_error": None,
            "owner_local_grouped_direct_prepare_calls": 0,
            "owner_local_grouped_direct_prepare_accepts": 0,
            "owner_local_grouped_direct_prepare_failures": 0,
            "owner_local_grouped_direct_solve_update_calls": 0,
            "owner_local_grouped_direct_solve_update_accepts": 0,
            "owner_local_grouped_direct_raw_update_accepts": 0,
            "owner_local_grouped_direct_solve_update_failures": 0,
            "owner_local_grouped_direct_solve_update_fallbacks": 0,
            "owner_half_sweep_calls": 0,
            "owner_half_sweep_accepts": 0,
            "owner_half_sweep_failures": 0,
            "owner_half_sweep_bonds": 0,
            "owner_half_sweep_seconds": 0.0,
            "owner_half_sweep_last_seconds": 0.0,
            "owner_half_sweep_last_direction": None,
            "owner_half_sweep_backend_actual": None,
            "owner_half_sweep_last_error": None,
            "owner_sweep_schedule_calls": 0,
            "owner_sweep_schedule_accepts": 0,
            "owner_sweep_schedule_failures": 0,
            "owner_sweep_schedule_halves": 0,
            "owner_sweep_schedule_seconds": 0.0,
            "owner_sweep_schedule_last_seconds": 0.0,
            "owner_sweep_schedule_backend_actual": None,
            "owner_sweep_schedule_builder_actual": None,
            "owner_sweep_schedule_last_error": None,
            "owner_direct_family_environment_calls": 0,
            "owner_direct_family_environment_builds": 0,
            "owner_direct_family_environment_cache_hits": 0,
            "owner_direct_family_environment_cache_misses": 0,
            "owner_direct_family_environment_entries": 0,
            "owner_direct_family_environment_seconds": 0.0,
            "owner_direct_family_environment_last_seconds": 0.0,
            "owner_direct_family_environment_cache_size": 0,
            "owner_direct_family_environment_last_bond": None,
            "owner_direct_family_environment_last_error": None,
            "owner_direct_family_environment_prepared_payloads": 0,
            "owner_direct_family_environment_prepared_hits": 0,
            "owner_direct_family_environment_prepared_misses": 0,
            "owner_direct_family_environment_prepared_cache_size": 0,
            "family_environment_descriptor_binds": 0,
            "family_environment_descriptor_families": 0,
            "family_environment_requests": 0,
            "family_environment_cache_hits": 0,
            "family_environment_cache_misses": 0,
            "family_environment_local_pair_builds": 0,
            "family_environment_cache_size": 0,
            "family_environment_cpp_descriptor_installs": 0,
            "family_environment_cpp_descriptor_failures": 0,
            "family_environment_cpp_descriptor_payload_builds": 0,
            "family_environment_cpp_descriptor_payload_seconds": 0.0,
            "family_environment_cpp_qchem_descriptor_key": None,
            "family_environment_cpp_qchem_descriptor_families": 0,
            "cpp_named_raw_payload_plan_builds": 0,
            "cpp_named_raw_payload_plan_cache_hits": 0,
            "cpp_named_raw_payload_plan_refresh_calls": 0,
            "cpp_named_raw_payload_plan_refresh_seconds": 0.0,
            "cpp_named_raw_payload_plan_failures": 0,
            "cpp_named_raw_payload_plan_index_rebuilds": 0,
            "cpp_named_raw_payload_plan_index_rebuild_seconds": 0.0,
            "compiled_flat_matvec_builds": 0,
            "compiled_flat_matvec_cache_hits": 0,
            "compiled_flat_matvec_calls": 0,
            "compiled_flat_matvec_seconds": 0.0,
            "compact_plan_builds": 0,
            "compact_plan_build_seconds": 0.0,
            "compact_plan_cache_hits": 0,
            "dense_cpp_sweep_workspace_enabled": False,
            "dense_cpp_sweep_workspace_creates": 0,
            "dense_cpp_sweep_workspace_records": 0,
            "dense_cpp_sweep_workspace_binds": 0,
            "dense_cpp_sweep_workspace_bind_seconds": 0.0,
            "dense_cpp_sweep_workspace_boundary_binds": 0,
            "dense_cpp_sweep_workspace_boundary_bind_seconds": 0.0,
            "dense_cpp_sweep_workspace_static_w_hits": 0,
            "dense_cpp_sweep_workspace_bind_cache_hits": 0,
            "dense_cpp_sweep_workspace_solve_calls": 0,
            "dense_cpp_sweep_workspace_solve_seconds": 0.0,
            "dense_cpp_sweep_workspace_two_site_solve_calls": 0,
            "dense_cpp_sweep_workspace_two_site_solve_accepts": 0,
            "dense_cpp_sweep_workspace_two_site_solve_rejections": 0,
            "dense_cpp_sweep_workspace_two_site_solve_seconds": 0.0,
            "dense_cpp_sweep_workspace_two_site_static_w_reuses": 0,
            "dense_cpp_sweep_workspace_two_site_mpo_builds": 0,
            "dense_cpp_sweep_workspace_two_site_mps_builds": 0,
            "dense_cpp_sweep_workspace_failures": 0,
            "dense_cpp_sweep_workspace_last_error": None,
            "dense_cpp_tensor_primitive_calls": 0,
            "dense_cpp_tensor_primitive_seconds": 0.0,
            "dense_cpp_tensor_primitive_failures": 0,
            "dense_cpp_tensor_primitive_last_error": None,
            "dense_cpp_coarse_grain_mpo_calls": 0,
            "dense_cpp_coarse_grain_mpo_cache_hits": 0,
            "dense_cpp_coarse_grain_mps_calls": 0,
            "dense_cpp_environment_update_calls": 0,
            "dense_cpp_environment_update_seconds": 0.0,
            "dense_cpp_environment_update_failures": 0,
            "dense_local_operator_builds": 0,
            "dense_local_operator_reuses": 0,
            "dense_solve_local_calls": 0,
            "dense_solve_local_accepts": 0,
            "dense_solve_local_rejections": 0,
            "dense_solve_local_seconds": 0.0,
            "dense_solve_local_last_seconds": 0.0,
            "dense_operatorless_local_problem_binds": 0,
            "dense_operatorless_local_problem_solve_calls": 0,
            "dense_operatorless_local_problem_solve_accepts": 0,
            "dense_operatorless_local_problem_solve_rejections": 0,
            "dense_operatorless_local_problem_solve_seconds": 0.0,
            "dense_operatorless_local_problem_solve_last_seconds": 0.0,
            "dense_operatorless_local_problem_last_error": None,
            "dense_cpp_split_calls": 0,
            "dense_cpp_split_accepts": 0,
            "dense_cpp_split_failures": 0,
            "dense_cpp_split_seconds": 0.0,
            "dense_cpp_split_last_seconds": 0.0,
            "dense_cpp_split_last_error": None,
            "cpp_moving_environment_enabled": self._cpp_moving_environment is not None,
            "cpp_moving_environment_compact_plan_installs": 0,
            "cpp_moving_environment_compact_plan_records": 0,
            "cpp_moving_environment_compact_plan_replacements": 0,
            "cpp_moving_environment_compact_plan_davidson_calls": 0,
            "cpp_moving_environment_compact_plan_davidson_workspace_reuses": 0,
            "cpp_moving_environment_compact_plan_diagonal_calls": 0,
            "cpp_moving_environment_compact_plan_diagonal_cache_hits": 0,
            "cpp_moving_environment_grouped_table_installs": 0,
            "cpp_moving_environment_grouped_table_records": 0,
            "cpp_moving_environment_grouped_table_replacements": 0,
            "cpp_moving_environment_grouped_table_matvec_calls": 0,
            "cpp_moving_environment_grouped_table_davidson_calls": 0,
            "cpp_moving_environment_grouped_table_davidson_workspace_reuses": 0,
            "cpp_moving_environment_grouped_table_diagonal_calls": 0,
            "cpp_moving_environment_grouped_table_diagonal_cache_hits": 0,
            "cpp_moving_environment_site_split_flat_calls": 0,
            "cpp_moving_environment_site_split_flat_failures": 0,
            "cpp_moving_environment_site_split_flat_blocks": 0,
            "cpp_moving_environment_site_split_flat_sectors": 0,
            "cpp_moving_environment_site_split_flat_rows": 0,
            "cpp_moving_environment_site_split_flat_cols": 0,
            "cpp_moving_environment_site_split_flat_dim": 0,
            "cpp_moving_environment_site_update_flat_calls": 0,
            "cpp_moving_environment_site_update_flat_failures": 0,
            "cpp_moving_environment_site_update_flat_left_blocks": 0,
            "cpp_moving_environment_site_update_flat_right_blocks": 0,
            "cpp_moving_environment_site_update_flat_dim": 0,
            "cpp_moving_environment_site_update_backend": None,
            "cpp_moving_environment_site_update_flat_last_error": None,
            "cpp_moving_environment_solve_update_flat_calls": 0,
            "cpp_moving_environment_solve_update_flat_accepted": 0,
            "cpp_moving_environment_solve_update_flat_failures": 0,
            "cpp_moving_environment_solve_update_auto_calls": 0,
            "cpp_moving_environment_solve_update_backend": None,
            "cpp_moving_environment_solve_update_flat_last_error": None,
            "cpp_moving_environment_sweep_cursor_plan_calls": 0,
            "cpp_moving_environment_sweep_cursor_lr_calls": 0,
            "cpp_moving_environment_sweep_cursor_rl_calls": 0,
            "cpp_moving_environment_sweep_cursor_recenter_calls": 0,
            "cpp_moving_environment_sweep_cursor_steps": 0,
            "cpp_moving_environment_sweep_cursor_last_n_sites": 0,
            "cpp_moving_environment_sweep_cursor_last_steps": 0,
            "cpp_moving_environment_sweep_cursor_failures": 0,
            "cpp_moving_environment_direct_family_payload_records": 0,
            "cpp_moving_environment_direct_family_payload_installs": 0,
            "cpp_moving_environment_direct_family_payload_replacements": 0,
            "cpp_moving_environment_direct_family_payload_hits": 0,
            "cpp_moving_environment_direct_family_payload_misses": 0,
            "cpp_moving_environment_direct_family_payload_clears": 0,
            "cpp_moving_environment_direct_family_payload_cleared_entries": 0,
            "cpp_moving_environment_direct_family_route_plan_records": 0,
            "cpp_moving_environment_direct_family_route_plan_record_hits": 0,
            "cpp_moving_environment_direct_family_route_plan_record_misses": 0,
            "cpp_moving_environment_direct_family_route_plan_installs": 0,
            "cpp_moving_environment_direct_family_route_plan_payload_builds": 0,
            "cpp_moving_environment_direct_family_route_plan_last_entries": 0,
            "cpp_moving_environment_direct_family_route_plan_payload_seconds": 0.0,
            "cpp_moving_environment_direct_family_route_plan_last_payload_seconds": 0.0,
            "cpp_moving_environment_direct_family_payload_builder_records": 0,
            "cpp_moving_environment_direct_family_payload_builder_installs": 0,
            "cpp_moving_environment_direct_family_payload_builder_replacements": 0,
            "cpp_moving_environment_direct_family_payload_builder_prepare_calls": 0,
            "cpp_moving_environment_direct_family_payload_builder_builds": 0,
            "cpp_moving_environment_direct_family_payload_builder_cache_hits": 0,
            "cpp_moving_environment_direct_family_payload_builder_misses": 0,
            "cpp_moving_environment_direct_family_payload_builder_failures": 0,
            "cpp_moving_environment_direct_family_payload_builder_clears": 0,
            "cpp_moving_environment_direct_family_payload_builder_cleared_entries": 0,
            "cpp_moving_environment_direct_family_payload_builder_entries": 0,
            "cpp_moving_environment_direct_family_payload_builder_last_entries": 0,
            "cpp_moving_environment_direct_family_payload_builder_build_seconds": 0.0,
            "cpp_moving_environment_direct_family_payload_builder_last_build_seconds": 0.0,
            "cpp_moving_environment_direct_family_payload_assembler_calls": 0,
            "cpp_moving_environment_direct_family_payload_assembler_builds": 0,
            "cpp_moving_environment_direct_family_payload_assembler_families": 0,
            "cpp_moving_environment_direct_family_payload_assembler_pieces": 0,
            "cpp_moving_environment_direct_family_payload_assembler_merges": 0,
            "cpp_moving_environment_direct_family_payload_assembler_empty_pieces": 0,
            "cpp_moving_environment_direct_family_payload_assembler_failures": 0,
            "cpp_moving_environment_direct_family_payload_assembler_seconds": 0.0,
            "cpp_moving_environment_direct_family_payload_assembler_last_seconds": 0.0,
            "cpp_moving_environment_direct_family_payload_assembler_last_error": None,
            "cpp_moving_environment_direct_family_piece_builder_plan_calls": 0,
            "cpp_moving_environment_direct_family_piece_builder_plan_builds": 0,
            "cpp_moving_environment_direct_family_piece_builder_plan_families": 0,
            "cpp_moving_environment_direct_family_piece_builder_plan_pieces": 0,
            "cpp_moving_environment_direct_family_piece_builder_plan_entries": 0,
            "cpp_moving_environment_direct_family_piece_builder_plan_empty_pieces": 0,
            "cpp_moving_environment_direct_family_piece_builder_plan_failures": 0,
            "cpp_moving_environment_direct_family_piece_builder_plan_seconds": 0.0,
            "cpp_moving_environment_direct_family_piece_builder_plan_last_seconds": 0.0,
            "cpp_moving_environment_direct_family_piece_builder_plan_last_error": None,
            "cpp_moving_environment_direct_family_phased_piece_plan_records": 0,
            "cpp_moving_environment_direct_family_phased_piece_plan_installs": 0,
            "cpp_moving_environment_direct_family_phased_piece_plan_replacements": 0,
            "cpp_moving_environment_direct_family_phased_piece_plan_prepare_calls": 0,
            "cpp_moving_environment_direct_family_phased_piece_plan_cache_hits": 0,
            "cpp_moving_environment_direct_family_phased_piece_plan_misses": 0,
            "cpp_moving_environment_direct_family_phased_piece_plan_failures": 0,
            "cpp_moving_environment_direct_family_phased_piece_plan_last_error": None,
            "cpp_moving_environment_direct_family_phased_family_plan_records": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_installs": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_replacements": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_prepare_calls": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_cache_hits": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_misses": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_failures": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_calls": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_families": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_pieces": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_entries": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_empty_pieces": 0,
            "cpp_moving_environment_direct_family_phased_family_plan_last_error": None,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_records": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_installs": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_replacements": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_prepare_calls": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_cache_hits": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_misses": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_failures": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_calls": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_families": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_pieces": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_entries": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_empty_pieces": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_factory_calls": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_installs": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_uses": 0,
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_last_error": None,
            "cpp_moving_environment_owner_bond_step_runner_calls": 0,
            "cpp_moving_environment_owner_bond_step_runner_accepted": 0,
            "cpp_moving_environment_owner_bond_step_runner_failures": 0,
            "cpp_moving_environment_owner_bond_step_runner_payload_prepares": 0,
            "cpp_moving_environment_owner_bond_step_runner_environment_moves": 0,
            "cpp_moving_environment_owner_bond_step_runner_environment_fallbacks": 0,
            "cpp_moving_environment_owner_bond_step_runner_assign_calls": 0,
            "cpp_moving_environment_owner_bond_step_runner_assign_skips": 0,
            "cpp_moving_environment_owner_bond_step_runner_seconds": 0.0,
            "cpp_moving_environment_owner_bond_step_runner_last_seconds": 0.0,
            "cpp_moving_environment_owner_bond_step_runner_payload_seconds": 0.0,
            "cpp_moving_environment_owner_bond_step_runner_payload_last_seconds": 0.0,
            "cpp_moving_environment_owner_bond_step_record_records": 0,
            "cpp_moving_environment_owner_bond_step_record_installs": 0,
            "cpp_moving_environment_owner_bond_step_record_replacements": 0,
            "cpp_moving_environment_owner_bond_step_record_hits": 0,
            "cpp_moving_environment_owner_bond_step_record_misses": 0,
            "cpp_moving_environment_owner_bond_step_record_clears": 0,
            "cpp_moving_environment_owner_bond_step_record_cleared_entries": 0,
            "cpp_moving_environment_owner_bond_step_record_last_error": None,
            "cpp_moving_environment_owner_typed_bond_step_record_records": 0,
            "cpp_moving_environment_owner_typed_bond_step_record_installs": 0,
            "cpp_moving_environment_owner_typed_bond_step_record_replacements": 0,
            "cpp_moving_environment_owner_typed_bond_step_record_hits": 0,
            "cpp_moving_environment_owner_typed_bond_step_record_misses": 0,
            "cpp_moving_environment_owner_typed_bond_step_record_clears": 0,
            "cpp_moving_environment_owner_typed_bond_step_record_cleared_entries": 0,
            "cpp_moving_environment_owner_typed_bond_step_environment_record_prepares": 0,
            "cpp_moving_environment_owner_typed_bond_step_environment_record_consumes": 0,
            "cpp_moving_environment_owner_typed_bond_step_python_prepare_calls": 0,
            "cpp_moving_environment_owner_typed_bond_step_python_move_calls": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_record_installs": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_calls": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_accepts": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_empty": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_failures": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_updates": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_update_misses": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_update_failures": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_calls": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_accepts": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_empty": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_failures": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_calls": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_accepts": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_empty": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_failures": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_calls": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_accepts": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_links": 0,
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_failures": 0,
            "cpp_moving_environment_direct_family_revision_state_updates": 0,
            "cpp_moving_environment_direct_family_revision_state_failures": 0,
            "cpp_moving_environment_direct_family_revision_cache_key_builds": 0,
            "cpp_moving_environment_direct_family_revision_cache_key_failures": 0,
            "cpp_moving_environment_direct_family_cpp_key_bundle_builds": 0,
            "cpp_moving_environment_direct_family_cpp_key_bundle_failures": 0,
            "cpp_moving_environment_direct_family_revision_state_last_error": None,
            "cpp_moving_environment_owner_typed_bond_step_record_last_error": None,
            "cpp_moving_environment_owner_typed_half_sweep_plan_records": 0,
            "cpp_moving_environment_owner_typed_half_sweep_plan_installs": 0,
            "cpp_moving_environment_owner_typed_half_sweep_plan_replacements": 0,
            "cpp_moving_environment_owner_typed_half_sweep_plan_hits": 0,
            "cpp_moving_environment_owner_typed_half_sweep_plan_misses": 0,
            "cpp_moving_environment_owner_typed_half_sweep_plan_runs": 0,
            "cpp_moving_environment_owner_typed_half_sweep_plan_bonds": 0,
            "cpp_moving_environment_owner_typed_half_sweep_template_plan_installs": 0,
            "cpp_moving_environment_owner_typed_half_sweep_template_plan_bonds": 0,
            "cpp_moving_environment_owner_typed_half_sweep_template_local_records": 0,
            "cpp_moving_environment_owner_typed_half_sweep_template_step_records": 0,
            "cpp_moving_environment_owner_typed_half_sweep_plan_last_error": None,
            "cpp_moving_environment_owner_sweep_schedule_plan_records": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_installs": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_replacements": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_alternating_installs": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_alternating_expanded_halves": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_noise_sets": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_noise_set_failures": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_hits": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_misses": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_runs": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_halves": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_converged": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_history_rows": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_configures": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_runs": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_skips": 0,
            "cpp_moving_environment_owner_sweep_schedule_plan_seconds": 0.0,
            "cpp_moving_environment_owner_sweep_schedule_plan_last_seconds": 0.0,
            "cpp_moving_environment_owner_sweep_schedule_plan_last_error": None,
            "cpp_moving_environment_owner_local_optimize_runner_calls": 0,
            "cpp_moving_environment_owner_local_optimize_runner_accepted": 0,
            "cpp_moving_environment_owner_local_optimize_runner_rejections": 0,
            "cpp_moving_environment_owner_local_optimize_runner_failures": 0,
            "cpp_moving_environment_owner_local_optimize_runner_seconds": 0.0,
            "cpp_moving_environment_owner_local_optimize_runner_last_seconds": 0.0,
            "cpp_moving_environment_owner_local_optimize_runner_last_error": None,
            "cpp_moving_environment_owner_local_optimize_runner_last_reason": None,
            "cpp_moving_environment_owner_local_optimize_record_noise_sets": 0,
            "cpp_moving_environment_owner_local_optimize_native_merge_calls": 0,
            "cpp_moving_environment_owner_local_optimize_native_merge_accepted": 0,
            "cpp_moving_environment_owner_local_optimize_native_merge_failures": 0,
            "cpp_moving_environment_owner_local_optimize_native_noise_injections": 0,
            "cpp_moving_environment_owner_local_optimize_native_noise_blocks": 0,
            "cpp_moving_environment_owner_local_optimize_bridge_merge_calls": 0,
            "cpp_moving_environment_owner_local_optimize_boundary_stack_reads": 0,
            "cpp_moving_environment_owner_local_optimize_boundary_bridge_calls": 0,
            "cpp_moving_environment_owner_local_problem_bind_owner_calls": 0,
            "cpp_moving_environment_owner_local_problem_bind_set_bond_fallbacks": 0,
            "cpp_moving_environment_owner_local_grouped_solve_update_calls": 0,
            "cpp_moving_environment_owner_local_grouped_solve_update_accepted": 0,
            "cpp_moving_environment_owner_local_grouped_solve_update_rejections": 0,
            "cpp_moving_environment_owner_local_grouped_solve_update_failures": 0,
            "cpp_moving_environment_owner_local_grouped_solve_update_seconds": 0.0,
            "cpp_moving_environment_owner_local_grouped_solve_update_last_seconds": 0.0,
            "cpp_moving_environment_owner_local_grouped_solve_update_last_error": None,
            "cpp_moving_environment_owner_local_grouped_solve_update_last_reason": None,
            "cpp_moving_environment_owner_half_sweep_runner_calls": 0,
            "cpp_moving_environment_owner_half_sweep_runner_accepted": 0,
            "cpp_moving_environment_owner_half_sweep_runner_failures": 0,
            "cpp_moving_environment_owner_half_sweep_runner_bonds": 0,
            "cpp_moving_environment_owner_half_sweep_runner_seconds": 0.0,
            "cpp_moving_environment_owner_half_sweep_runner_last_seconds": 0.0,
            "cpp_moving_environment_owner_half_sweep_runner_last_direction": None,
            "compact_plan_bond_slot_stores": 0,
            "compact_plan_bond_slot_hits": 0,
            "compact_plan_bond_slot_refreshes": 0,
            "compact_plan_bond_slot_refresh_failures": 0,
            "compact_plan_refreshes": 0,
            "compact_plan_refresh_seconds": 0.0,
            "compact_plan_refresh_failures": 0,
            "compact_plan_failures": 0,
            "compact_renormalized_table_builds": 0,
            "compact_renormalized_table_build_seconds": 0.0,
            "compact_renormalized_table_cache_hits": 0,
            "compact_renormalized_table_bond_slot_stores": 0,
            "compact_renormalized_table_bond_slot_hits": 0,
            "compact_renormalized_table_bond_slot_reuses": 0,
            "compact_renormalized_table_cpp_block_constructor_builds": 0,
            "compact_renormalized_table_python_stack_constructor_builds": 0,
            "compact_renormalized_table_refreshes": 0,
            "compact_renormalized_table_refresh_seconds": 0.0,
            "compact_renormalized_table_refresh_failures": 0,
            "compact_renormalized_table_cpp_block_refreshes": 0,
            "compact_renormalized_table_python_stack_refreshes": 0,
            "compact_renormalized_table_diagonal_calls": 0,
            "compact_renormalized_table_diagonal_seconds": 0.0,
            "compact_renormalized_table_diagonal_fallbacks": 0,
            "compact_renormalized_table_failures": 0,
            "compact_plan_matvec_calls": 0,
            "compact_plan_matvec_seconds": 0.0,
            "compact_plan_validation_calls": 0,
            "compact_plan_validation_failures": 0,
            "compact_block_table_builds": 0,
            "compact_block_table_build_seconds": 0.0,
            "compact_block_table_cache_hits": 0,
            "compact_block_table_failures": 0,
            "cpp_grouped_renormalized_table_builds": 0,
            "cpp_grouped_renormalized_table_build_seconds": 0.0,
            "cpp_grouped_renormalized_table_failures": 0,
            "cpp_grouped_renormalized_table_refreshes": 0,
            "cpp_grouped_renormalized_table_refresh_seconds": 0.0,
            "cpp_grouped_renormalized_table_refresh_failures": 0,
            "cpp_grouped_renormalized_table_slot_reuses": 0,
            "cpp_grouped_renormalized_table_fast_refreshes": 0,
            "cpp_grouped_renormalized_table_rebuild_refreshes": 0,
            "cpp_grouped_renormalized_table_rebuild_in_place_refreshes": 0,
            "cpp_grouped_renormalized_table_bond_slot_reuses": 0,
            "cpp_grouped_renormalized_table_structural_slot_reuses": 0,
            "cpp_renormalized_table_builds": 0,
            "cpp_sparse_renormalized_table_builds": 0,
            "cpp_renormalized_table_failures": 0,
            "cpp_renormalized_table_validation_calls": 0,
            "cpp_renormalized_table_validation_failures": 0,
            "cpp_renormalized_table_diagonal_calls": 0,
            "cpp_renormalized_table_diagonal_seconds": 0.0,
            "cpp_renormalized_table_matvec_calls": 0,
            "cpp_renormalized_table_matvec_seconds": 0.0,
            "cpp_block_table_builds": 0,
            "cpp_block_table_failures": 0,
            "cpp_block_matvec_calls": 0,
            "cpp_block_matvec_seconds": 0.0,
            "cpp_block_matvec_failures": 0,
            "cpp_davidson_attempts": 0,
            "cpp_davidson_calls": 0,
            "cpp_davidson_workspace_reuses": 0,
            "cpp_davidson_seconds": 0.0,
            "cpp_davidson_failures": 0,
            "cpp_davidson_rejected": 0,
            "cpp_solution_validation_calls": 0,
            "cpp_solution_validation_failures": 0,
            "environment_updates": {},
            "environment_stack_updates": {},
            "direct_family_cache_invalidations": 0,
            "local_operator_reuses": 0,
            "operatorless_local_problem_binds": 0,
            "operatorless_local_problem_rejections": 0,
            "operatorless_local_problem_solve_calls": 0,
            "operatorless_local_problem_solve_accepts": 0,
            "operatorless_local_problem_solve_rejections": 0,
            "owner_operatorless_local_problem_binds": 0,
            "owner_operatorless_local_problem_rejections": 0,
            "owner_local_problem_bind_backend_actual": None,
        }
        if self._cpp_qchem_family_descriptor_key:
            self.moving_profile_stats[
                "family_environment_cpp_qchem_descriptor_key"
            ] = str(self._cpp_qchem_family_descriptor_key)
            self.moving_profile_stats[
                "family_environment_cpp_qchem_descriptor_families"
            ] = int(len(self._cpp_qchem_family_descriptor_names))

    def bind_sweep_stacks(
        self,
        *,
        left_environments=None,
        right_environments=None,
        complementary_left_environments=None,
        complementary_right_environments=None,
        complementary_operator_mpos=None,
        direct_family_cache_invalidator=None,
        direct_family_revision_ref=None,
        direct_family_cache_maps=None,
    ):
        self.left_environments = left_environments
        self.right_environments = right_environments
        self.complementary_left_environments = dict(
            complementary_left_environments or {}
        )
        self.complementary_right_environments = dict(
            complementary_right_environments or {}
        )
        if complementary_operator_mpos is not None:
            self.complementary_operator_mpos = {
                str(name): factors
                for name, factors in complementary_operator_mpos.items()
            }
            self._family_local_pair_cache.clear()
            self._family_environment_cache.clear()
            self._family_environment_descriptor = tuple(
                (
                    str(name),
                    int(len(factors)),
                    id(factors),
                )
                for name, factors in sorted(
                    self.complementary_operator_mpos.items(),
                    key=lambda item: str(item[0]),
                )
            )
        self._direct_family_cache_invalidator = direct_family_cache_invalidator
        self._direct_family_revision_ref = direct_family_revision_ref
        self._direct_family_cache_maps = tuple(direct_family_cache_maps or ())
        if direct_family_revision_ref is not None:
            try:
                self.direct_family_revision = int(direct_family_revision_ref[0])
            except Exception:
                self.direct_family_revision = 0
        self.moving_profile_stats["sweep_stack_bindings"] = int(
            self.moving_profile_stats.get("sweep_stack_bindings", 0)
        ) + 1
        self.moving_profile_stats["sweep_stack_left_bound"] = (
            left_environments is not None
        )
        self.moving_profile_stats["sweep_stack_right_bound"] = (
            right_environments is not None
        )
        self.moving_profile_stats["sweep_stack_family_count"] = int(
            len(self.complementary_left_environments)
            + len(self.complementary_right_environments)
        )
        self.moving_profile_stats["family_environment_descriptor_binds"] = int(
            self.moving_profile_stats.get(
                "family_environment_descriptor_binds",
                0,
            )
        ) + 1
        self.moving_profile_stats["family_environment_descriptor_families"] = int(
            len(self.complementary_operator_mpos)
        )
        self._install_cpp_family_mpo_descriptor()
        return self

    def _install_cpp_family_mpo_descriptor(self):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not self.complementary_operator_mpos
            or not hasattr(owner, "install_family_mpo_descriptor")
            or not hasattr(owner, "set_environment_stack")
        ):
            self._cpp_family_mpo_descriptor_key = None
            self._cpp_family_mpo_descriptor_names = ()
            return False
        names = []
        left_keys = []
        right_keys = []
        factors = []
        try:
            for name, family_factors in sorted(
                self.complementary_operator_mpos.items(),
                key=lambda item: str(item[0]),
            ):
                name = str(name)
                left_stack = self.complementary_left_environments.get(name)
                right_stack = self.complementary_right_environments.get(name)
                if left_stack is None or right_stack is None:
                    continue
                left_key = self._cpp_environment_stack_key("left", f"family:{name}")
                right_key = self._cpp_environment_stack_key("right", f"family:{name}")
                owner.set_environment_stack(left_key, tuple(left_stack))
                owner.set_environment_stack(right_key, tuple(right_stack))
                names.append(name)
                left_keys.append(left_key)
                right_keys.append(right_key)
                factors.append(family_factors)
            if not names:
                self._cpp_family_mpo_descriptor_key = None
                self._cpp_family_mpo_descriptor_names = ()
                return False
            key = f"family-mpo-descriptor:{id(self)}"
            names_tuple = tuple(names)
            left_keys_tuple = tuple(left_keys)
            right_keys_tuple = tuple(right_keys)
            factors_tuple = tuple(factors)
            installed_from_owned = False
            if (
                hasattr(owner, "install_owned_family_mpos")
                and hasattr(owner, "install_family_mpo_descriptor_from_owned")
            ):
                option_owned_key = self._cpp_owned_family_mpo_key
                option_owned_names = tuple(self._cpp_owned_family_mpo_names or ())
                owned_key = (
                    str(option_owned_key)
                    if option_owned_key
                    else f"owned-family-mpos:{id(self)}"
                )
                try:
                    if option_owned_key and option_owned_names == names_tuple:
                        owner.install_family_mpo_descriptor_from_owned(
                            key,
                            owned_key,
                            names_tuple,
                            left_keys_tuple,
                            right_keys_tuple,
                        )
                    else:
                        owner.install_owned_family_mpos(
                            owned_key,
                            names_tuple,
                            factors_tuple,
                        )
                        self._cpp_owned_family_mpo_key = owned_key
                        self._cpp_owned_family_mpo_names = names_tuple
                        self.moving_profile_stats[
                            "family_environment_cpp_owned_mpo_installs"
                        ] = int(
                            self.moving_profile_stats.get(
                                "family_environment_cpp_owned_mpo_installs",
                                0,
                            )
                        ) + 1
                        owner.install_family_mpo_descriptor_from_owned(
                            key,
                            owned_key,
                            names_tuple,
                            left_keys_tuple,
                            right_keys_tuple,
                        )
                    installed_from_owned = True
                    self.moving_profile_stats[
                        "family_environment_cpp_owned_descriptor_installs"
                    ] = int(
                        self.moving_profile_stats.get(
                            "family_environment_cpp_owned_descriptor_installs",
                            0,
                        )
                    ) + 1
                except Exception as exc:
                    self.moving_profile_stats[
                        "family_environment_cpp_owned_descriptor_failures"
                    ] = int(
                        self.moving_profile_stats.get(
                            "family_environment_cpp_owned_descriptor_failures",
                            0,
                        )
                    ) + 1
                    self.moving_profile_stats[
                        "family_environment_cpp_owned_descriptor_last_error"
                    ] = str(exc)
            if not installed_from_owned:
                owner.install_family_mpo_descriptor(
                    key,
                    names_tuple,
                    left_keys_tuple,
                    right_keys_tuple,
                    factors_tuple,
                )
            self._cpp_family_mpo_descriptor_key = key
            self._cpp_family_mpo_descriptor_names = names_tuple
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "family_environment_cpp_descriptor_installs"
            ] = int(
                self.moving_profile_stats.get(
                    "family_environment_cpp_descriptor_installs",
                    0,
                )
            ) + 1
            return True
        except Exception as exc:
            self._cpp_family_mpo_descriptor_key = None
            self._cpp_family_mpo_descriptor_names = ()
            self.moving_profile_stats[
                "family_environment_cpp_descriptor_failures"
            ] = int(
                self.moving_profile_stats.get(
                    "family_environment_cpp_descriptor_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "family_environment_cpp_descriptor_last_error"
            ] = str(exc)
            return False

    def uses_cpp_family_mpo_descriptor(self):
        owner = self._cpp_moving_environment
        return bool(
            self._cpp_family_mpo_descriptor_key
            and owner is not None
            and hasattr(owner, "build_named_payload_from_family_mpo_descriptor")
        )

    def _build_cpp_named_payload_from_family_descriptor(self, plan, bond, layout):
        owner = self._cpp_moving_environment
        if not self.uses_cpp_family_mpo_descriptor():
            return None
        start = time.perf_counter()
        try:
            builder = owner.build_named_payload_from_family_mpo_descriptor(
                plan,
                self._cpp_family_mpo_descriptor_key,
                int(bond),
                tuple(layout),
            )
        except Exception as exc:
            self.moving_profile_stats[
                "family_environment_cpp_descriptor_payload_failures"
            ] = int(
                self.moving_profile_stats.get(
                    "family_environment_cpp_descriptor_payload_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "family_environment_cpp_descriptor_payload_last_error"
            ] = str(exc)
            return None
        elapsed = time.perf_counter() - start
        self._sync_cpp_moving_environment_stats()
        self.moving_profile_stats[
            "family_environment_cpp_descriptor_payload_builds"
        ] = int(
            self.moving_profile_stats.get(
                "family_environment_cpp_descriptor_payload_builds",
                0,
            )
        ) + 1
        self.moving_profile_stats[
            "family_environment_cpp_descriptor_payload_seconds"
        ] = float(
            self.moving_profile_stats.get(
                "family_environment_cpp_descriptor_payload_seconds",
                0.0,
            )
        ) + elapsed
        self.moving_profile_stats[
            "family_environment_cpp_descriptor_payload_last_seconds"
        ] = elapsed
        return builder

    def family_environments_for_bond(self, bond):
        bond = int(bond)
        stats = self.moving_profile_stats
        stats["family_environment_requests"] = int(
            stats.get("family_environment_requests", 0)
        ) + 1
        if (
            not self.complementary_operator_mpos
            or not self.complementary_left_environments
            or not self.complementary_right_environments
        ):
            stats["family_environment_last_error"] = "unbound_family_stacks"
            return None

        rows = []
        for name, factors in sorted(
            self.complementary_operator_mpos.items(),
            key=lambda item: str(item[0]),
        ):
            left_stack = self.complementary_left_environments.get(name)
            right_stack = self.complementary_right_environments.get(name)
            if left_stack is None or right_stack is None:
                continue
            try:
                left_env = left_stack[-1]
                right_env = right_stack[-1]
                W_left = factors[bond]
                W_right = factors[bond + 1]
            except (IndexError, KeyError, TypeError):
                continue
            pair_key = (str(name), bond, id(W_left), id(W_right))
            local_pair = self._family_local_pair_cache.get(pair_key)
            if local_pair is None:
                local_pair = [W_left, W_right]
                self._family_local_pair_cache[pair_key] = local_pair
                stats["family_environment_local_pair_builds"] = int(
                    stats.get("family_environment_local_pair_builds", 0)
                ) + 1
            rows.append((str(name), left_env, local_pair, right_env))
        if not rows:
            stats["family_environment_last_error"] = "empty_family_environment"
            return None

        cache_key = (
            "family_environment",
            bond,
            tuple(
                (
                    name,
                    id(left_env),
                    id(local_pair[0]),
                    id(local_pair[1]),
                    id(right_env),
                )
                for name, left_env, local_pair, right_env in rows
            ),
        )
        cached = self._family_environment_cache.get(cache_key)
        if cached is not None:
            stats["family_environment_cache_hits"] = int(
                stats.get("family_environment_cache_hits", 0)
            ) + 1
            stats["family_environment_cache_size"] = int(
                len(self._family_environment_cache)
            )
            stats["family_environment_last_error"] = None
            return cached

        envs = {
            name: (left_env, local_pair, right_env)
            for name, left_env, local_pair, right_env in rows
        }
        if len(self._family_environment_cache) >= int(
            self._option_value(
                self.matvec_options,
                "moving_environment_family_environment_cache_max_entries",
                128,
            )
        ):
            self._family_environment_cache.clear()
            stats["family_environment_cache_clears"] = int(
                stats.get("family_environment_cache_clears", 0)
            ) + 1
        self._family_environment_cache[cache_key] = envs
        stats["family_environment_cache_misses"] = int(
            stats.get("family_environment_cache_misses", 0)
        ) + 1
        stats["family_environment_cache_size"] = int(
            len(self._family_environment_cache)
        )
        stats["family_environment_last_error"] = None
        return envs

    def left_environment(self):
        if not self.left_environments:
            raise RuntimeError("MovingEnvironment has no active left stack")
        return self.left_environments[-1]

    def right_environment(self):
        if not self.right_environments:
            raise RuntimeError("MovingEnvironment has no active right stack")
        return self.right_environments[-1]

    def update_left_stack(self, W, A, B, *, stack=None, stack_name="hamiltonian"):
        stack = self.left_environments if stack is None else stack
        if not stack:
            raise RuntimeError("cannot update an empty left environment stack")
        start = time.perf_counter()
        self._sync_cpp_environment_stack("left", stack_name, stack)
        self._environment_advance_slot_key = (
            "left",
            str(stack_name),
            int(len(stack) - 1),
        )
        try:
            new_env = self.compiled_backend.update_left_environment(W, A, stack[-1], B)
        finally:
            self._environment_advance_slot_key = None
        stack.append(new_env)
        self._cpp_environment_stack_push("left", stack_name, new_env)
        self.moving_profile_stats["environment_update_backend"] = getattr(
            self,
            "_last_environment_update_backend",
            "python_contract",
        )
        self._record_environment_update("update_left", time.perf_counter() - start)
        self._record_environment_stack_update(
            "push_left",
            stack_name,
            len(stack),
        )
        return new_env

    def update_right_stack(self, W, A, B, *, stack=None, stack_name="hamiltonian"):
        stack = self.right_environments if stack is None else stack
        if not stack:
            raise RuntimeError("cannot update an empty right environment stack")
        start = time.perf_counter()
        self._sync_cpp_environment_stack("right", stack_name, stack)
        self._environment_advance_slot_key = (
            "right",
            str(stack_name),
            int(len(stack) - 1),
        )
        try:
            new_env = self.compiled_backend.update_right_environment(W, A, stack[-1], B)
        finally:
            self._environment_advance_slot_key = None
        stack.append(new_env)
        self._cpp_environment_stack_push("right", stack_name, new_env)
        self.moving_profile_stats["environment_update_backend"] = getattr(
            self,
            "_last_environment_update_backend",
            "python_contract",
        )
        self._record_environment_update("update_right", time.perf_counter() - start)
        self._record_environment_stack_update(
            "push_right",
            stack_name,
            len(stack),
        )
        return new_env

    def pop_left_stack(self, *, stack=None, stack_name="hamiltonian"):
        stack = self.left_environments if stack is None else stack
        self._cpp_environment_stack_pop("left", stack_name, stack=stack)
        value = stack.pop()
        self._record_environment_stack_update("pop_left", stack_name, len(stack))
        return value

    def pop_right_stack(self, *, stack=None, stack_name="hamiltonian"):
        stack = self.right_environments if stack is None else stack
        self._cpp_environment_stack_pop("right", stack_name, stack=stack)
        value = stack.pop()
        self._record_environment_stack_update("pop_right", stack_name, len(stack))
        return value

    def _cpp_environment_stack_enabled(self):
        owner = self._cpp_moving_environment
        if owner is None:
            return False
        if not bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_environment_stack_owner",
                True,
            )
        ):
            return False
        return (
            hasattr(owner, "set_environment_stack")
            and hasattr(owner, "environment_stack_depth")
        )

    @staticmethod
    def _cpp_environment_stack_key(direction, stack_name):
        return f"environment-stack:{direction}:{stack_name}"

    def _cpp_environment_stack_apply(
        self,
        direction,
        stack_name,
        action,
        *,
        stack=None,
        value=None,
    ):
        if not self._cpp_environment_stack_enabled():
            return None
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "environment_stack_apply"):
            return None
        key = self._cpp_environment_stack_key(direction, stack_name)
        stats = self.moving_profile_stats
        values = None if stack is None else tuple(stack)
        try:
            result = owner.environment_stack_apply(
                key,
                str(action),
                values,
                value,
            )
            did_sync, did_push, did_pop, did_replace, _depth, _popped = result
            if bool(did_sync):
                stats["cpp_environment_stack_resets"] = int(
                    stats.get("cpp_environment_stack_resets", 0)
                ) + 1
            if bool(did_push):
                stats["cpp_environment_stack_pushes"] = int(
                    stats.get("cpp_environment_stack_pushes", 0)
                ) + 1
            if bool(did_pop):
                stats["cpp_environment_stack_pops"] = int(
                    stats.get("cpp_environment_stack_pops", 0)
                ) + 1
            if bool(did_replace):
                stats["cpp_environment_stack_resets"] = int(
                    stats.get("cpp_environment_stack_resets", 0)
                ) + 1
            stats["cpp_environment_stack_apply_calls"] = int(
                stats.get("cpp_environment_stack_apply_calls", 0)
            ) + 1
            action_name = str(action)
            if action_name == "sync":
                stats["cpp_environment_stack_apply_syncs"] = int(
                    stats.get("cpp_environment_stack_apply_syncs", 0)
                ) + 1
            elif action_name == "push":
                stats["cpp_environment_stack_apply_pushes"] = int(
                    stats.get("cpp_environment_stack_apply_pushes", 0)
                ) + 1
            elif action_name == "pop":
                stats["cpp_environment_stack_apply_pops"] = int(
                    stats.get("cpp_environment_stack_apply_pops", 0)
                ) + 1
            elif action_name == "replace":
                stats["cpp_environment_stack_apply_replaces"] = int(
                    stats.get("cpp_environment_stack_apply_replaces", 0)
                ) + 1
            self._sync_cpp_moving_environment_stats()
            stats["cpp_environment_stack_backend_actual"] = "cpp_moving_environment"
            return result
        except Exception as exc:
            stats["cpp_environment_stack_failures"] = int(
                stats.get("cpp_environment_stack_failures", 0)
            ) + 1
            stats["cpp_environment_stack_last_error"] = str(exc)
            return None

    def _sync_cpp_environment_stack(self, direction, stack_name, stack):
        if not self._cpp_environment_stack_enabled():
            return False
        applied = self._cpp_environment_stack_apply(
            direction,
            stack_name,
            "sync",
            stack=stack,
        )
        if applied is not None:
            return True
        owner = self._cpp_moving_environment
        key = self._cpp_environment_stack_key(direction, stack_name)
        stats = self.moving_profile_stats
        try:
            depth = int(owner.environment_stack_depth(key))
            if depth == int(len(stack)):
                return True
            owner.set_environment_stack(key, tuple(stack))
            stats["cpp_environment_stack_resets"] = int(
                stats.get("cpp_environment_stack_resets", 0)
            ) + 1
            self._sync_cpp_moving_environment_stats()
            stats["cpp_environment_stack_backend_actual"] = "cpp_moving_environment"
            return True
        except Exception as exc:
            stats["cpp_environment_stack_failures"] = int(
                stats.get("cpp_environment_stack_failures", 0)
            ) + 1
            stats["cpp_environment_stack_last_error"] = str(exc)
            return False

    def _cpp_environment_stack_push(self, direction, stack_name, value):
        if not self._cpp_environment_stack_enabled():
            return False
        applied = self._cpp_environment_stack_apply(
            direction,
            stack_name,
            "push",
            value=value,
        )
        if applied is not None:
            return True
        owner = self._cpp_moving_environment
        key = self._cpp_environment_stack_key(direction, stack_name)
        stats = self.moving_profile_stats
        try:
            owner.environment_stack_push(key, value)
            stats["cpp_environment_stack_pushes"] = int(
                stats.get("cpp_environment_stack_pushes", 0)
            ) + 1
            self._sync_cpp_moving_environment_stats()
            stats["cpp_environment_stack_backend_actual"] = "cpp_moving_environment"
            return True
        except Exception as exc:
            stats["cpp_environment_stack_failures"] = int(
                stats.get("cpp_environment_stack_failures", 0)
            ) + 1
            stats["cpp_environment_stack_last_error"] = str(exc)
            return False

    def _cpp_environment_stack_pop(self, direction, stack_name, *, stack=None):
        if not self._cpp_environment_stack_enabled():
            return False
        applied = self._cpp_environment_stack_apply(
            direction,
            stack_name,
            "pop",
            stack=stack,
        )
        if applied is not None:
            return True
        owner = self._cpp_moving_environment
        key = self._cpp_environment_stack_key(direction, stack_name)
        stats = self.moving_profile_stats
        try:
            owner.environment_stack_pop(key)
            stats["cpp_environment_stack_pops"] = int(
                stats.get("cpp_environment_stack_pops", 0)
            ) + 1
            self._sync_cpp_moving_environment_stats()
            stats["cpp_environment_stack_backend_actual"] = "cpp_moving_environment"
            return True
        except Exception as exc:
            stats["cpp_environment_stack_failures"] = int(
                stats.get("cpp_environment_stack_failures", 0)
            ) + 1
            stats["cpp_environment_stack_last_error"] = str(exc)
            return False

    def _cpp_environment_stack_seed_direct(self, direction, value):
        if not self._cpp_environment_stack_enabled():
            return False
        applied = self._cpp_environment_stack_apply(
            direction,
            "direct",
            "replace",
            value=value,
        )
        if applied is not None:
            return True
        owner = self._cpp_moving_environment
        key = self._cpp_environment_stack_key(direction, "direct")
        stats = self.moving_profile_stats
        try:
            owner.set_environment_stack(key, (value,))
            stats["cpp_environment_stack_resets"] = int(
                stats.get("cpp_environment_stack_resets", 0)
            ) + 1
            self._sync_cpp_moving_environment_stats()
            stats["cpp_environment_stack_backend_actual"] = "cpp_moving_environment"
            return True
        except Exception as exc:
            stats["cpp_environment_stack_failures"] = int(
                stats.get("cpp_environment_stack_failures", 0)
            ) + 1
            stats["cpp_environment_stack_last_error"] = str(exc)
            return False

    def _cpp_environment_stack_replace_direct(self, direction, value):
        return self._cpp_environment_stack_seed_direct(direction, value)

    def prepare_cpp_bond_environment_step(
        self,
        *,
        sweep_direction,
        bond,
        environment_direction,
        update_specs,
        pop_specs,
        store=True,
    ):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "bond_step_update_and_environment_auto"):
            self._pending_cpp_bond_environment_step = None
            return False if store else None
        if not bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_bond_step_transaction",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        ):
            self._pending_cpp_bond_environment_step = None
            return False if store else None
        update_rows = []
        update_records = []
        for stack_name, stack, W, site_selector in tuple(update_specs or ()):
            if not stack:
                self._pending_cpp_bond_environment_step = None
                return False if store else None
            update_records.append((str(stack_name), stack, str(environment_direction)))
            update_rows.append(
                (
                    self._cpp_environment_stack_key(
                        str(environment_direction),
                        stack_name,
                    ),
                    W,
                    str(site_selector),
                    stack,
                )
            )
        pop_rows = []
        pop_records = []
        for pop_direction, stack_name, stack in tuple(pop_specs or ()):
            pop_rows.append(
                (
                    self._cpp_environment_stack_key(pop_direction, stack_name),
                    stack,
                )
            )
            pop_records.append((str(pop_direction), str(stack_name), stack))
        record = {
            "sweep_direction": str(sweep_direction),
            "bond": int(bond),
            "environment_direction": str(environment_direction),
            "update_rows": tuple(update_rows),
            "pop_rows": tuple(pop_rows),
            "update_records": tuple(update_records),
            "pop_records": tuple(pop_records),
        }
        if not store:
            self.moving_profile_stats["cpp_bond_step_transaction_record_builds"] = int(
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_record_builds",
                    0,
                )
            ) + 1
            return record
        self._pending_cpp_bond_environment_step = record
        self._last_cpp_bond_environment_step = None
        self.moving_profile_stats["cpp_bond_step_transaction_prepared"] = int(
            self.moving_profile_stats.get(
                "cpp_bond_step_transaction_prepared",
                0,
            )
        ) + 1
        return True

    def consume_cpp_bond_environment_step(self, sweep_direction, bond):
        info = self._last_cpp_bond_environment_step
        if not info:
            return None
        if (
            str(info.get("sweep_direction")) != str(sweep_direction)
            or int(info.get("bond", -1)) != int(bond)
        ):
            return None
        self._last_cpp_bond_environment_step = None
        direction = str(info.get("environment_direction", ""))
        push_phase = "push_left" if direction == "left" else "push_right"
        for stack_name, stack, _direction in tuple(info.get("update_records", ())):
            self._record_environment_stack_update(
                push_phase,
                stack_name,
                len(stack),
            )
        for pop_direction, stack_name, stack in tuple(info.get("pop_records", ())):
            pop_phase = "pop_left" if str(pop_direction) == "left" else "pop_right"
            self._record_environment_stack_update(
                pop_phase,
                stack_name,
                len(stack),
            )
        self.moving_profile_stats["cpp_bond_step_transaction_commits"] = int(
            self.moving_profile_stats.get(
                "cpp_bond_step_transaction_commits",
                0,
            )
        ) + 1
        self.moving_profile_stats[
            "cpp_bond_step_transaction_commit_backend_actual"
        ] = "cpp_moving_environment"
        self._last_environment_update_backend = "cpp_bond_step_transaction"
        return dict(info)

    def clear_cpp_bond_environment_step(self):
        self._pending_cpp_bond_environment_step = None
        self._last_cpp_bond_environment_step = None

    def _run_cpp_owner_bond_step(
        self,
        *,
        prepare,
        prepare_payload,
        optimize,
        assign,
        invalidate,
        cache_guess,
        move_environment,
        fallback_environment,
        direct_family_payload_key=None,
        direct_family_builder_key=None,
        direct_family_plan_key=None,
        owner_local_optimize_key=None,
    ):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "run_owner_bond_step"):
            return None
        if not bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_owner_bond_step_runner",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        ):
            return None
        return owner.run_owner_bond_step(
            prepare,
            prepare_payload if prepare_payload is not None else None,
            optimize,
            assign,
            invalidate,
            cache_guess if cache_guess is not None else None,
            move_environment,
            fallback_environment if fallback_environment is not None else None,
            "" if direct_family_payload_key is None else str(direct_family_payload_key),
            "" if direct_family_builder_key is None else str(direct_family_builder_key),
            "" if direct_family_plan_key is None else str(direct_family_plan_key),
            "" if owner_local_optimize_key is None else str(owner_local_optimize_key),
        )

    def run_single_state_bond_step(
        self,
        *,
        sweep_direction,
        bond,
        prepare,
        optimize,
        assign,
        invalidate,
        cache_guess=None,
        prepare_payload=None,
        direct_family_payload_key=None,
        direct_family_builder_key=None,
        direct_family_plan_key=None,
        owner_local_optimize_key=None,
        move_environment,
        fallback_environment=None,
    ):
        start = time.perf_counter()
        stats = self.moving_profile_stats
        stats["owner_bond_step_calls"] = int(
            stats.get("owner_bond_step_calls", 0)
        ) + 1
        prepared = False
        moved = False
        try:
            cpp_result = self._run_cpp_owner_bond_step(
                prepare=prepare,
                prepare_payload=prepare_payload,
                optimize=optimize,
                assign=assign,
                invalidate=invalidate,
                cache_guess=cache_guess,
                move_environment=move_environment,
                fallback_environment=fallback_environment,
                direct_family_payload_key=direct_family_payload_key,
                direct_family_builder_key=direct_family_builder_key,
                direct_family_plan_key=direct_family_plan_key,
                owner_local_optimize_key=owner_local_optimize_key,
            )
            if cpp_result is not None:
                self._sync_cpp_moving_environment_stats()
                result = cpp_result["result"]
                prepared = bool(cpp_result.get("prepared", False))
                moved = bool(cpp_result.get("moved", False))
                if bool(cpp_result.get("payload_prepared", False)):
                    payload_elapsed = float(cpp_result.get("payload_seconds", 0.0))
                    stats["owner_bond_step_payload_prepares"] = int(
                        stats.get("owner_bond_step_payload_prepares", 0)
                    ) + 1
                    stats["owner_bond_step_payload_prepare_seconds"] = float(
                        stats.get("owner_bond_step_payload_prepare_seconds", 0.0)
                    ) + payload_elapsed
                    stats["owner_bond_step_payload_prepare_last_seconds"] = (
                        payload_elapsed
                    )
                    if bool(
                        cpp_result.get("direct_family_payload_prepared", False)
                    ):
                        stats[
                            "owner_direct_family_environment_prepared_payloads"
                        ] = int(
                            stats.get(
                                "owner_direct_family_environment_prepared_payloads",
                                0,
                            )
                        ) + 1
                if bool(cpp_result.get("fallback_used", False)):
                    stats["owner_bond_step_environment_fallbacks"] = int(
                        stats.get("owner_bond_step_environment_fallbacks", 0)
                    ) + 1
                if moved:
                    stats["owner_bond_step_environment_moves"] = int(
                        stats.get("owner_bond_step_environment_moves", 0)
                    ) + 1
                stats["owner_bond_step_accepts"] = int(
                    stats.get("owner_bond_step_accepts", 0)
                ) + 1
                stats["owner_bond_step_orchestrator_actual"] = (
                    "cpp_moving_environment"
                )
                backend = getattr(self, "_last_environment_update_backend", None)
                if backend is None:
                    backend = "cpp_owner_bond_step_runner"
                stats["owner_bond_step_backend_actual"] = backend
                stats["owner_bond_step_last_error"] = None
                return result
            prepared = bool(prepare())
            if prepare_payload is not None:
                payload_start = time.perf_counter()
                prepare_payload()
                payload_elapsed = time.perf_counter() - payload_start
                stats["owner_bond_step_payload_prepares"] = int(
                    stats.get("owner_bond_step_payload_prepares", 0)
                ) + 1
                stats["owner_bond_step_payload_prepare_seconds"] = float(
                    stats.get("owner_bond_step_payload_prepare_seconds", 0.0)
                ) + payload_elapsed
                stats["owner_bond_step_payload_prepare_last_seconds"] = (
                    payload_elapsed
                )
            result = optimize()
            assign(result)
            invalidate()
            if cache_guess is not None:
                cache_guess()
            moved = bool(move_environment())
            if not moved and fallback_environment is not None:
                fallback_environment()
                moved = True
                stats["owner_bond_step_environment_fallbacks"] = int(
                    stats.get("owner_bond_step_environment_fallbacks", 0)
                ) + 1
            if moved:
                stats["owner_bond_step_environment_moves"] = int(
                    stats.get("owner_bond_step_environment_moves", 0)
                ) + 1
            stats["owner_bond_step_accepts"] = int(
                stats.get("owner_bond_step_accepts", 0)
            ) + 1
            backend = getattr(self, "_last_environment_update_backend", None)
            if backend is None:
                backend = "cpp_moving_environment" if prepared else "python"
            stats["owner_bond_step_orchestrator_actual"] = "python"
            stats["owner_bond_step_backend_actual"] = backend
            stats["owner_bond_step_last_error"] = None
            return result
        except Exception as exc:
            self.clear_cpp_bond_environment_step()
            stats["owner_bond_step_failures"] = int(
                stats.get("owner_bond_step_failures", 0)
            ) + 1
            stats["owner_bond_step_last_error"] = str(exc)
            raise
        finally:
            elapsed = time.perf_counter() - start
            stats["owner_bond_step_seconds"] = float(
                stats.get("owner_bond_step_seconds", 0.0)
            ) + elapsed
            stats["owner_bond_step_last_seconds"] = elapsed

    def _run_cpp_owner_half_sweep(
        self,
        *,
        direction,
        bonds,
        make_step,
        make_update,
        after_step,
        step_direction,
    ):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "run_owner_half_sweep"):
            return None
        if not bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_owner_half_sweep_runner",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        ):
            return None
        if bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_owner_half_sweep_step_records",
                False,
            )
        ) and hasattr(owner, "install_owner_bond_step") and hasattr(
            owner,
            "run_owner_half_sweep_from_step_keys",
        ):
            step_keys = []
            install_start = time.perf_counter()
            for bond in tuple(int(bond) for bond in bonds):
                spec = dict(make_step(bond))
                step_key = (
                    f"owner-bond-step:{id(self)}:{direction}:"
                    f"{step_direction or direction}:{bond}"
                )
                owner.install_owner_bond_step(
                    step_key,
                    spec.get("prepare"),
                    spec.get("prepare_payload"),
                    spec.get("optimize"),
                    spec.get("assign"),
                    spec.get("invalidate"),
                    spec.get("cache_guess"),
                    spec.get("move_environment"),
                    spec.get("fallback_environment"),
                    "" if spec.get("direct_family_payload_key") is None else str(
                        spec.get("direct_family_payload_key")
                    ),
                    "" if spec.get("direct_family_builder_key") is None else str(
                        spec.get("direct_family_builder_key")
                    ),
                    "" if spec.get("direct_family_plan_key") is None else str(
                        spec.get("direct_family_plan_key")
                    ),
                    "" if spec.get("owner_local_optimize_key") is None else str(
                        spec.get("owner_local_optimize_key")
                    ),
                )
                step_keys.append((bond, step_key))
            stats = self.moving_profile_stats
            elapsed = time.perf_counter() - install_start
            stats["owner_half_sweep_step_record_installs"] = int(
                stats.get("owner_half_sweep_step_record_installs", 0)
            ) + len(step_keys)
            stats["owner_half_sweep_step_record_install_seconds"] = float(
                stats.get("owner_half_sweep_step_record_install_seconds", 0.0)
            ) + elapsed
            stats["owner_half_sweep_step_record_install_last_seconds"] = elapsed
            return owner.run_owner_half_sweep_from_step_keys(
                str(direction),
                tuple(step_keys),
                make_update if make_update is not None else None,
                after_step if after_step is not None else None,
                str(step_direction or direction),
            )
        return owner.run_owner_half_sweep(
            str(direction),
            tuple(int(bond) for bond in bonds),
            make_step,
            make_update if make_update is not None else None,
            after_step if after_step is not None else None,
            str(step_direction or direction),
        )

    def run_single_state_half_sweep(
        self,
        *,
        direction,
        bonds,
        make_step,
        make_update=None,
        after_step=None,
        step_direction=None,
    ):
        start = time.perf_counter()
        stats = self.moving_profile_stats
        stats["owner_half_sweep_calls"] = int(
            stats.get("owner_half_sweep_calls", 0)
        ) + 1
        stats["owner_half_sweep_last_direction"] = str(direction)
        updates = []
        last_bond = None
        last_result = None
        n_bonds = 0
        try:
            cpp_summary = self._run_cpp_owner_half_sweep(
                direction=direction,
                bonds=bonds,
                make_step=make_step,
                make_update=make_update,
                after_step=after_step,
                step_direction=step_direction,
            )
            if cpp_summary is not None:
                self._sync_cpp_moving_environment_stats()
                updates = list(cpp_summary.get("updates", ()))
                last_bond_value = cpp_summary.get("last_bond")
                last_bond = (
                    None if last_bond_value is None else int(last_bond_value)
                )
                last_result = cpp_summary.get("last_result")
                n_bonds = int(cpp_summary.get("bonds", 0))
                payload_prepares = int(cpp_summary.get("payload_prepares", 0))
                direct_payload_prepares = int(
                    cpp_summary.get("direct_family_payload_prepares", 0)
                )
                payload_seconds = float(cpp_summary.get("payload_seconds", 0.0))
                env_moves = int(cpp_summary.get("environment_moves", 0))
                env_fallbacks = int(cpp_summary.get("environment_fallbacks", 0))
                stats["owner_bond_step_calls"] = int(
                    stats.get("owner_bond_step_calls", 0)
                ) + n_bonds
                stats["owner_bond_step_accepts"] = int(
                    stats.get("owner_bond_step_accepts", 0)
                ) + n_bonds
                stats["owner_bond_step_payload_prepares"] = int(
                    stats.get("owner_bond_step_payload_prepares", 0)
                ) + payload_prepares
                stats["owner_bond_step_payload_prepare_seconds"] = float(
                    stats.get("owner_bond_step_payload_prepare_seconds", 0.0)
                ) + payload_seconds
                stats["owner_bond_step_payload_prepare_last_seconds"] = (
                    payload_seconds
                )
                if direct_payload_prepares:
                    stats[
                        "owner_direct_family_environment_prepared_payloads"
                    ] = int(
                        stats.get(
                            "owner_direct_family_environment_prepared_payloads",
                            0,
                        )
                    ) + direct_payload_prepares
                stats["owner_bond_step_environment_moves"] = int(
                    stats.get("owner_bond_step_environment_moves", 0)
                ) + env_moves
                stats["owner_bond_step_environment_fallbacks"] = int(
                    stats.get("owner_bond_step_environment_fallbacks", 0)
                ) + env_fallbacks
                stats["owner_bond_step_orchestrator_actual"] = (
                    "cpp_moving_environment"
                )
                backend = getattr(self, "_last_environment_update_backend", None)
                if backend is None:
                    backend = "cpp_owner_half_sweep_runner"
                stats["owner_bond_step_backend_actual"] = backend
                stats["owner_bond_step_last_error"] = None
                stats["owner_half_sweep_bonds"] = int(
                    stats.get("owner_half_sweep_bonds", 0)
                ) + n_bonds
                stats["owner_half_sweep_accepts"] = int(
                    stats.get("owner_half_sweep_accepts", 0)
                ) + 1
                stats["owner_half_sweep_backend_actual"] = str(
                    cpp_summary.get("backend", "cpp_owner_half_sweep_runner")
                )
                stats["owner_half_sweep_last_error"] = None
                return {
                    "updates": updates,
                    "last_bond": last_bond,
                    "last_result": last_result,
                    "seconds": float(cpp_summary.get("seconds", 0.0)),
                }
            for bond in tuple(int(bond) for bond in bonds):
                n_bonds += 1
                last_bond = bond
                local_start = time.perf_counter()
                spec = dict(make_step(bond))
                last_result = self.run_single_state_bond_step(
                    sweep_direction=str(step_direction or direction),
                    bond=bond,
                    **spec,
                )
                local_seconds = time.perf_counter() - local_start
                update = None
                if make_update is not None:
                    update = make_update(bond, last_result, local_seconds)
                    if update is not None:
                        updates.append(update)
                if after_step is not None:
                    after_step(bond, last_result, update)
            stats["owner_half_sweep_bonds"] = int(
                stats.get("owner_half_sweep_bonds", 0)
            ) + n_bonds
            stats["owner_half_sweep_accepts"] = int(
                stats.get("owner_half_sweep_accepts", 0)
            ) + 1
            stats["owner_half_sweep_backend_actual"] = (
                stats.get("owner_bond_step_backend_actual") or "python"
            )
            stats["owner_half_sweep_last_error"] = None
            return {
                "updates": updates,
                "last_bond": last_bond,
                "last_result": last_result,
                "seconds": time.perf_counter() - start,
            }
        except Exception as exc:
            stats["owner_half_sweep_failures"] = int(
                stats.get("owner_half_sweep_failures", 0)
            ) + 1
            stats["owner_half_sweep_last_error"] = str(exc)
            raise
        finally:
            elapsed = time.perf_counter() - start
            stats["owner_half_sweep_seconds"] = float(
                stats.get("owner_half_sweep_seconds", 0.0)
            ) + elapsed
            stats["owner_half_sweep_last_seconds"] = elapsed

    def sweep_environment_step(self, direction, update_specs, pop_specs):
        direction = str(direction)
        if direction not in {"left", "right"}:
            raise ValueError("sweep environment step direction must be left or right")
        owner = self._cpp_moving_environment
        enabled = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_sweep_environment_step",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        )
        if (
            not enabled
            or owner is None
            or not hasattr(owner, "sweep_environment_step")
        ):
            return None
        update_specs = tuple(update_specs or ())
        pop_specs = tuple(pop_specs or ())
        update_auto = getattr(owner, "sweep_environment_step_auto", None)
        if update_auto is not None:
            update_rows = []
            for stack_name, stack, W, A, B in update_specs:
                if not stack:
                    return None
                update_rows.append(
                    (
                        self._cpp_environment_stack_key(direction, stack_name),
                        W,
                        A,
                        B,
                        stack,
                    )
                )
        else:
            update_rows = []
            for stack_name, stack, W, A, B in update_specs:
                if not stack:
                    return None
                current = stack[-1]
                plan_key = self.compiled_backend._environment_plan_key(
                    direction,
                    W,
                    A,
                    current,
                    B,
                )
                owner_key, signature = self.compiled_backend._environment_plan_owner_key(
                    plan_key
                )
                update_rows.append(
                    (
                        self._cpp_environment_stack_key(direction, stack_name),
                        owner_key,
                        signature,
                        W,
                        A,
                        B,
                        stack,
                    )
                )
        pop_rows = tuple(
            (
                self._cpp_environment_stack_key(pop_direction, stack_name),
                stack,
            )
            for pop_direction, stack_name, stack in pop_specs
        )
        stats = self.moving_profile_stats
        start = time.perf_counter()
        try:
            if update_auto is not None:
                updates, pops, syncs, failures = update_auto(
                    direction,
                    AbelianEnvironmentTensorData,
                    tuple(update_rows),
                    pop_rows,
                )
                stats["cpp_sweep_environment_step_auto_calls"] = int(
                    stats.get("cpp_sweep_environment_step_auto_calls", 0)
                ) + 1
            else:
                updates, pops, syncs, failures = owner.sweep_environment_step(
                    direction,
                    AbelianEnvironmentTensorData,
                    tuple(update_rows),
                    pop_rows,
                )
        except Exception as exc:
            stats["cpp_sweep_environment_step_failures"] = int(
                stats.get("cpp_sweep_environment_step_failures", 0)
            ) + 1
            stats["cpp_sweep_environment_step_last_error"] = str(exc)
            raise
        elapsed = float(time.perf_counter() - start)
        stats["cpp_sweep_environment_step_calls"] = int(
            stats.get("cpp_sweep_environment_step_calls", 0)
        ) + 1
        stats["cpp_sweep_environment_step_updates"] = int(
            stats.get("cpp_sweep_environment_step_updates", 0)
        ) + int(updates)
        stats["cpp_sweep_environment_step_pops"] = int(
            stats.get("cpp_sweep_environment_step_pops", 0)
        ) + int(pops)
        stats["cpp_sweep_environment_step_syncs"] = int(
            stats.get("cpp_sweep_environment_step_syncs", 0)
        ) + int(syncs)
        stats["cpp_sweep_environment_step_seconds"] = float(
            stats.get("cpp_sweep_environment_step_seconds", 0.0)
        ) + elapsed
        stats["cpp_sweep_environment_step_last_seconds"] = elapsed
        stats["cpp_sweep_environment_step_failures"] = int(
            stats.get("cpp_sweep_environment_step_failures", 0)
        ) + int(failures)
        stats["cpp_sweep_environment_step_backend_actual"] = (
            "cpp_moving_environment"
        )
        self._last_environment_update_backend = "cpp_sweep_environment_step"
        phase = "update_left" if direction == "left" else "update_right"
        self._record_environment_update(phase, elapsed)
        push_phase = "push_left" if direction == "left" else "push_right"
        for stack_name, stack, _W, _A, _B in update_specs:
            self._record_environment_stack_update(
                push_phase,
                stack_name,
                len(stack),
            )
        for pop_direction, stack_name, stack in pop_specs:
            pop_phase = "pop_left" if str(pop_direction) == "left" else "pop_right"
            self._record_environment_stack_update(
                pop_phase,
                stack_name,
                len(stack),
            )
        self._sync_cpp_moving_environment_stats()
        return {
            "updates": int(updates),
            "pops": int(pops),
            "syncs": int(syncs),
            "failures": int(failures),
            "seconds": elapsed,
        }

    def invalidate_direct_family_caches(self):
        if self._direct_family_revision_ref is not None:
            self._direct_family_revision_ref[0] += 1
            self.direct_family_revision = int(self._direct_family_revision_ref[0])
            for cache in self._direct_family_cache_maps:
                try:
                    cache.clear()
                except AttributeError:
                    continue
            self.moving_profile_stats["direct_family_cache_maps_cleared"] = int(
                self.moving_profile_stats.get("direct_family_cache_maps_cleared", 0)
            ) + int(len(self._direct_family_cache_maps))
        else:
            self.direct_family_revision += 1
            invalidator = self._direct_family_cache_invalidator
            if invalidator is not None:
                invalidator()
        self.moving_profile_stats["direct_family_cache_invalidations"] = int(
            self.moving_profile_stats.get("direct_family_cache_invalidations", 0)
        ) + 1
        self.moving_profile_stats["direct_family_cache_revision"] = int(
            self.direct_family_revision
        )
        self.clear_owner_direct_family_environment_cache()

    def clear_owner_direct_family_environment_cache(self):
        cleared = int(len(self._owner_direct_family_environment_cache))
        prepared_cleared = int(len(self._owner_direct_family_prepared_payloads))
        if cleared:
            self._owner_direct_family_environment_cache.clear()
            self.moving_profile_stats[
                "owner_direct_family_environment_cache_clears"
            ] = int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_cache_clears",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "owner_direct_family_environment_cache_cleared_entries"
            ] = int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_cache_cleared_entries",
                    0,
                )
            ) + cleared
        if prepared_cleared:
            self._owner_direct_family_prepared_payloads.clear()
            self.moving_profile_stats[
                "owner_direct_family_environment_prepared_cache_clears"
            ] = int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_prepared_cache_clears",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "owner_direct_family_environment_prepared_cleared_entries"
            ] = int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_prepared_cleared_entries",
                    0,
                )
            ) + prepared_cleared
        owner = self._cpp_moving_environment
        if owner is not None and hasattr(owner, "clear_direct_family_payloads"):
            try:
                owner.clear_direct_family_payloads()
                if hasattr(owner, "clear_direct_family_payload_builders"):
                    owner.clear_direct_family_payload_builders()
                self._sync_cpp_moving_environment_stats()
            except Exception as exc:
                self.moving_profile_stats[
                    "cpp_moving_environment_direct_family_payload_last_error"
                ] = str(exc)
        self.moving_profile_stats[
            "owner_direct_family_environment_cache_size"
        ] = int(len(self._owner_direct_family_environment_cache))
        self.moving_profile_stats[
            "owner_direct_family_environment_prepared_cache_size"
        ] = int(len(self._owner_direct_family_prepared_payloads))

    @staticmethod
    def _owner_direct_family_prepared_key(bond, cache_key):
        if cache_key is None:
            return None
        return (int(bond), cache_key)

    def _owner_direct_family_cpp_key_bundle(self, bond, cache_key):
        if cache_key is None:
            return None
        owner = self._cpp_moving_environment
        if owner is not None and hasattr(owner, "direct_family_cpp_key_bundle"):
            try:
                bundle = owner.direct_family_cpp_key_bundle(int(bond), cache_key)
                self._sync_cpp_moving_environment_stats()
                self.moving_profile_stats[
                    "owner_direct_family_cpp_key_bundle_backend_actual"
                ] = "cpp_moving_environment"
                return bundle
            except Exception as exc:
                self.moving_profile_stats[
                    "cpp_moving_environment_direct_family_cpp_key_bundle_last_error"
                ] = str(exc)
        return {
            "payload_key": f"direct-family-payload:{int(bond)}:{cache_key!r}",
            "builder_key": f"direct-family-builder:{int(bond)}:{cache_key!r}",
            "plan_key": f"direct-family-plan:{int(bond)}:{cache_key!r}",
            "cache_key": cache_key,
        }

    def _owner_direct_family_cpp_payload_key(self, bond, cache_key):
        bundle = self._owner_direct_family_cpp_key_bundle(bond, cache_key)
        if bundle is None:
            return None
        return str(bundle.get("payload_key", "") or "")

    def _owner_direct_family_cpp_builder_key(self, bond, cache_key):
        bundle = self._owner_direct_family_cpp_key_bundle(bond, cache_key)
        if bundle is None:
            return None
        return str(bundle.get("builder_key", "") or "")

    def _owner_direct_family_cpp_plan_key(self, bond, cache_key):
        bundle = self._owner_direct_family_cpp_key_bundle(bond, cache_key)
        if bundle is None:
            return None
        return str(bundle.get("plan_key", "") or "")

    def _install_cpp_direct_family_payload(self, bond, cache_key, env):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "install_direct_family_payload"):
            return False
        payload_key = self._owner_direct_family_cpp_payload_key(bond, cache_key)
        if payload_key is None:
            return False
        try:
            owner.install_direct_family_payload(payload_key, env)
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment"
            return True
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_payload_last_error"
            ] = str(exc)
            return False

    def _cpp_direct_family_payload(self, bond, cache_key):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "direct_family_payload"):
            return None
        payload_key = self._owner_direct_family_cpp_payload_key(bond, cache_key)
        if payload_key is None:
            return None
        try:
            payload = owner.direct_family_payload(payload_key)
            self._sync_cpp_moving_environment_stats()
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_payload_last_error"
            ] = str(exc)
            return None
        if payload is None:
            return None
        self.moving_profile_stats[
            "owner_direct_family_environment_payload_owner"
        ] = "cpp_moving_environment"
        return payload

    def assemble_cpp_direct_family_payload(
        self,
        family_names,
        pieces,
        *,
        payload_key=None,
        install=False,
    ):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "assemble_direct_family_payload"):
            return None
        try:
            payload = owner.assemble_direct_family_payload(
                "" if payload_key is None else str(payload_key),
                tuple(family_names),
                tuple(pieces),
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
                bool(install),
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_assembler"
            return payload
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_payload_assembler_last_error"
            ] = str(exc)
            return None

    def build_cpp_direct_family_payload_from_piece_builders(
        self,
        initial_parts,
        family_names,
        builders,
        *,
        payload_key=None,
        install=False,
    ):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not hasattr(owner, "build_direct_family_payload_from_piece_builders")
        ):
            return None
        try:
            payload = owner.build_direct_family_payload_from_piece_builders(
                "" if payload_key is None else str(payload_key),
                tuple(name for name, _entries in initial_parts),
                tuple(entries for _name, entries in initial_parts),
                tuple(family_names),
                tuple(builders),
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
                bool(install),
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_piece_plan"
            return payload
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_piece_builder_plan_last_error"
            ] = str(exc)
            return None

    def build_cpp_direct_family_payload_from_phased_piece_builders(
        self,
        first_family_names,
        first_builders,
        second_builder_factory,
        *,
        payload_key=None,
        install=False,
    ):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not hasattr(owner, "build_direct_family_payload_from_phased_piece_builders")
        ):
            return None
        try:
            payload = owner.build_direct_family_payload_from_phased_piece_builders(
                "" if payload_key is None else str(payload_key),
                tuple(first_family_names),
                tuple(first_builders),
                second_builder_factory,
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
                bool(install),
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_phased_piece_plan"
            return payload
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_piece_builder_plan_last_error"
            ] = str(exc)
            return None

    def prepare_cpp_direct_family_payload_from_phased_piece_plan(
        self,
        plan_key,
        first_family_names,
        first_builders,
        second_builder_factory,
        *,
        payload_key=None,
        install=False,
    ):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not hasattr(owner, "install_direct_family_phased_piece_plan")
            or not hasattr(owner, "prepare_direct_family_payload_from_phased_piece_plan")
        ):
            return None
        try:
            owner.install_direct_family_phased_piece_plan(
                str(plan_key),
                tuple(first_family_names),
                tuple(first_builders),
                second_builder_factory,
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
            )
            payload = owner.prepare_direct_family_payload_from_phased_piece_plan(
                "" if payload_key is None else str(payload_key),
                str(plan_key),
                False,
                bool(install),
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_phased_piece_plan_handle"
            return payload
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_phased_piece_plan_last_error"
            ] = str(exc)
            return None

    def prepare_cpp_direct_family_payload_from_phased_family_plan(
        self,
        plan_key,
        first_family_names,
        first_builders,
        family_plan_factory,
        *,
        payload_key=None,
        install=False,
    ):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not hasattr(owner, "install_direct_family_phased_family_plan")
            or not hasattr(owner, "prepare_direct_family_payload_from_phased_family_plan")
        ):
            return None
        try:
            owner.install_direct_family_phased_family_plan(
                str(plan_key),
                tuple(first_family_names),
                tuple(first_builders),
                family_plan_factory,
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
            )
            payload = owner.prepare_direct_family_payload_from_phased_family_plan(
                "" if payload_key is None else str(payload_key),
                str(plan_key),
                False,
                bool(install),
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_phased_family_plan_handle"
            return payload
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_phased_family_plan_last_error"
            ] = str(exc)
            return None

    def prepare_cpp_direct_family_payload_from_two_phase_dispatch_plan(
        self,
        plan_key,
        first_plan_factory,
        second_plan_factory,
        *,
        payload_key=None,
        install=False,
    ):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not hasattr(owner, "install_direct_family_two_phase_dispatch_plan")
            or not hasattr(owner, "prepare_direct_family_payload_from_two_phase_dispatch_plan")
        ):
            return None
        try:
            owner.install_direct_family_two_phase_dispatch_plan(
                str(plan_key),
                first_plan_factory,
                second_plan_factory,
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
            )
            payload = owner.prepare_direct_family_payload_from_two_phase_dispatch_plan(
                "" if payload_key is None else str(payload_key),
                str(plan_key),
                False,
                bool(install),
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_two_phase_dispatch_plan_handle"
            return payload
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_two_phase_dispatch_plan_last_error"
            ] = str(exc)
            return None

    def _install_cpp_direct_family_two_phase_dispatch_static_plan(
        self,
        bond,
        cache_key,
        first_plan,
        second_plan,
    ):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not hasattr(owner, "install_direct_family_two_phase_dispatch_static_plan")
        ):
            return None
        bundle = self._owner_direct_family_cpp_key_bundle(bond, cache_key)
        if bundle is None:
            return None
        payload_key = str(bundle.get("payload_key", "") or "")
        plan_key = str(bundle.get("plan_key", "") or "")
        if not payload_key or not plan_key:
            return None

        try:
            owner.install_direct_family_two_phase_dispatch_static_plan(
                plan_key,
                first_plan,
                second_plan,
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_static_two_phase_dispatch_plan"
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_two_phase_dispatch_plan_last_error"
            ] = str(exc)
            return None
        return payload_key, plan_key

    def _install_cpp_direct_family_static_payload(
        self,
        bond,
        cache_key,
        family_names,
        pieces,
    ):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "assemble_direct_family_payload"):
            return None
        bundle = self._owner_direct_family_cpp_key_bundle(bond, cache_key)
        if bundle is None:
            return None
        payload_key = str(bundle.get("payload_key", "") or "")
        if not payload_key:
            return None
        try:
            payload = owner.assemble_direct_family_payload(
                payload_key,
                tuple(family_names),
                tuple(pieces),
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
                True,
            )
            self._sync_cpp_moving_environment_stats()
            if payload is None:
                return None
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_static_direct_family_payload"
            return payload_key
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_static_payload_last_error"
            ] = str(exc)
            return None

    def _install_cpp_direct_family_two_phase_dispatch_plan(
        self,
        bond,
        cache_key,
        first_plan_factory,
        second_plan_factory,
    ):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "install_direct_family_two_phase_dispatch_plan"):
            return None
        bundle = self._owner_direct_family_cpp_key_bundle(bond, cache_key)
        if bundle is None:
            return None
        payload_key = str(bundle.get("payload_key", "") or "")
        plan_key = str(bundle.get("plan_key", "") or "")
        if not payload_key or not plan_key:
            return None

        try:
            owner.install_direct_family_two_phase_dispatch_plan(
                plan_key,
                first_plan_factory,
                second_plan_factory,
                AbelianCompositePackedDirectFamilyEntries,
                AbelianPackedDirectFamilyEntries,
            )
            self._sync_cpp_moving_environment_stats()
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_two_phase_dispatch_plan_handle"
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_two_phase_dispatch_plan_last_error"
            ] = str(exc)
            return None
        return payload_key, plan_key

    def _install_cpp_direct_family_payload_builder(self, bond, cache_key, build):
        owner = self._cpp_moving_environment
        if owner is None or not hasattr(owner, "install_direct_family_payload_builder"):
            return None
        bundle = self._owner_direct_family_cpp_key_bundle(bond, cache_key)
        if bundle is None:
            return None
        payload_key = str(bundle.get("payload_key", "") or "")
        builder_key = str(bundle.get("builder_key", "") or "")
        if not payload_key or not builder_key:
            return None

        try:
            owner.install_direct_family_payload_builder(
                builder_key,
                build,
            )
            self._sync_cpp_moving_environment_stats()
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_payload_builder_last_error"
            ] = str(exc)
            return None
        return payload_key, builder_key

    def _prepare_cpp_direct_family_payload_from_builder(
        self,
        bond,
        cache_key,
        build,
    ):
        owner = self._cpp_moving_environment
        if (
            owner is None
            or not hasattr(owner, "install_direct_family_payload_builder")
            or not hasattr(owner, "prepare_direct_family_payload_from_builder")
        ):
            return None
        key_pair = self._install_cpp_direct_family_payload_builder(
            bond,
            cache_key,
            build,
        )
        if key_pair is None:
            return None
        payload_key, builder_key = key_pair

        try:
            payload = owner.prepare_direct_family_payload_from_builder(
                payload_key,
                builder_key,
            )
            self._sync_cpp_moving_environment_stats()
            if payload is None:
                return None
            self.moving_profile_stats[
                "owner_direct_family_environment_payload_owner"
            ] = "cpp_moving_environment_builder"
            return payload
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_payload_builder_last_error"
            ] = str(exc)
            return None

    def direct_family_environment_for_bond(self, bond, build, *, cache_key=None):
        stats = self.moving_profile_stats
        stats["owner_direct_family_environment_calls"] = int(
            stats.get("owner_direct_family_environment_calls", 0)
        ) + 1
        stats["owner_direct_family_environment_last_bond"] = int(bond)
        if cache_key is not None:
            cached = self._owner_direct_family_environment_cache.get(cache_key)
            if cached is not None:
                stats["owner_direct_family_environment_cache_hits"] = int(
                    stats.get("owner_direct_family_environment_cache_hits", 0)
                ) + 1
                stats["owner_direct_family_environment_cache_size"] = int(
                    len(self._owner_direct_family_environment_cache)
                )
                stats["owner_direct_family_environment_last_error"] = None
                return cached
            stats["owner_direct_family_environment_cache_misses"] = int(
                stats.get("owner_direct_family_environment_cache_misses", 0)
            ) + 1
        start = time.perf_counter()
        try:
            env = build()
        except Exception as exc:
            stats["owner_direct_family_environment_last_error"] = str(exc)
            raise
        elapsed = time.perf_counter() - start
        stats["owner_direct_family_environment_builds"] = int(
            stats.get("owner_direct_family_environment_builds", 0)
        ) + 1
        stats["owner_direct_family_environment_seconds"] = float(
            stats.get("owner_direct_family_environment_seconds", 0.0)
        ) + elapsed
        stats["owner_direct_family_environment_last_seconds"] = elapsed
        try:
            entry_count = sum(int(len(entries)) for entries in (env or {}).values())
        except Exception:
            entry_count = 0
        stats["owner_direct_family_environment_entries"] = int(
            stats.get("owner_direct_family_environment_entries", 0)
        ) + int(entry_count)
        stats["owner_direct_family_environment_last_entries"] = int(entry_count)
        stats["owner_direct_family_environment_last_error"] = None
        if cache_key is not None:
            self._owner_direct_family_environment_cache[cache_key] = env
        stats["owner_direct_family_environment_cache_size"] = int(
            len(self._owner_direct_family_environment_cache)
        )
        return env

    def prepare_direct_family_environment_for_bond(
        self,
        bond,
        build,
        *,
        cache_key=None,
    ):
        env = self._prepare_cpp_direct_family_payload_from_builder(
            bond,
            cache_key,
            build,
        )
        used_cpp_builder = env is not None
        if env is None:
            env = self.direct_family_environment_for_bond(
                bond,
                build,
                cache_key=cache_key,
            )
        prepared_key = self._owner_direct_family_prepared_key(bond, cache_key)
        if prepared_key is not None:
            self._owner_direct_family_prepared_payloads[prepared_key] = env
            if not used_cpp_builder:
                self._install_cpp_direct_family_payload(bond, cache_key, env)
            stats = self.moving_profile_stats
            stats["owner_direct_family_environment_prepared_payloads"] = int(
                stats.get("owner_direct_family_environment_prepared_payloads", 0)
            ) + 1
            stats["owner_direct_family_environment_prepared_cache_size"] = int(
                len(self._owner_direct_family_prepared_payloads)
            )
        return env

    def direct_family_prepared_environment_for_bond(
        self,
        bond,
        build,
        *,
        cache_key=None,
    ):
        prepared_key = self._owner_direct_family_prepared_key(bond, cache_key)
        stats = self.moving_profile_stats
        if prepared_key is not None:
            cpp_payload = self._cpp_direct_family_payload(bond, cache_key)
            if cpp_payload is not None:
                stats["owner_direct_family_environment_prepared_hits"] = int(
                    stats.get(
                        "owner_direct_family_environment_prepared_hits",
                        0,
                    )
                ) + 1
                stats["owner_direct_family_environment_prepared_cache_size"] = (
                    int(len(self._owner_direct_family_prepared_payloads))
                )
                stats["owner_direct_family_environment_last_error"] = None
                return cpp_payload
            if prepared_key in self._owner_direct_family_prepared_payloads:
                stats["owner_direct_family_environment_prepared_hits"] = int(
                    stats.get(
                        "owner_direct_family_environment_prepared_hits",
                        0,
                    )
                ) + 1
                stats["owner_direct_family_environment_prepared_cache_size"] = (
                    int(len(self._owner_direct_family_prepared_payloads))
                )
                stats["owner_direct_family_environment_last_error"] = None
                return self._owner_direct_family_prepared_payloads[prepared_key]
            stats["owner_direct_family_environment_prepared_misses"] = int(
                stats.get(
                    "owner_direct_family_environment_prepared_misses",
                    0,
                )
            ) + 1
        return self.direct_family_environment_for_bond(
            bond,
            build,
            cache_key=cache_key,
        )

    def _record_environment_stack_update(self, phase, stack_name, depth):
        updates = self.moving_profile_stats.setdefault(
            "environment_stack_updates",
            {},
        )
        key = str(phase)
        entry = updates.setdefault(
            key,
            {"calls": 0, "last_stack": None, "last_depth": 0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["last_stack"] = str(stack_name)
        entry["last_depth"] = int(depth)

    @staticmethod
    def _layout(A):
        return HamiltonianMultiplyU1._layout(A)

    @staticmethod
    def _qns_from_layout_with_proto(layout, proto):
        return HamiltonianMultiplyU1._qns_from_layout_with_proto(layout, proto)

    @staticmethod
    def _size(layout):
        return HamiltonianMultiplyU1._size(layout)

    @staticmethod
    def _flatten(A, layout):
        return HamiltonianMultiplyU1._flatten(A, layout)

    @staticmethod
    def _unflatten(vec, proto, layout, *, drop_zero_blocks=False, zero_tol=0.0):
        return HamiltonianMultiplyU1._unflatten(
            vec,
            proto,
            layout,
            drop_zero_blocks=drop_zero_blocks,
            zero_tol=zero_tol,
        )

    @staticmethod
    def _layout_from_map(layout_map):
        return HamiltonianMultiplyU1._layout_from_map(layout_map)

    @staticmethod
    def _remap_flat(vec, old_layout, new_layout):
        return HamiltonianMultiplyU1._remap_flat(vec, old_layout, new_layout)

    @staticmethod
    def _block_data_dtype(*objects):
        return HamiltonianMultiplyU1._block_data_dtype(*objects)

    @staticmethod
    def _zero_proto_from_layout(proto, layout, dtype):
        return HamiltonianMultiplyU1._zero_proto_from_layout(proto, layout, dtype)

    def _safe_two_site_layout_map(self, proto):
        return abelian_safe_two_site_layout_map(proto, self.W)

    def _boundary_family_tables(self):
        return ()

    def _local_action_dtype(self, *objects):
        return self._block_data_dtype(
            self.E,
            self.W,
            self.F,
            self.complementary_family_environments,
            self.complementary_direct_family_environments,
            *objects,
        )

    def _action_token(self):
        return (
            HamiltonianMultiplyU1._tensor_token(self.E),
            HamiltonianMultiplyU1._tensor_token(self.W[0]),
            HamiltonianMultiplyU1._tensor_token(self.W[1]),
            HamiltonianMultiplyU1._tensor_token(self.F),
            None if self.bond is None else int(self.bond),
            id(self.complementary_operator_families),
            id(self.complementary_family_environments),
            id(self.complementary_direct_family_environments),
        )

    def jacobi_preconditioner(self, proto, *, floor=1.0e-8):
        del proto, floor
        return None

    def _operatorless_local_problem_enabled(self, options, families):
        if not bool(
            self._option_value(
                options,
                "moving_environment_operatorless_local_problem",
                False,
            )
        ):
            return False
        if self._cpp_moving_environment is None or families is None:
            return False
        return bool(self.uses_cpp_family_mpo_descriptor())

    def _bind_operatorless_local_options(self, options, families):
        def _option(name, default):
            if options is not None:
                if isinstance(options, dict) and name in options:
                    return options[name]
                if not isinstance(options, dict) and hasattr(options, name):
                    return getattr(options, name)
            return getattr(families, name, default)

        self._packed_local_davidson = bool(_option("packed_local_davidson", False))
        self._packed_local_davidson_min_dim = max(
            0,
            int(_option("packed_local_davidson_min_dim", 0)),
        )
        self._packed_local_davidson_max_dim = max(
            0,
            int(_option("packed_local_davidson_max_dim", 0)),
        )
        self._packed_local_davidson_max_iter = max(
            0,
            int(_option("packed_local_davidson_max_iter", 0)),
        )
        self._packed_local_davidson_restart_dim = max(
            0,
            int(_option("packed_local_davidson_restart_dim", 0)),
        )
        self._packed_local_flat_preconditioner = bool(
            _option("packed_local_flat_preconditioner", False)
        ) or self._use_table_flat_preconditioner(options)
        self._packed_local_safe_layout_expansion = bool(
            _option("packed_local_safe_layout_expansion", True)
        )
        self._packed_local_use_safe_closure = bool(
            _option("packed_local_use_safe_closure", True)
        )
        self._packed_local_project_current_support = bool(
            _option("packed_local_project_current_support", False)
        )
        self._packed_local_disable_generic_fallback = bool(
            _option("packed_local_disable_generic_fallback", False)
        )
        self._moving_environment_cpp_davidson = bool(
            _option("moving_environment_cpp_davidson", False)
        )
        self._packed_local_family_flat_matvec = bool(
            _option("packed_local_family_flat_matvec", False)
        )
        self._packed_local_family_flat_direct_matvec = bool(
            _option("packed_local_family_flat_direct_matvec", False)
        )
        backend = str(
            _option(
                "packed_local_family_flat_direct_matvec_backend",
                "compiled",
            )
        ).strip().lower()
        if backend in {
            "block2",
            "block2_like",
            "block2-like",
            "block2_table",
            "renormalized",
            "renormalized_operator_table",
            "renormalized-operator-table",
        }:
            backend = "renormalized_table"
        self._packed_local_family_flat_direct_matvec_backend = backend
        self._packed_local_family_flat_direct_matvec_min_dim = max(
            0,
            int(_option("packed_local_family_flat_direct_matvec_min_dim", 0)),
        )
        self._renormalized_operator_table_dense_block_max_elements = max(
            0,
            int(
                _option(
                    "renormalized_operator_table_dense_block_max_elements",
                    20_000_000,
                )
            ),
        )
        self._renormalized_operator_table_sparse_density_threshold = max(
            0.0,
            float(_option("renormalized_operator_table_sparse_density_threshold", 0.0)),
        )

    def _bind_operatorless_local_problem(
        self,
        E,
        W,
        F,
        *,
        bond=None,
        complementary_operator_families=None,
        complementary_boundary_payloads=None,
        complementary_split_stats=None,
        complementary_family_environments=None,
        complementary_direct_family_environments=None,
        matvec_options=None,
    ):
        families = (
            self.complementary_operator_families
            if complementary_operator_families is None
            else complementary_operator_families
        )
        options = self.matvec_options if matvec_options is None else matvec_options
        if not self._operatorless_local_problem_enabled(options, families):
            return False
        self.E = E
        self.W = W
        self.F = F
        self.complementary_operator_families = families
        self.complementary_boundary_payloads = complementary_boundary_payloads or {}
        self.complementary_split_stats = complementary_split_stats
        self.complementary_family_environments = (
            complementary_family_environments or {}
        )
        self.complementary_direct_family_environments = (
            complementary_direct_family_environments or {}
        )
        self._bind_operatorless_local_options(options, families)
        self._local_profile_stats = {
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "paths": {},
            "plan_builds": {},
        }
        self.last_packed_davidson_candidate = None
        self.last_packed_davidson_candidate_flat = None
        self.last_packed_davidson_candidate_layout = None
        self.last_packed_davidson_candidate_energy = None
        self.last_packed_davidson_candidate_residual = None
        self.last_packed_davidson_solution_flat = None
        self.last_packed_davidson_solution_layout = None
        self.last_packed_davidson_solution_energy = None
        self.last_packed_davidson_solution_residual = None
        self.last_packed_davidson_solution_converged = None
        self.last_packed_davidson_solution_update = None
        self._operatorless_local_problem_active = True
        self._moving_environment = self
        self.moving_profile_stats["operatorless_local_problem_binds"] = int(
            self.moving_profile_stats.get("operatorless_local_problem_binds", 0)
        ) + 1
        return True

    def set_bond(
        self,
        E,
        W,
        F,
        *,
        bond=None,
        complementary_operator_families=None,
        complementary_boundary_payloads=None,
        complementary_split_stats=None,
        complementary_family_environments=None,
        complementary_direct_family_environments=None,
        matvec_options=None,
    ):
        self.bond = None if bond is None else int(bond)
        self._dense_operatorless_local_problem_active = False
        self._dense_operatorless_key = None
        families = (
            self.complementary_operator_families
            if complementary_operator_families is None
            else complementary_operator_families
        )
        options = self.matvec_options if matvec_options is None else matvec_options
        if self._bind_operatorless_local_problem(
            E,
            W,
            F,
            bond=bond,
            complementary_operator_families=families,
            complementary_boundary_payloads=complementary_boundary_payloads,
            complementary_split_stats=complementary_split_stats,
            complementary_family_environments=complementary_family_environments,
            complementary_direct_family_environments=(
                complementary_direct_family_environments
            ),
            matvec_options=options,
        ):
            self.operator = None
            return self
        self._operatorless_local_problem_active = False
        reuse_enabled = bool(
            self._option_value(
                options,
                "moving_environment_reuse_local_operator",
                True,
            )
        )
        reused = False
        if reuse_enabled and self.operator is not None:
            resetter = getattr(self.operator, "reset_local_problem", None)
            if resetter is not None:
                reused = bool(
                    resetter(
                        E,
                        W,
                        F,
                        complementary_operator_families=families,
                        bond=bond,
                        complementary_boundary_payloads=complementary_boundary_payloads,
                        complementary_split_stats=complementary_split_stats,
                        complementary_family_environments=complementary_family_environments,
                        complementary_direct_family_environments=(
                            complementary_direct_family_environments
                        ),
                        matvec_options=options,
                    )
                )
        if reused:
            self.moving_profile_stats["local_operator_reuses"] = int(
                self.moving_profile_stats.get("local_operator_reuses", 0)
            ) + 1
        else:
            self.operator = HamiltonianMultiplyU1(
                E,
                W,
                F,
                complementary_operator_families=families,
                bond=bond,
                complementary_boundary_payloads=complementary_boundary_payloads,
                complementary_split_stats=complementary_split_stats,
                complementary_family_environments=complementary_family_environments,
                complementary_direct_family_environments=complementary_direct_family_environments,
                matvec_options=options,
            )
            self.moving_profile_stats["local_operator_builds"] = int(
                self.moving_profile_stats.get("local_operator_builds", 0)
            ) + 1
        if self._use_table_flat_preconditioner(options):
            self.operator._packed_local_flat_preconditioner = True
        self.operator._moving_environment = self
        return self

    def _dense_cpp_davidson_enabled(self, options):
        if not bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_davidson",
                False,
            )
        ):
            return False
        return (
            _cpp_davidson is not None
            and bool(getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False))
            and getattr(_cpp_davidson, "DenseSweepWorkspace", None) is not None
        )

    def _dense_cpp_tensor_primitives_enabled(self, options):
        if (
            _cpp_davidson is None
            or not bool(getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False))
        ):
            return False
        if (
            getattr(_cpp_davidson, "dense_coarse_grain_mpo", None) is None
            or getattr(_cpp_davidson, "dense_coarse_grain_mps", None) is None
        ):
            return False
        default = bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_davidson",
                False,
            )
        )
        return bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_tensor_primitives",
                default,
            )
        )

    def dense_coarse_grain_mpo(self, W1, W2, *, bond=None, matvec_options=None):
        options = self.matvec_options if matvec_options is None else matvec_options
        if (
            not self._dense_cpp_tensor_primitives_enabled(options)
            or not isinstance(W1, np.ndarray)
            or not isinstance(W2, np.ndarray)
        ):
            return coarse_grain_MPO(W1, W2)
        cache_enabled = bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_cache_coarse_grained_w",
                bool(
                    self._option_value(
                        options,
                        "moving_environment_dense_cpp_davidson",
                        False,
                    )
                ),
            )
        )
        cache_key = None if bond is None else self._dense_cpp_workspace_key(bond)
        signature = self._dense_cpp_w_pair_signature(
            W1,
            W2,
            bond=bond,
            options=options,
        )
        if (
            cache_enabled
            and cache_key is not None
            and self._dense_cpp_coarse_grained_w_signatures.get(cache_key)
            == signature
            and cache_key in self._dense_cpp_coarse_grained_w_cache
        ):
            self.moving_profile_stats["dense_cpp_coarse_grain_mpo_cache_hits"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_coarse_grain_mpo_cache_hits",
                    0,
                )
            ) + 1
            return self._dense_cpp_coarse_grained_w_cache[cache_key]
        start = time.perf_counter()
        try:
            value = _cpp_davidson.dense_coarse_grain_mpo(
                np.asarray(W1, dtype=np.complex128),
                np.asarray(W2, dtype=np.complex128),
            )
        except Exception as exc:
            self.moving_profile_stats["dense_cpp_tensor_primitive_failures"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_tensor_primitive_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["dense_cpp_tensor_primitive_last_error"] = str(exc)
            return coarse_grain_MPO(W1, W2)
        elapsed = float(time.perf_counter() - start)
        self.moving_profile_stats["dense_cpp_tensor_primitive_calls"] = int(
            self.moving_profile_stats.get("dense_cpp_tensor_primitive_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_cpp_coarse_grain_mpo_calls"] = int(
            self.moving_profile_stats.get("dense_cpp_coarse_grain_mpo_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_cpp_tensor_primitive_seconds"] = float(
            self.moving_profile_stats.get("dense_cpp_tensor_primitive_seconds", 0.0)
        ) + elapsed
        value = np.asarray(value)
        if cache_enabled and cache_key is not None:
            self._dense_cpp_coarse_grained_w_cache[cache_key] = value
            self._dense_cpp_coarse_grained_w_signatures[cache_key] = signature
        return value

    def dense_coarse_grain_mps(self, A, B, *, matvec_options=None):
        options = self.matvec_options if matvec_options is None else matvec_options
        if (
            not self._dense_cpp_tensor_primitives_enabled(options)
            or not isinstance(A, np.ndarray)
            or not isinstance(B, np.ndarray)
        ):
            return coarse_grain_MPS(A, B)
        start = time.perf_counter()
        try:
            value = _cpp_davidson.dense_coarse_grain_mps(
                np.asarray(A, dtype=np.complex128),
                np.asarray(B, dtype=np.complex128),
            )
        except Exception as exc:
            self.moving_profile_stats["dense_cpp_tensor_primitive_failures"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_tensor_primitive_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["dense_cpp_tensor_primitive_last_error"] = str(exc)
            return coarse_grain_MPS(A, B)
        elapsed = float(time.perf_counter() - start)
        self.moving_profile_stats["dense_cpp_tensor_primitive_calls"] = int(
            self.moving_profile_stats.get("dense_cpp_tensor_primitive_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_cpp_coarse_grain_mps_calls"] = int(
            self.moving_profile_stats.get("dense_cpp_coarse_grain_mps_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_cpp_tensor_primitive_seconds"] = float(
            self.moving_profile_stats.get("dense_cpp_tensor_primitive_seconds", 0.0)
        ) + elapsed
        return np.asarray(value)

    def _dense_cpp_workspace_key(self, bond):
        return "dense-bond:{}".format("none" if bond is None else int(bond))

    @staticmethod
    def _dense_cpp_array_signature(arr):
        arr_obj = np.asarray(arr)
        return (
            id(arr),
            tuple(int(x) for x in arr_obj.shape),
            tuple(int(x) for x in arr_obj.strides),
            str(arr_obj.dtype),
        )

    @staticmethod
    def _dense_cpp_payload_signature(E, W, F):
        return tuple(MovingEnvironment._dense_cpp_array_signature(arr) for arr in (E, W, F))

    def _dense_cpp_w_signature(self, W, *, bond=None, options=None):
        arr_obj = np.asarray(W)
        if (
            bond is not None
            and bool(
                self._option_value(
                    options,
                    "moving_environment_dense_cpp_static_w_by_bond",
                    True,
                )
            )
        ):
            return (
                "static-bond",
                int(bond),
                tuple(int(x) for x in arr_obj.shape),
                str(arr_obj.dtype),
            )
        return self._dense_cpp_array_signature(W)

    def _dense_cpp_w_pair_signature(self, W1, W2, *, bond=None, options=None):
        if (
            bond is not None
            and bool(
                self._option_value(
                    options,
                    "moving_environment_dense_cpp_static_w_by_bond",
                    True,
                )
            )
        ):
            arr1 = np.asarray(W1)
            arr2 = np.asarray(W2)
            return (
                "static-two-site-bond",
                int(bond),
                tuple(int(x) for x in arr1.shape),
                str(arr1.dtype),
                tuple(int(x) for x in arr2.shape),
                str(arr2.dtype),
            )
        return (
            self._dense_cpp_array_signature(W1),
            self._dense_cpp_array_signature(W2),
        )

    def _dense_cpp_sweep_owner(self, options):
        if not self._dense_cpp_davidson_enabled(options):
            return None
        if self._dense_cpp_sweep_workspace is None:
            owner_cls = getattr(_cpp_davidson, "DenseSweepWorkspace", None)
            if owner_cls is None:
                return None
            try:
                self._dense_cpp_sweep_workspace = owner_cls()
            except Exception as exc:
                self.moving_profile_stats["dense_cpp_sweep_workspace_failures"] = int(
                    self.moving_profile_stats.get(
                        "dense_cpp_sweep_workspace_failures",
                        0,
                    )
                ) + 1
                self.moving_profile_stats["dense_cpp_sweep_workspace_last_error"] = str(exc)
                return None
            self.moving_profile_stats["dense_cpp_sweep_workspace_creates"] = int(
                self.moving_profile_stats.get("dense_cpp_sweep_workspace_creates", 0)
            ) + 1
        self.moving_profile_stats["dense_cpp_sweep_workspace_enabled"] = True
        return self._dense_cpp_sweep_workspace

    def bind_dense_cpp_workspace(self, E, W, F, *, bond=None, matvec_options=None):
        options = self.matvec_options if matvec_options is None else matvec_options
        owner = self._dense_cpp_sweep_owner(options)
        if owner is None:
            return None
        key = self._dense_cpp_workspace_key(bond)
        signature = self._dense_cpp_payload_signature(E, W, F)
        w_signature = self._dense_cpp_w_signature(W, bond=bond, options=options)
        if self._dense_cpp_sweep_bind_signatures.get(key) == signature:
            self.moving_profile_stats["dense_cpp_sweep_workspace_bind_cache_hits"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_bind_cache_hits",
                    0,
                )
            ) + 1
            return key
        reuse_static_w = bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_reuse_static_w",
                True,
            )
        )
        if reuse_static_w and self._dense_cpp_sweep_w_signatures.get(key) == w_signature:
            start = time.perf_counter()
            try:
                owner.bind_boundaries(
                    key,
                    np.asarray(E, dtype=np.complex128),
                    np.asarray(F, dtype=np.complex128),
                )
            except Exception as exc:
                self.moving_profile_stats["dense_cpp_sweep_workspace_failures"] = int(
                    self.moving_profile_stats.get(
                        "dense_cpp_sweep_workspace_failures",
                        0,
                    )
                ) + 1
                self.moving_profile_stats["dense_cpp_sweep_workspace_last_error"] = str(exc)
            else:
                elapsed = float(time.perf_counter() - start)
                self._dense_cpp_sweep_bind_signatures[key] = signature
                self.moving_profile_stats[
                    "dense_cpp_sweep_workspace_boundary_binds"
                ] = int(
                    self.moving_profile_stats.get(
                        "dense_cpp_sweep_workspace_boundary_binds",
                        0,
                    )
                ) + 1
                self.moving_profile_stats[
                    "dense_cpp_sweep_workspace_boundary_bind_seconds"
                ] = float(
                    self.moving_profile_stats.get(
                        "dense_cpp_sweep_workspace_boundary_bind_seconds",
                        0.0,
                    )
                ) + elapsed
                self.moving_profile_stats["dense_cpp_sweep_workspace_static_w_hits"] = int(
                    self.moving_profile_stats.get(
                        "dense_cpp_sweep_workspace_static_w_hits",
                        0,
                    )
                ) + 1
                try:
                    stats = dict(owner.stats())
                    self.moving_profile_stats["dense_cpp_sweep_workspace_records"] = int(
                        stats.get("records", 0)
                    )
                except Exception:
                    pass
                return key
        start = time.perf_counter()
        try:
            owner.bind(
                key,
                np.asarray(E, dtype=np.complex128),
                np.asarray(W, dtype=np.complex128),
                np.asarray(F, dtype=np.complex128),
            )
        except Exception as exc:
            self.moving_profile_stats["dense_cpp_sweep_workspace_failures"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["dense_cpp_sweep_workspace_last_error"] = str(exc)
            self._dense_cpp_sweep_bind_signatures.pop(key, None)
            return None
        elapsed = float(time.perf_counter() - start)
        self._dense_cpp_sweep_bind_signatures[key] = signature
        self._dense_cpp_sweep_w_signatures[key] = w_signature
        self.moving_profile_stats["dense_cpp_sweep_workspace_binds"] = int(
            self.moving_profile_stats.get("dense_cpp_sweep_workspace_binds", 0)
        ) + 1
        self.moving_profile_stats["dense_cpp_sweep_workspace_bind_seconds"] = float(
            self.moving_profile_stats.get("dense_cpp_sweep_workspace_bind_seconds", 0.0)
        ) + elapsed
        try:
            stats = dict(owner.stats())
            self.moving_profile_stats["dense_cpp_sweep_workspace_records"] = int(
                stats.get("records", 0)
            )
        except Exception:
            pass
        return key

    def solve_dense_cpp_workspace(
        self,
        key,
        AA,
        *,
        tol,
        max_iter,
        restart_dim,
        accept_unconverged,
        backend,
        block_davidson=False,
        block_size=1,
    ):
        owner = self._dense_cpp_sweep_workspace
        if owner is None or key is None:
            return None
        start = time.perf_counter()
        try:
            if bool(block_davidson) and hasattr(owner, "solve_bound_block"):
                result = owner.solve_bound_block(
                    str(key),
                    np.asarray(AA, dtype=np.complex128).reshape(-1),
                    float(tol),
                    int(max_iter),
                    int(restart_dim),
                    bool(accept_unconverged),
                    str(backend),
                    max(1, int(block_size)),
                )
            else:
                result = owner.solve_bound(
                    str(key),
                    np.asarray(AA, dtype=np.complex128).reshape(-1),
                    float(tol),
                    int(max_iter),
                    int(restart_dim),
                    bool(accept_unconverged),
                    str(backend),
                )
        except Exception as exc:
            self.moving_profile_stats["dense_cpp_sweep_workspace_failures"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["dense_cpp_sweep_workspace_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - start)
        self.moving_profile_stats["dense_cpp_sweep_workspace_solve_calls"] = int(
            self.moving_profile_stats.get("dense_cpp_sweep_workspace_solve_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_cpp_sweep_workspace_solve_seconds"] = float(
            self.moving_profile_stats.get("dense_cpp_sweep_workspace_solve_seconds", 0.0)
        ) + elapsed
        if bool(result.get("block_davidson", False)):
            self.moving_profile_stats["dense_cpp_sweep_workspace_block_davidson_calls"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_block_davidson_calls",
                    0,
                )
            ) + 1
        try:
            stats = dict(owner.stats())
            self.moving_profile_stats["dense_cpp_sweep_workspace_records"] = int(
                stats.get("records", 0)
            )
        except Exception:
            pass
        return result

    def solve_dense_cpp_two_site_workspace(
        self,
        E,
        W1,
        W2,
        F,
        A,
        B,
        *,
        bond=None,
        nstates=1,
        tol=1.0e-9,
        max_iter=5000,
        matvec_options=None,
    ):
        options = self.matvec_options if matvec_options is None else matvec_options
        if int(nstates) != 1:
            return None
        if not bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_two_site_solve",
                bool(
                    self._option_value(
                        options,
                        "moving_environment_dense_cpp_davidson",
                        False,
                    )
                ),
            )
        ):
            return None
        owner = self._dense_cpp_sweep_owner(options)
        if owner is None or not hasattr(owner, "solve_two_site"):
            return None
        if not all(isinstance(x, np.ndarray) for x in (E, W1, W2, F, A, B)):
            return None
        key = self._dense_cpp_workspace_key(bond)
        restart_dim = int(
            self._option_value(
                options,
                "moving_environment_dense_cpp_davidson_restart_dim",
                min(max(8, int(max_iter)), 64),
            )
        )
        backend = str(
            self._option_value(
                options,
                "moving_environment_dense_cpp_davidson_backend",
                "blas",
            )
        )
        accept_unconverged = bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_davidson_accept_unconverged",
                False,
            )
        )
        reuse_static_w = bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_reuse_static_w",
                True,
            )
        )
        block_davidson = bool(
            self._option_value(
                options,
                "moving_environment_dense_cpp_block_davidson",
                False,
            )
        )
        block_size = max(
            1,
            int(
                self._option_value(
                    options,
                    "moving_environment_dense_cpp_block_davidson_size",
                    2,
                )
            ),
        )
        self.bond = None if bond is None else int(bond)
        self.operator = None
        self._operatorless_local_problem_active = False
        self._dense_operatorless_local_problem_active = True
        self._dense_operatorless_key = key
        self.moving_profile_stats["solve_local_calls"] = int(
            self.moving_profile_stats.get("solve_local_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_solve_local_calls"] = int(
            self.moving_profile_stats.get("dense_solve_local_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_operatorless_local_problem_solve_calls"] = int(
            self.moving_profile_stats.get(
                "dense_operatorless_local_problem_solve_calls",
                0,
            )
        ) + 1
        start = time.perf_counter()
        try:
            if block_davidson and hasattr(owner, "solve_two_site_block"):
                result = owner.solve_two_site_block(
                    str(key),
                    np.asarray(E, dtype=np.complex128),
                    np.asarray(W1, dtype=np.complex128),
                    np.asarray(W2, dtype=np.complex128),
                    np.asarray(F, dtype=np.complex128),
                    np.asarray(A, dtype=np.complex128),
                    np.asarray(B, dtype=np.complex128),
                    float(tol),
                    int(max_iter),
                    int(restart_dim),
                    bool(accept_unconverged),
                    str(backend),
                    bool(reuse_static_w),
                    int(block_size),
                )
            else:
                result = owner.solve_two_site(
                    str(key),
                    np.asarray(E, dtype=np.complex128),
                    np.asarray(W1, dtype=np.complex128),
                    np.asarray(W2, dtype=np.complex128),
                    np.asarray(F, dtype=np.complex128),
                    np.asarray(A, dtype=np.complex128),
                    np.asarray(B, dtype=np.complex128),
                    float(tol),
                    int(max_iter),
                    int(restart_dim),
                    bool(accept_unconverged),
                    str(backend),
                    bool(reuse_static_w),
                )
        except Exception as exc:
            elapsed = float(time.perf_counter() - start)
            self.moving_profile_stats["dense_cpp_sweep_workspace_failures"] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["dense_cpp_sweep_workspace_last_error"] = str(exc)
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_two_site_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_two_site_solve_rejections",
                    0,
                )
            ) + 1
            self.moving_profile_stats["dense_solve_local_seconds"] = float(
                self.moving_profile_stats.get("dense_solve_local_seconds", 0.0)
            ) + elapsed
            return None
        elapsed = float(time.perf_counter() - start)
        self.moving_profile_stats["dense_cpp_sweep_workspace_two_site_solve_calls"] = int(
            self.moving_profile_stats.get(
                "dense_cpp_sweep_workspace_two_site_solve_calls",
                0,
            )
        ) + 1
        self.moving_profile_stats[
            "dense_cpp_sweep_workspace_two_site_solve_seconds"
        ] = float(
            self.moving_profile_stats.get(
                "dense_cpp_sweep_workspace_two_site_solve_seconds",
                0.0,
            )
        ) + elapsed
        self.moving_profile_stats["solve_local_seconds"] = float(
            self.moving_profile_stats.get("solve_local_seconds", 0.0)
        ) + elapsed
        self.moving_profile_stats["solve_local_last_seconds"] = elapsed
        self.moving_profile_stats["dense_solve_local_seconds"] = float(
            self.moving_profile_stats.get("dense_solve_local_seconds", 0.0)
        ) + elapsed
        self.moving_profile_stats["dense_solve_local_last_seconds"] = elapsed
        self.moving_profile_stats[
            "dense_operatorless_local_problem_solve_seconds"
        ] = float(
            self.moving_profile_stats.get(
                "dense_operatorless_local_problem_solve_seconds",
                0.0,
            )
        ) + elapsed
        self.moving_profile_stats[
            "dense_operatorless_local_problem_solve_last_seconds"
        ] = elapsed
        try:
            stats = dict(owner.stats())
            self.moving_profile_stats["dense_cpp_sweep_workspace_records"] = int(
                stats.get("records", 0)
            )
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_two_site_static_w_reuses"
            ] = int(stats.get("two_site_static_w_reuses", 0))
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_two_site_mpo_builds"
            ] = int(stats.get("two_site_mpo_builds", 0))
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_two_site_mps_builds"
            ] = int(stats.get("two_site_mps_builds", 0))
            self.moving_profile_stats["dense_cpp_sweep_workspace_binds"] = int(
                stats.get("bind_calls", 0)
            )
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_boundary_binds"
            ] = int(stats.get("boundary_bind_calls", 0))
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_block_solve_bound_calls"
            ] = int(stats.get("block_solve_bound_calls", 0))
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_two_site_block_solve_calls"
            ] = int(stats.get("two_site_block_solve_calls", 0))
        except Exception:
            pass
        if result is None or not bool(result.get("accepted", False)):
            self.moving_profile_stats["solve_local_rejections"] = int(
                self.moving_profile_stats.get("solve_local_rejections", 0)
            ) + 1
            self.moving_profile_stats["dense_solve_local_rejections"] = int(
                self.moving_profile_stats.get("dense_solve_local_rejections", 0)
            ) + 1
            self.moving_profile_stats[
                "dense_operatorless_local_problem_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "dense_operatorless_local_problem_solve_rejections",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_two_site_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_two_site_solve_rejections",
                    0,
                )
            ) + 1
            return None
        vector = np.asarray(result["vector"]).reshape(-1)
        energy = float(result["energy"])
        meta = dict(result)
        meta.pop("vector", None)
        record_stats = {}
        try:
            record_stats = dict(owner.record_stats(str(key)))
        except Exception:
            record_stats = {}
        if record_stats:
            meta["stats"] = record_stats
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_batched_matvec_calls"
            ] = int(record_stats.get("batched_matvec_calls", 0))
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_batched_matvec_vectors"
            ] = int(record_stats.get("batched_matvec_vectors", 0))
        solver_kind = str(meta.get("kind", "cpp_dense_davidson"))
        self._dense_local_profile_stats = {
            "bond": self.bond,
            "matvec_calls": 0,
            "matvec_seconds": 0.0,
            "paths": {},
            "local_solver": {},
            "cpp_dense_davidson": meta,
        }
        matvec_calls = int(meta.get("matvec_calls", 0))
        matvec_seconds = float(meta.get("seconds", elapsed))
        self._dense_local_profile_stats["local_solver"] = {
            "kind": solver_kind,
            "dimension": int(vector.size),
            "roots": int(nstates),
            "seconds": elapsed,
            "tol": float(tol),
            "max_iter": int(max_iter),
            "backend": str(meta.get("backend", backend)),
            "iterations": int(meta.get("iterations", 0)),
            "residual_norm": float(meta.get("residual_norm", np.nan)),
            "workspace_reused": bool(meta.get("workspace_reused", False)),
            "matvec_calls": matvec_calls,
            "operatorless": True,
            "two_site_solver": True,
            "block_davidson": bool(meta.get("block_davidson", False)),
            "block_size": int(meta.get("block_size", 1)),
        }
        self._record_dense_operatorless_path(
            (
                "dense_cpp_block_davidson_"
                if bool(meta.get("block_davidson", False))
                else "dense_cpp_davidson_"
            )
            + str(meta.get("backend", backend)),
            matvec_seconds,
            matvec_calls,
        )
        self.moving_profile_stats["solve_local_accepts"] = int(
            self.moving_profile_stats.get("solve_local_accepts", 0)
        ) + 1
        self.moving_profile_stats["dense_solve_local_accepts"] = int(
            self.moving_profile_stats.get("dense_solve_local_accepts", 0)
        ) + 1
        self.moving_profile_stats[
            "dense_operatorless_local_problem_solve_accepts"
        ] = int(
            self.moving_profile_stats.get(
                "dense_operatorless_local_problem_solve_accepts",
                0,
            )
        ) + 1
        self.moving_profile_stats[
            "dense_cpp_sweep_workspace_two_site_solve_accepts"
        ] = int(
            self.moving_profile_stats.get(
                "dense_cpp_sweep_workspace_two_site_solve_accepts",
                0,
            )
        ) + 1
        if bool(meta.get("block_davidson", False)):
            self.moving_profile_stats[
                "dense_cpp_sweep_workspace_block_davidson_accepts"
            ] = int(
                self.moving_profile_stats.get(
                    "dense_cpp_sweep_workspace_block_davidson_accepts",
                    0,
                )
            ) + 1
        return np.array([energy]), vector[:, None]

    def dense_cpp_workspace_record_stats(self, key):
        owner = self._dense_cpp_sweep_workspace
        if owner is None or key is None:
            return {}
        try:
            return dict(owner.record_stats(str(key)))
        except Exception:
            return {}

    def split_dense_single_state_cpp(
        self,
        flat,
        *,
        chi_left,
        phys_left,
        phys_right,
        chi_right,
        m_max,
        direction,
    ):
        if (
            _cpp_davidson is None
            or not bool(getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False))
            or getattr(_cpp_davidson, "lapack_svd", None) is None
        ):
            return None
        if not bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_dense_cpp_split",
                True,
            )
        ):
            return None
        self.moving_profile_stats["dense_cpp_split_calls"] = int(
            self.moving_profile_stats.get("dense_cpp_split_calls", 0)
        ) + 1
        start = time.perf_counter()
        try:
            theta = np.asarray(flat, dtype=np.complex128).reshape(
                int(chi_left),
                int(phys_left),
                int(phys_right),
                int(chi_right),
            )
            matrix = theta.reshape(
                int(chi_left) * int(phys_left),
                int(phys_right) * int(chi_right),
            )
            U, S, V = _cpp_davidson.lapack_svd(matrix)
            U = np.asarray(U, dtype=np.complex128)
            S = np.asarray(S, dtype=float)
            V = np.asarray(V, dtype=np.complex128)
            U, S, V = _canonicalize_svd_pair(U, S, V)
            kept = min(int(len(S)), int(m_max))
            trunc = float(np.sum(S[kept:]))
            S = S[:kept]
            A = U[:, :kept].reshape(int(chi_left), int(phys_left), kept)
            B = V[:kept, :].reshape(kept, int(phys_right), int(chi_right))
            if str(direction) == "right":
                B = S[:, None, None] * B
            else:
                if str(direction) != "left":
                    raise ValueError(f"unknown dense split direction {direction!r}")
                A = A * S[None, None, :]
        except Exception as exc:
            self.moving_profile_stats["dense_cpp_split_failures"] = int(
                self.moving_profile_stats.get("dense_cpp_split_failures", 0)
            ) + 1
            self.moving_profile_stats["dense_cpp_split_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - start)
        self.moving_profile_stats["dense_cpp_split_accepts"] = int(
            self.moving_profile_stats.get("dense_cpp_split_accepts", 0)
        ) + 1
        self.moving_profile_stats["dense_cpp_split_seconds"] = float(
            self.moving_profile_stats.get("dense_cpp_split_seconds", 0.0)
        ) + elapsed
        self.moving_profile_stats["dense_cpp_split_last_seconds"] = elapsed
        return A, B, trunc, kept

    def set_dense_bond(self, E, W, F, *, bond=None, matvec_options=None):
        self.bond = None if bond is None else int(bond)
        options = self.matvec_options if matvec_options is None else matvec_options
        self._operatorless_local_problem_active = False
        self._dense_operatorless_local_problem_active = False
        self._dense_operatorless_key = None
        dense_cpp_key = self.bind_dense_cpp_workspace(
            E,
            W,
            F,
            bond=bond,
            matvec_options=options,
        )
        if (
            dense_cpp_key is not None
            and bool(
                self._option_value(
                    options,
                    "moving_environment_dense_cpp_operatorless",
                    True,
                )
            )
        ):
            self.operator = None
            self._dense_operatorless_local_problem_active = True
            self._dense_operatorless_key = dense_cpp_key
            self._dense_local_profile_stats = {
                "bond": self.bond,
                "matvec_calls": 0,
                "matvec_seconds": 0.0,
                "paths": {},
                "local_solver": {},
                "cpp_dense_davidson": {},
            }
            self.moving_profile_stats["dense_operatorless_local_problem_binds"] = int(
                self.moving_profile_stats.get(
                    "dense_operatorless_local_problem_binds",
                    0,
                )
            ) + 1
            return self
        reused = False
        reuse_enabled = bool(
            self._option_value(
                options,
                "moving_environment_reuse_local_operator",
                True,
            )
        )
        if reuse_enabled and isinstance(self.operator, DenseLocalProblem):
            reused = bool(
                self.operator.reset_local_problem(
                    E,
                    W,
                    F,
                    bond=bond,
                    matvec_options=options,
                )
            )
        if reused:
            self.moving_profile_stats["local_operator_reuses"] = int(
                self.moving_profile_stats.get("local_operator_reuses", 0)
            ) + 1
            self.moving_profile_stats["dense_local_operator_reuses"] = int(
                self.moving_profile_stats.get("dense_local_operator_reuses", 0)
            ) + 1
        else:
            self.operator = DenseLocalProblem(
                E,
                W,
                F,
                bond=bond,
                matvec_options=options,
            )
            self.moving_profile_stats["local_operator_builds"] = int(
                self.moving_profile_stats.get("local_operator_builds", 0)
            ) + 1
            self.moving_profile_stats["dense_local_operator_builds"] = int(
                self.moving_profile_stats.get("dense_local_operator_builds", 0)
            ) + 1
        self.operator._moving_environment = self
        self.operator._dense_cpp_sweep_workspace_key = dense_cpp_key
        return self

    def bind_owner_operatorless_local_problem(
        self,
        E,
        W,
        F,
        *,
        bond=None,
        complementary_operator_families=None,
        complementary_boundary_payloads=None,
        complementary_split_stats=None,
        complementary_family_environments=None,
        complementary_direct_family_environments=None,
        matvec_options=None,
    ):
        self.bond = None if bond is None else int(bond)
        families = (
            self.complementary_operator_families
            if complementary_operator_families is None
            else complementary_operator_families
        )
        options = self.matvec_options if matvec_options is None else matvec_options
        bound = self._bind_operatorless_local_problem(
            E,
            W,
            F,
            bond=bond,
            complementary_operator_families=families,
            complementary_boundary_payloads=complementary_boundary_payloads,
            complementary_split_stats=complementary_split_stats,
            complementary_family_environments=complementary_family_environments,
            complementary_direct_family_environments=(
                complementary_direct_family_environments
            ),
            matvec_options=options,
        )
        if not bound:
            self.moving_profile_stats["owner_operatorless_local_problem_rejections"] = int(
                self.moving_profile_stats.get(
                    "owner_operatorless_local_problem_rejections",
                    0,
                )
            ) + 1
            return self.set_bond(
                E,
                W,
                F,
                bond=bond,
                complementary_operator_families=families,
                complementary_boundary_payloads=complementary_boundary_payloads,
                complementary_split_stats=complementary_split_stats,
                complementary_family_environments=complementary_family_environments,
                complementary_direct_family_environments=(
                    complementary_direct_family_environments
                ),
                matvec_options=options,
            )
        self.operator = None
        self.moving_profile_stats["owner_operatorless_local_problem_binds"] = int(
            self.moving_profile_stats.get(
                "owner_operatorless_local_problem_binds",
                0,
            )
        ) + 1
        self.moving_profile_stats["owner_local_problem_bind_backend_actual"] = (
            "cpp_owner_operatorless_local_problem"
        )
        return self

    def local_operator(self):
        if (
            self._operatorless_local_problem_active
            or self._dense_operatorless_local_problem_active
        ):
            return self
        if self.operator is None:
            raise RuntimeError("MovingEnvironment has no active local operator")
        return self

    def solve_packed_davidson(
        self,
        v0,
        *,
        tol=1.0e-5,
        max_iter=30,
        preconditioner=None,
        current=None,
        return_flat=False,
        initial_flat=None,
        initial_layout=None,
        initial_is_current=False,
        return_update=False,
        update_direction="right",
        update_m_max=None,
    ):
        if not self._operatorless_local_problem_active:
            if self.operator is None:
                return None
            return self.operator.solve_packed_davidson(
                v0,
                tol=tol,
                max_iter=max_iter,
                preconditioner=preconditioner,
                current=current,
                return_flat=return_flat,
                initial_flat=initial_flat,
                initial_layout=initial_layout,
                initial_is_current=initial_is_current,
                return_update=return_update,
                update_direction=update_direction,
                update_m_max=update_m_max,
            )
        del preconditioner, current
        self.moving_profile_stats["operatorless_local_problem_solve_calls"] = int(
            self.moving_profile_stats.get(
                "operatorless_local_problem_solve_calls",
                0,
            )
        ) + 1
        self.last_packed_davidson_solution_flat = None
        self.last_packed_davidson_solution_layout = None
        self.last_packed_davidson_solution_energy = None
        self.last_packed_davidson_solution_residual = None
        self.last_packed_davidson_solution_converged = None
        self.last_packed_davidson_solution_update = None

        layout_map = {key: tuple(shape) for key, shape in self._layout(v0)}
        allowed_layout_map = None
        if self._packed_local_safe_layout_expansion:
            allowed_layout_map = self._safe_two_site_layout_map(v0)
            if allowed_layout_map is not None:
                for key, shape in layout_map.items():
                    if allowed_layout_map.get(key) != tuple(shape):
                        self._local_profile_stats["packed_local_davidson"] = {
                            "iterations": 0,
                            "dimension": int(self._size(self._layout_from_map(layout_map))),
                            "basis_size": 0,
                            "layout_blocks": int(len(layout_map)),
                            "converged": False,
                            "rejected_reason": "initial_layout_not_safe",
                            "operatorless": True,
                        }
                        self.moving_profile_stats[
                            "operatorless_local_problem_solve_rejections"
                        ] = int(
                            self.moving_profile_stats.get(
                                "operatorless_local_problem_solve_rejections",
                                0,
                            )
                        ) + 1
                        return None
                if self._packed_local_use_safe_closure:
                    layout_map = dict(allowed_layout_map)
        layout = self._layout_from_map(layout_map)
        dim = int(self._size(layout))
        if dim <= 0 or dim < int(self._packed_local_davidson_min_dim):
            self.moving_profile_stats[
                "operatorless_local_problem_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "operatorless_local_problem_solve_rejections",
                    0,
                )
            ) + 1
            return None
        active_max_dim = int(self._packed_local_davidson_max_dim)
        if active_max_dim > 0 and dim > active_max_dim:
            self._local_profile_stats["packed_local_davidson"] = {
                "iterations": 0,
                "dimension": int(dim),
                "basis_size": 0,
                "layout_blocks": int(len(layout)),
                "converged": False,
                "rejected_reason": "layout_too_large",
                "operatorless": True,
            }
            self.moving_profile_stats[
                "operatorless_local_problem_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "operatorless_local_problem_solve_rejections",
                    0,
                )
            ) + 1
            return None

        flat = None
        initial_flat_present = initial_flat is not None and initial_layout is not None
        if initial_flat_present:
            try:
                flat_layout = tuple(
                    (tuple(key), tuple(int(dim_i) for dim_i in shape))
                    for key, shape in tuple(initial_layout or ())
                )
                flat_vec = np.asarray(initial_flat)
                if flat_layout != tuple(layout):
                    flat_vec = self._remap_flat(flat_vec, flat_layout, layout)
                if int(flat_vec.size) == int(dim):
                    flat = np.asarray(
                        flat_vec.reshape(int(dim)),
                        dtype=np.complex128,
                    ).copy()
            except Exception as exc:
                self.moving_profile_stats[
                    "operatorless_local_problem_initial_flat_last_error"
                ] = str(exc)
                flat = None
        if flat is None:
            flat = np.asarray(
                self._flatten(v0, layout),
                dtype=np.complex128,
            ).reshape(int(dim))
        norm = float(np.linalg.norm(flat))
        if norm < 1.0e-12:
            self.moving_profile_stats[
                "operatorless_local_problem_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "operatorless_local_problem_solve_rejections",
                    0,
                )
            ) + 1
            return None

        restart_dim = int(self._packed_local_davidson_restart_dim)
        accept_unconverged = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_accept_unconverged",
                False,
            )
        )
        if return_update:
            solved = self.solve_cpp_davidson_update(
                self,
                v0,
                layout,
                flat,
                tol=float(tol),
                max_iter=int(max_iter),
                restart_dim=restart_dim,
                accept_unconverged=accept_unconverged,
                direction=update_direction,
                m_max=update_m_max,
            )
            if solved is not None:
                result, update = solved
            else:
                result, update = None, None
        else:
            result = self.solve_cpp_davidson(
                self,
                v0,
                layout,
                flat,
                tol=float(tol),
                max_iter=int(max_iter),
                restart_dim=restart_dim,
                accept_unconverged=accept_unconverged,
            )
            update = None
        if result is None or not bool(result.get("accepted", False)):
            self.moving_profile_stats[
                "operatorless_local_problem_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "operatorless_local_problem_solve_rejections",
                    0,
                )
            ) + 1
            return None

        vector = np.asarray(result.get("vector"), dtype=np.complex128).reshape(dim)
        normalized = abelian_normalize_flat_vector(vector, min_norm=1.0e-12)
        if not normalized.accepted:
            return None
        energy = complex(result.get("energy"))
        residual = float(result.get("residual_norm", math.inf))
        converged = bool(result.get("converged", False))
        self.last_packed_davidson_solution_flat = np.asarray(
            normalized.vector,
            dtype=np.complex128,
        ).copy()
        self.last_packed_davidson_solution_layout = tuple(layout)
        self.last_packed_davidson_solution_energy = energy
        self.last_packed_davidson_solution_residual = residual
        self.last_packed_davidson_solution_converged = converged
        self.last_packed_davidson_solution_update = update
        self._local_profile_stats["packed_local_davidson"] = {
            "iterations": int(result.get("iterations", 0)),
            "dimension": int(dim),
            "basis_size": int(result.get("basis_size", 0)),
            "layout_blocks": int(len(layout)),
            "layout_expansions": 0,
            "restarts": int(result.get("restarts", 0)),
            "residual_norm": residual,
            "converged": converged,
            "cpp_davidson": True,
            "cpp_davidson_table_source": str(result.get("table_source", "")),
            "operatorless": True,
            "flat_solution_available": True,
            "flat_solution_layout_blocks": int(len(layout)),
            "initial_flat_guess_present": bool(
                initial_flat_present and not initial_is_current
            ),
            "initial_current_flat_present": bool(
                initial_flat_present and initial_is_current
            ),
        }
        self.moving_profile_stats["operatorless_local_problem_solve_accepts"] = int(
            self.moving_profile_stats.get(
                "operatorless_local_problem_solve_accepts",
                0,
            )
        ) + 1
        if return_update and update is not None:
            return energy, update
        if return_flat:
            return energy, self.last_packed_davidson_solution_flat
        return energy, self._unflatten(
            self.last_packed_davidson_solution_flat,
            v0,
            layout,
        )

    def solve_local(
        self,
        proto,
        *,
        operator=None,
        nstates=1,
        tol=1.0e-5,
        max_iter=30,
        preconditioner=None,
        current=None,
        return_flat=False,
        initial_flat=None,
        initial_layout=None,
        initial_is_current=False,
        return_update=False,
        update_direction="right",
        update_m_max=None,
    ):
        if int(nstates) != 1:
            return None
        local_operator = self.operator if operator is None else operator
        if isinstance(local_operator, MovingEnvironment):
            if local_operator._operatorless_local_problem_active:
                local_operator = local_operator
            else:
                local_operator = local_operator.operator
        if (
            local_operator is None
            and self._operatorless_local_problem_active
            and (operator is None or operator is self)
        ):
            local_operator = self
        if local_operator is None:
            raise RuntimeError("MovingEnvironment has no active local operator")
        self.moving_profile_stats["solve_local_calls"] = int(
            self.moving_profile_stats.get("solve_local_calls", 0)
        ) + 1
        if not bool(getattr(local_operator, "_packed_local_davidson", False)):
            self.moving_profile_stats["solve_local_rejections"] = int(
                self.moving_profile_stats.get("solve_local_rejections", 0)
            ) + 1
            self.moving_profile_stats["solve_local_rejected_reason"] = (
                "packed_local_davidson_disabled"
            )
            return None
        start = time.perf_counter()
        try:
            result = local_operator.solve_packed_davidson(
                proto,
                tol=float(tol),
                max_iter=int(max_iter),
                preconditioner=preconditioner,
                current=current,
                return_flat=return_flat,
                initial_flat=initial_flat,
                initial_layout=initial_layout,
                initial_is_current=initial_is_current,
                return_update=return_update,
                update_direction=update_direction,
                update_m_max=update_m_max,
            )
        finally:
            elapsed = float(time.perf_counter() - start)
            self.moving_profile_stats["solve_local_seconds"] = float(
                self.moving_profile_stats.get("solve_local_seconds", 0.0)
            ) + elapsed
            self.moving_profile_stats["solve_local_last_seconds"] = elapsed
        if result is None:
            self.moving_profile_stats["solve_local_rejections"] = int(
                self.moving_profile_stats.get("solve_local_rejections", 0)
            ) + 1
        else:
            self.moving_profile_stats["solve_local_accepts"] = int(
                self.moving_profile_stats.get("solve_local_accepts", 0)
            ) + 1
        return result

    def _record_dense_operatorless_path(self, name, elapsed, calls):
        paths = self._dense_local_profile_stats.setdefault("paths", {})
        entry = paths.setdefault(
            str(name),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + int(calls)
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)
        self._dense_local_profile_stats["matvec_calls"] = int(
            self._dense_local_profile_stats.get("matvec_calls", 0)
        ) + int(calls)
        self._dense_local_profile_stats["matvec_seconds"] = float(
            self._dense_local_profile_stats.get("matvec_seconds", 0.0)
        ) + float(elapsed)

    def _solve_dense_operatorless_cpp(
        self,
        AA,
        nstates,
        *,
        tol=1.0e-9,
        max_iter=5000,
    ):
        if int(nstates) != 1:
            self.moving_profile_stats[
                "dense_operatorless_local_problem_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "dense_operatorless_local_problem_solve_rejections",
                    0,
                )
            ) + 1
            return None
        key = self._dense_operatorless_key
        if key is None:
            return None
        restart_dim = int(
            self._option_value(
                self.matvec_options,
                "moving_environment_dense_cpp_davidson_restart_dim",
                min(max(8, int(max_iter)), 64),
            )
        )
        backend = str(
            self._option_value(
                self.matvec_options,
                "moving_environment_dense_cpp_davidson_backend",
                "blas",
            )
        )
        accept_unconverged = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_dense_cpp_davidson_accept_unconverged",
                False,
            )
        )
        solver_start = time.perf_counter()
        result = self.solve_dense_cpp_workspace(
            key,
            AA,
            tol=float(tol),
            max_iter=int(max_iter),
            restart_dim=restart_dim,
            accept_unconverged=accept_unconverged,
            backend=backend,
        )
        solver_seconds = float(time.perf_counter() - solver_start)
        if result is None or not bool(result.get("accepted", False)):
            self.moving_profile_stats[
                "dense_operatorless_local_problem_solve_rejections"
            ] = int(
                self.moving_profile_stats.get(
                    "dense_operatorless_local_problem_solve_rejections",
                    0,
                )
            ) + 1
            rejection = {} if result is None else dict(result)
            rejection.pop("vector", None)
            self._dense_local_profile_stats["cpp_dense_davidson_last_rejection"] = (
                rejection
            )
            return None
        vector = np.asarray(result["vector"]).reshape(-1)
        energy = float(result["energy"])
        meta = dict(result)
        meta.pop("vector", None)
        cpp_stats = self.dense_cpp_workspace_record_stats(key)
        meta["stats"] = cpp_stats
        self._dense_local_profile_stats["cpp_dense_davidson"] = meta
        nloc = int(np.asarray(AA).size)
        matvec_calls = int(meta.get("matvec_calls", 0))
        matvec_seconds = float(meta.get("seconds", solver_seconds))
        self._dense_local_profile_stats["local_solver"] = {
            "kind": "cpp_dense_davidson",
            "dimension": int(nloc),
            "roots": int(nstates),
            "seconds": solver_seconds,
            "tol": float(tol),
            "max_iter": int(max_iter),
            "backend": str(meta.get("backend", backend)),
            "iterations": int(meta.get("iterations", 0)),
            "residual_norm": float(meta.get("residual_norm", np.nan)),
            "workspace_reused": bool(meta.get("workspace_reused", False)),
            "matvec_calls": matvec_calls,
            "operatorless": True,
        }
        self._record_dense_operatorless_path(
            "dense_cpp_davidson_" + str(meta.get("backend", backend)),
            matvec_seconds,
            matvec_calls,
        )
        self.moving_profile_stats[
            "dense_operatorless_local_problem_solve_accepts"
        ] = int(
            self.moving_profile_stats.get(
                "dense_operatorless_local_problem_solve_accepts",
                0,
            )
        ) + 1
        return np.array([energy]), vector[:, None]

    def solve_dense_local(
        self,
        AA,
        *,
        nstates=1,
        tol=1.0e-9,
        max_iter=5000,
        operator=None,
    ):
        if operator is None and self._dense_operatorless_local_problem_active:
            local_operator = self
        else:
            local_operator = self.operator if operator is None else operator
        if isinstance(local_operator, MovingEnvironment):
            if local_operator._dense_operatorless_local_problem_active:
                pass
            else:
                local_operator = local_operator.operator
        dense_operatorless = (
            isinstance(local_operator, MovingEnvironment)
            and local_operator._dense_operatorless_local_problem_active
        )
        if not dense_operatorless and not isinstance(local_operator, DenseLocalProblem):
            self.moving_profile_stats["dense_solve_local_rejections"] = int(
                self.moving_profile_stats.get("dense_solve_local_rejections", 0)
            ) + 1
            self.moving_profile_stats["dense_solve_local_rejected_reason"] = (
                "no_dense_local_problem"
            )
            return None
        self.moving_profile_stats["solve_local_calls"] = int(
            self.moving_profile_stats.get("solve_local_calls", 0)
        ) + 1
        self.moving_profile_stats["dense_solve_local_calls"] = int(
            self.moving_profile_stats.get("dense_solve_local_calls", 0)
        ) + 1
        if dense_operatorless:
            self.moving_profile_stats[
                "dense_operatorless_local_problem_solve_calls"
            ] = int(
                self.moving_profile_stats.get(
                    "dense_operatorless_local_problem_solve_calls",
                    0,
                )
            ) + 1
        start = time.perf_counter()
        try:
            if dense_operatorless:
                result = local_operator._solve_dense_operatorless_cpp(
                    AA,
                    int(nstates),
                    tol=float(tol),
                    max_iter=int(max_iter),
                )
            else:
                result = local_operator.solve(
                    AA,
                    int(nstates),
                    tol=float(tol),
                    maxiter=int(max_iter),
                )
        finally:
            elapsed = float(time.perf_counter() - start)
            self.moving_profile_stats["solve_local_seconds"] = float(
                self.moving_profile_stats.get("solve_local_seconds", 0.0)
            ) + elapsed
            self.moving_profile_stats["solve_local_last_seconds"] = elapsed
            self.moving_profile_stats["dense_solve_local_seconds"] = float(
                self.moving_profile_stats.get("dense_solve_local_seconds", 0.0)
            ) + elapsed
            self.moving_profile_stats["dense_solve_local_last_seconds"] = elapsed
            if dense_operatorless:
                self.moving_profile_stats[
                    "dense_operatorless_local_problem_solve_seconds"
                ] = float(
                    self.moving_profile_stats.get(
                        "dense_operatorless_local_problem_solve_seconds",
                        0.0,
                    )
                ) + elapsed
                self.moving_profile_stats[
                    "dense_operatorless_local_problem_solve_last_seconds"
                ] = elapsed
        if result is None:
            self.moving_profile_stats["solve_local_rejections"] = int(
                self.moving_profile_stats.get("solve_local_rejections", 0)
            ) + 1
            self.moving_profile_stats["dense_solve_local_rejections"] = int(
                self.moving_profile_stats.get("dense_solve_local_rejections", 0)
            ) + 1
            return None
        self.moving_profile_stats["solve_local_accepts"] = int(
            self.moving_profile_stats.get("solve_local_accepts", 0)
        ) + 1
        self.moving_profile_stats["dense_solve_local_accepts"] = int(
            self.moving_profile_stats.get("dense_solve_local_accepts", 0)
        ) + 1
        return result

    def split_flat_two_site_svd_data(
        self,
        flat,
        layout,
        *,
        qns=None,
        dirs=None,
        direction="right",
        m_max=None,
    ):
        env = self._cpp_moving_environment
        enabled = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_site_split_owner",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        )
        if (
            enabled
            and env is not None
            and hasattr(env, "split_flat_two_site_svd_data")
        ):
            try:
                split = abelian_split_flat_two_site_svd_data_from_kernel(
                    env.split_flat_two_site_svd_data,
                    flat,
                    layout,
                    qns=qns,
                    dirs=dirs,
                    direction=direction,
                    m_max=m_max,
                )
            except Exception as exc:
                self.moving_profile_stats[
                    "cpp_moving_environment_site_split_flat_failures"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_moving_environment_site_split_flat_failures",
                        0,
                    )
                ) + 1
                self.moving_profile_stats[
                    "cpp_moving_environment_site_split_flat_last_error"
                ] = str(exc)
            else:
                self.moving_profile_stats[
                    "cpp_moving_environment_site_split_backend"
                ] = "cpp_moving_environment"
                self._sync_cpp_moving_environment_stats()
                return split
            finally:
                self._sync_cpp_moving_environment_stats()
        self.moving_profile_stats[
            "cpp_moving_environment_site_split_backend"
        ] = "free_function"
        return abelian_split_flat_two_site_svd_data(
            flat,
            layout,
            qns=qns,
            dirs=dirs,
            direction=direction,
            m_max=m_max,
        )

    def split_flat_two_site_update(
        self,
        flat,
        layout,
        *,
        qns=None,
        dirs=None,
        direction="right",
        m_max=None,
    ):
        env = self._cpp_moving_environment
        enabled = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_site_update_owner",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        )
        if (
            enabled
            and env is not None
            and hasattr(env, "split_flat_two_site_update")
        ):
            try:
                packed_layout, sector_decoder = (
                    _pack_two_site_split_layout_integer_sector_ids(layout)
                )
                (
                    left,
                    right,
                    s_data,
                    bond_qns,
                    trunc,
                    kept,
                    _native_stats,
                ) = env.split_flat_two_site_update(
                    flat,
                    packed_layout,
                    tuple(tuple(axis) for axis in (qns or ())),
                    tuple(int(d) for d in (dirs or ())),
                    str(direction),
                    AbelianSiteTensorData,
                    m_max,
                    sector_decoder,
                )
            except Exception as exc:
                self.moving_profile_stats[
                    "cpp_moving_environment_site_update_flat_failures"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_moving_environment_site_update_flat_failures",
                        0,
                    )
                ) + 1
                self.moving_profile_stats[
                    "cpp_moving_environment_site_update_flat_last_error"
                ] = str(exc)
            else:
                self.moving_profile_stats[
                    "cpp_moving_environment_site_split_backend"
                ] = "cpp_moving_environment"
                self.moving_profile_stats[
                    "cpp_moving_environment_site_update_backend"
                ] = "cpp_moving_environment"
                self._sync_cpp_moving_environment_stats()
                return AbelianTwoSiteUpdateData(
                    left,
                    right,
                    OrderedDict(
                        (key, np.asarray(block))
                        for key, block in (s_data or {}).items()
                    ),
                    tuple(bond_qns or ()),
                    float(trunc),
                    int(kept),
                )
            finally:
                self._sync_cpp_moving_environment_stats()
        split = self.split_flat_two_site_svd_data(
            flat,
            layout,
            qns=qns,
            dirs=dirs,
            direction=direction,
            m_max=m_max,
        )
        self.moving_profile_stats[
            "cpp_moving_environment_site_update_backend"
        ] = "python_wrap"
        return abelian_site_tensors_from_split(split)

    def sweep_bonds(self, direction, n_sites, *, center_i=-1, last_i=-1):
        direction = str(direction)
        n_sites = int(n_sites)
        center_i = int(center_i)
        last_i = int(last_i)

        def _python_range():
            if direction == "lr":
                return tuple(range(0, n_sites - 2))
            if direction == "rl":
                return tuple(range(n_sites - 2, 0, -1))
            if direction == "recenter_left":
                ci = n_sites // 2 - 1 if center_i < 0 else center_i
                return tuple(range(n_sites - 2, ci - 1, -1))
            if direction == "recenter_right":
                ci = n_sites // 2 - 1 if center_i < 0 else center_i
                return tuple(range(0, ci + 1))
            raise ValueError(f"unknown MovingEnvironment sweep cursor direction: {direction}")

        env = self._cpp_moving_environment
        enabled = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_sweep_cursor",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        )
        if enabled and env is not None and hasattr(env, "sweep_bonds"):
            try:
                bonds = tuple(
                    int(i)
                    for i in env.sweep_bonds(n_sites, direction, center_i, last_i)
                )
            except Exception as exc:
                self.moving_profile_stats[
                    "cpp_moving_environment_sweep_cursor_failures"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_moving_environment_sweep_cursor_failures",
                        0,
                    )
                ) + 1
                self.moving_profile_stats[
                    "cpp_moving_environment_sweep_cursor_last_error"
                ] = str(exc)
            else:
                self.moving_profile_stats[
                    "cpp_moving_environment_sweep_cursor_backend"
                ] = "cpp_moving_environment"
                self._sync_cpp_moving_environment_stats()
                return bonds
            finally:
                self._sync_cpp_moving_environment_stats()
        self.moving_profile_stats[
            "cpp_moving_environment_sweep_cursor_backend"
        ] = "python_range"
        return _python_range()

    def __getattr__(self, name):
        operator = self.__dict__.get("operator")
        if operator is None:
            raise AttributeError(name)
        return getattr(operator, name)

    @property
    def profile_stats(self):
        if self._operatorless_local_problem_active:
            return self._local_profile_stats
        if self.operator is None:
            return {}
        return self.operator.profile_stats

    @staticmethod
    def _option_value(options, name, default):
        if options is not None:
            if isinstance(options, dict) and name in options:
                return options[name]
            if not isinstance(options, dict) and hasattr(options, name):
                return getattr(options, name)
        return default

    def _use_table_flat_preconditioner(self, options):
        enabled = self._option_value(
            options,
            "moving_environment_flat_preconditioner",
            True,
        )
        if not bool(enabled):
            return False
        backend = str(
            self._option_value(
                options,
                "packed_local_family_flat_direct_matvec_backend",
                "",
            )
        ).strip().lower()
        if backend in {
            "block2",
            "block2_like",
            "block2-like",
            "block2_table",
            "renormalized",
            "renormalized_operator_table",
            "renormalized-operator-table",
        }:
            backend = "renormalized_table"
        return (
            bool(
                self._option_value(
                    options,
                    "packed_local_family_flat_direct_matvec",
                    False,
                )
            )
            and backend == "renormalized_table"
        )

    def cpp_renormalized_table(self, table, validation_vector=None):
        if table is None:
            return None
        cached = getattr(table, "_moving_environment_cpp_renormalized_table", None)
        if cached is not None:
            self.moving_profile_stats["cpp_renormalized_table_storage"] = str(
                getattr(table, "storage", "unknown")
            )
            return cached
        has_dense_blocks = getattr(table, "block_matrices", None) is not None
        has_sparse_blocks = getattr(table, "block_sparse_values", None) is not None
        if not (has_dense_blocks or has_sparse_blocks):
            return None
        if bool(getattr(table, "_moving_environment_cpp_renormalized_table_disabled", False)):
            return None
        if (
            _cpp_davidson is None
            or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
        ):
            return None
        if has_sparse_blocks:
            backend_cls = getattr(_cpp_davidson, "SparseRenormalizedTable", None)
        else:
            backend_cls = getattr(_cpp_davidson, "RenormalizedTable", None)
            if backend_cls is None:
                backend_cls = getattr(_cpp_davidson, "BlockTable", None)
        if backend_cls is None:
            return None
        try:
            if has_sparse_blocks:
                cpp_table = backend_cls(
                    table.block_sparse_rows,
                    table.block_sparse_cols,
                    table.block_sparse_values,
                    table.block_in_starts,
                    table.block_out_starts,
                    int(table.dim),
                )
            else:
                cpp_table = backend_cls(
                    table.block_matrices,
                    table.block_in_starts,
                    table.block_out_starts,
                    int(table.dim),
                )
        except Exception as exc:
            self.moving_profile_stats["cpp_renormalized_table_failures"] = int(
                self.moving_profile_stats.get("cpp_renormalized_table_failures", 0)
            ) + 1
            self.moving_profile_stats["cpp_renormalized_table_last_error"] = str(exc)
            return None
        setattr(table, "_moving_environment_cpp_renormalized_table", cpp_table)
        self.moving_profile_stats["cpp_renormalized_table_builds"] = int(
            self.moving_profile_stats.get("cpp_renormalized_table_builds", 0)
        ) + 1
        self.moving_profile_stats["cpp_renormalized_table_storage"] = str(
            getattr(table, "storage", "unknown")
        )
        if has_sparse_blocks:
            self.moving_profile_stats["cpp_sparse_renormalized_table_builds"] = int(
                self.moving_profile_stats.get(
                    "cpp_sparse_renormalized_table_builds",
                    0,
                )
            ) + 1
        # Compatibility counter for the old generic BlockTable reporting.
        self.moving_profile_stats["cpp_block_table_builds"] = int(
            self.moving_profile_stats.get("cpp_block_table_builds", 0)
        ) + 1
        if not self._validate_cpp_renormalized_table(
            table,
            cpp_table,
            validation_vector,
        ):
            return None
        return cpp_table

    def _validate_cpp_renormalized_table(self, table, cpp_table, validation_vector=None):
        validate = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec",
                True,
            )
        )
        if not validate:
            return True
        validated = getattr(
            table,
            "_moving_environment_cpp_renormalized_table_validated",
            None,
        )
        if validated is not None:
            return bool(validated)
        if validation_vector is None:
            dim = int(getattr(table, "dim", 0))
            if dim <= 0:
                return False
            validation_vector = np.zeros(dim, dtype=np.complex128)
            validation_vector[0] = 1.0
        vector = np.ascontiguousarray(validation_vector, dtype=np.complex128).reshape(
            int(table.dim)
        )
        try:
            ref = np.asarray(table.matvec(vector), dtype=np.complex128).reshape(
                int(table.dim)
            )
            test = np.asarray(cpp_table.matvec(vector), dtype=np.complex128).reshape(
                int(table.dim)
            )
        except Exception as exc:
            self.moving_profile_stats["cpp_renormalized_table_validation_failures"] = int(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_validation_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "cpp_renormalized_table_validation_last_error"
            ] = str(exc)
            setattr(table, "_moving_environment_cpp_renormalized_table_disabled", True)
            setattr(table, "_moving_environment_cpp_renormalized_table_validated", False)
            return False
        diff = float(np.linalg.norm(test - ref))
        scale = max(1.0, float(np.linalg.norm(ref)))
        rel = diff / scale
        tol = float(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec_tol",
                1.0e-10,
            )
        )
        self.moving_profile_stats["cpp_renormalized_table_validation_calls"] = int(
            self.moving_profile_stats.get("cpp_renormalized_table_validation_calls", 0)
        ) + 1
        self.moving_profile_stats[
            "cpp_renormalized_table_validation_last_error_norm"
        ] = diff
        self.moving_profile_stats[
            "cpp_renormalized_table_validation_last_relative_error"
        ] = rel
        if rel > tol:
            self.moving_profile_stats["cpp_renormalized_table_validation_failures"] = int(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_validation_failures",
                    0,
                )
            ) + 1
            setattr(table, "_moving_environment_cpp_renormalized_table_disabled", True)
            setattr(table, "_moving_environment_cpp_renormalized_table_validated", False)
            return False
        setattr(table, "_moving_environment_cpp_renormalized_table_validated", True)
        return True

    def cpp_block_table(self, table, validation_vector=None):
        if table is None or table.block_matrices is None:
            return None
        if bool(getattr(table, "_moving_environment_cpp_block_table_disabled", False)):
            return None
        if (
            _cpp_davidson is None
            or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
            or getattr(_cpp_davidson, "BlockTable", None) is None
        ):
            return None
        cached = getattr(table, "_moving_environment_cpp_block_table", None)
        if cached is not None:
            return cached
        try:
            cpp_table = _cpp_davidson.BlockTable(
                table.block_matrices,
                table.block_in_starts,
                table.block_out_starts,
                int(table.dim),
            )
        except Exception as exc:
            self.moving_profile_stats["cpp_block_table_failures"] = int(
                self.moving_profile_stats.get("cpp_block_table_failures", 0)
            ) + 1
            self.moving_profile_stats["cpp_block_table_last_error"] = str(exc)
            return None
        setattr(table, "_moving_environment_cpp_block_table", cpp_table)
        self.moving_profile_stats["cpp_block_table_builds"] = int(
            self.moving_profile_stats.get("cpp_block_table_builds", 0)
        ) + 1
        if not self._validate_cpp_block_table(table, cpp_table, validation_vector):
            return None
        return cpp_table

    def _validate_cpp_block_table(self, table, cpp_table, validation_vector=None):
        validate = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec",
                True,
            )
        )
        if not validate:
            return True
        validated = getattr(table, "_moving_environment_cpp_block_table_validated", None)
        if validated is not None:
            return bool(validated)
        if validation_vector is None:
            dim = int(getattr(table, "dim", 0))
            if dim <= 0:
                return False
            validation_vector = np.zeros(dim, dtype=np.complex128)
            validation_vector[0] = 1.0
        vector = np.ascontiguousarray(validation_vector, dtype=np.complex128).reshape(
            int(table.dim)
        )
        try:
            ref = np.asarray(table.matvec(vector), dtype=np.complex128).reshape(
                int(table.dim)
            )
            test = np.asarray(cpp_table.matvec(vector), dtype=np.complex128).reshape(
                int(table.dim)
            )
        except Exception as exc:
            self.moving_profile_stats["cpp_block_table_validation_failures"] = int(
                self.moving_profile_stats.get(
                    "cpp_block_table_validation_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["cpp_block_table_validation_last_error"] = str(exc)
            setattr(table, "_moving_environment_cpp_block_table_disabled", True)
            setattr(table, "_moving_environment_cpp_block_table_validated", False)
            return False
        diff = float(np.linalg.norm(test - ref))
        scale = max(1.0, float(np.linalg.norm(ref)))
        rel = diff / scale
        tol = float(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec_tol",
                1.0e-10,
            )
        )
        self.moving_profile_stats["cpp_block_table_validation_calls"] = int(
            self.moving_profile_stats.get("cpp_block_table_validation_calls", 0)
        ) + 1
        self.moving_profile_stats["cpp_block_table_validation_last_error_norm"] = diff
        self.moving_profile_stats["cpp_block_table_validation_last_relative_error"] = rel
        if rel > tol:
            self.moving_profile_stats["cpp_block_table_validation_failures"] = int(
                self.moving_profile_stats.get(
                    "cpp_block_table_validation_failures",
                    0,
                )
            ) + 1
            setattr(table, "_moving_environment_cpp_block_table_disabled", True)
            setattr(table, "_moving_environment_cpp_block_table_validated", False)
            return False
        setattr(table, "_moving_environment_cpp_block_table_validated", True)
        return True

    def _operator_cache_key(self, operator, proto, layout):
        left = None
        right = None
        payloads = getattr(operator, "complementary_boundary_payloads", None) or {}
        if payloads:
            left = payloads.get("left")
            right = payloads.get("right")
        family_env_tokens = []
        for name, env in sorted(
            (getattr(operator, "complementary_family_environments", None) or {}).items()
        ):
            try:
                E, W, F = env
            except Exception:
                family_env_tokens.append((str(name), id(env)))
                continue
            family_env_tokens.append(
                (
                    str(name),
                    HamiltonianMultiplyU1._component_action_token(E, W, F),
                )
            )
        direct_family_env_tokens = []
        for name, entries in sorted(
            (
                getattr(
                    operator,
                    "complementary_direct_family_environments",
                    None,
                )
                or {}
            ).items(),
            key=lambda item: str(item[0]),
        ):
            entry_groups = tuple(getattr(entries, "entry_groups", ()) or ())
            direct_family_env_tokens.append(
                (
                    str(name),
                    id(entries),
                    int(len(entries)),
                    tuple(int(len(group)) for group in entry_groups),
                )
            )
        return (
            "moving_environment_renormalized_operator_table",
            None if operator.bond is None else int(operator.bond),
            tuple(layout),
            tuple(proto.dirs),
            id(getattr(operator, "complementary_operator_families", None)),
            id(left),
            id(right),
            tuple(family_env_tokens),
            tuple(direct_family_env_tokens),
            int(getattr(self, "direct_family_revision", 0)),
            int(operator._renormalized_operator_table_dense_block_max_elements),
            float(operator._renormalized_operator_table_sparse_density_threshold),
        )

    def _use_cpp_grouped_renormalized_table(self, operator):
        enabled_opt = self._option_value(
            self.matvec_options,
            "moving_environment_cpp_grouped_renormalized_table",
            None,
        )
        if enabled_opt is None:
            enabled = bool(
                self._option_value(
                    self.matvec_options,
                    "moving_environment_cpp_davidson",
                    False,
                )
                or self._option_value(
                    self.matvec_options,
                    "moving_environment_cpp_matvec",
                    False,
                )
            )
        else:
            enabled = bool(enabled_opt)
        if not enabled:
            return False
        if (
            _cpp_davidson is None
            or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
            or (
                getattr(_cpp_davidson, "GroupedFactorizedTable", None) is None
                and getattr(_cpp_davidson, "GroupedRenormalizedTable", None) is None
            )
        ):
            return False
        if not bool(getattr(operator, "_packed_local_family_flat_direct_matvec", False)):
            return False
        backend = str(
            getattr(operator, "_packed_local_family_flat_direct_matvec_backend", "")
        )
        return backend == "renormalized_table"

    def _renormalized_operator_table_structural_key(
        self,
        operator,
        proto,
        layout,
        collected,
    ):
        raw_builder = collected.get("raw_builder")
        if "group_dims_array" in collected:
            dims = np.asarray(collected.get("group_dims_array"), dtype=np.int64)
            in_starts = np.asarray(collected.get("group_in_starts_array"), dtype=np.int64)
            out_starts = np.asarray(collected.get("group_out_starts_array"), dtype=np.int64)
            left_shapes = tuple(
                np.asarray(block).shape for block in collected.get("group_left", ())
            )
            right_shapes = tuple(
                np.asarray(block).shape for block in collected.get("group_right", ())
            )
            group_scales = collected.get("group_scales")
            scale_shapes = None
            if group_scales is not None:
                scale_shapes = tuple(
                    None if scale is None else tuple(np.asarray(scale).shape)
                    for scale in group_scales
                )
            payload_kind = "grouped"
        elif raw_builder is not None:
            dims = np.asarray(raw_builder.dims_array(), dtype=np.int64)
            in_starts = np.asarray(raw_builder.in_starts_array(), dtype=np.int64)
            out_starts = np.asarray(raw_builder.out_starts_array(), dtype=np.int64)
            left_shapes = tuple(tuple(shape) for shape in raw_builder.left_shapes())
            right_shapes = tuple(tuple(shape) for shape in raw_builder.right_shapes())
            scale_shapes = (int(raw_builder.size()),) if raw_builder.has_scales() else None
            payload_kind = "raw_builder"
        else:
            dims = np.asarray(collected.get("dims_array"), dtype=np.int64)
            in_starts = np.asarray(collected.get("in_starts_array"), dtype=np.int64)
            out_starts = np.asarray(collected.get("out_starts_array"), dtype=np.int64)
            left_shapes = tuple(
                np.asarray(block).shape for block in collected.get("left", ())
            )
            right_shapes = tuple(
                np.asarray(block).shape for block in collected.get("right", ())
            )
            scales = collected.get("scales_array")
            scale_shapes = None if scales is None else tuple(np.asarray(scales).shape)
            payload_kind = "raw"
        direct_kind = (
            "direct"
            if getattr(operator, "complementary_direct_family_environments", None)
            else "named"
        )
        return (
            "moving_environment_grouped_renormalized_operator_table",
            payload_kind,
            direct_kind,
            int(operator._size(layout)),
            tuple(np.asarray(dims, dtype=np.int64).reshape(-1).tolist()),
            tuple(np.asarray(in_starts, dtype=np.int64).reshape(-1).tolist()),
            tuple(np.asarray(out_starts, dtype=np.int64).reshape(-1).tolist()),
            left_shapes,
            right_shapes,
            scale_shapes,
            tuple(collected.get("family_names", ())),
            int(operator._renormalized_operator_table_dense_block_max_elements),
            float(operator._renormalized_operator_table_sparse_density_threshold),
        )

    def _record_renormalized_operator_table_build(self, operator, table, build_seconds):
        self.moving_profile_stats["renormalized_operator_table_builds"] = int(
            self.moving_profile_stats.get("renormalized_operator_table_builds", 0)
        ) + 1
        self.moving_profile_stats["renormalized_operator_table_build_seconds"] = float(
            self.moving_profile_stats.get(
                "renormalized_operator_table_build_seconds",
                0.0,
            )
        ) + float(build_seconds)
        stats = operator.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["renormalized_operator_table_builds"] = int(
            stats.get("renormalized_operator_table_builds", 0)
        ) + 1
        stats["renormalized_operator_table_entries"] = int(
            stats.get("renormalized_operator_table_entries", 0)
        ) + int(table.n_entries)
        stats["renormalized_operator_table_groups"] = int(
            stats.get("renormalized_operator_table_groups", 0)
        ) + int(table.n_groups)
        stats["renormalized_operator_table_group_channels"] = int(
            stats.get("renormalized_operator_table_group_channels", 0)
        ) + int(table.n_group_channels)
        stats["renormalized_operator_table_block_matrices"] = int(
            stats.get("renormalized_operator_table_block_matrices", 0)
        ) + int(table.n_block_matrices)
        stats["renormalized_operator_table_block_matrix_elements"] = int(
            stats.get("renormalized_operator_table_block_matrix_elements", 0)
        ) + int(table.block_matrix_elements)
        stats["renormalized_operator_table_block_sparse_nnz"] = int(
            stats.get("renormalized_operator_table_block_sparse_nnz", 0)
        ) + int(table.block_sparse_nnz)
        stats["renormalized_operator_table_last_storage"] = str(table.storage)

    def _record_renormalized_operator_table_refresh(
        self,
        operator,
        table,
        refresh_seconds,
        *,
        cache_kind="structural",
    ):
        self.moving_profile_stats["renormalized_operator_table_refreshes"] = int(
            self.moving_profile_stats.get("renormalized_operator_table_refreshes", 0)
        ) + 1
        self.moving_profile_stats["renormalized_operator_table_refresh_seconds"] = float(
            self.moving_profile_stats.get(
                "renormalized_operator_table_refresh_seconds",
                0.0,
            )
        ) + float(refresh_seconds)
        if str(cache_kind) == "slot":
            self.moving_profile_stats["renormalized_operator_table_slot_reuses"] = int(
                self.moving_profile_stats.get(
                    "renormalized_operator_table_slot_reuses",
                    0,
                )
            ) + 1
        else:
            self.moving_profile_stats[
                "renormalized_operator_table_structural_cache_hits"
            ] = int(
                self.moving_profile_stats.get(
                    "renormalized_operator_table_structural_cache_hits",
                    0,
                )
            ) + 1
        stats = operator.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        stats["renormalized_operator_table_refreshes"] = int(
            stats.get("renormalized_operator_table_refreshes", 0)
        ) + 1
        stats["renormalized_operator_table_last_storage"] = str(table.storage)

    def _record_cpp_grouped_renormalized_table_refresh_kind(self, table):
        kind = str(getattr(table, "last_refresh_kind", "unknown"))
        self.moving_profile_stats[
            "cpp_grouped_renormalized_table_last_refresh_kind"
        ] = kind
        if kind in {
            "dense_in_place",
            "raw_dense_in_place",
            "raw_dense_schedule_in_place",
            "factorized_refresh",
        }:
            key = "cpp_grouped_renormalized_table_fast_refreshes"
        elif kind in {"dense_rebuild_in_place", "raw_dense_rebuild_in_place"}:
            key = "cpp_grouped_renormalized_table_rebuild_in_place_refreshes"
        else:
            key = "cpp_grouped_renormalized_table_rebuild_refreshes"
        self.moving_profile_stats[key] = int(
            self.moving_profile_stats.get(key, 0)
        ) + 1

    def _refresh_cached_grouped_table_from_family_descriptor(
        self,
        cached,
        operator,
        proto,
        layout,
    ):
        if not bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_named_payload_fused_table_refresh",
                True,
            )
        ):
            return None
        env = self._cpp_moving_environment
        if env is None or cached is None:
            return None
        refresh = getattr(
            env,
            "refresh_grouped_table_from_family_mpo_descriptor",
            None,
        )
        if refresh is None:
            return None
        if getattr(operator, "complementary_direct_family_environments", None):
            return None
        descriptor_names = tuple(self._cpp_family_mpo_descriptor_names or ())
        descriptor_key = self._cpp_family_mpo_descriptor_key
        if (
            not descriptor_names
            or descriptor_key is None
            or operator.bond is None
            or not self.uses_cpp_family_mpo_descriptor()
            or not self.compiled_backend.use_cpp_named_raw_payload_plan()
        ):
            return None
        plan_cls = None if _cpp_davidson is None else getattr(
            _cpp_davidson,
            "NamedRawPayloadPlan",
            None,
        )
        if plan_cls is None:
            return None
        plan_key = (
            "moving_environment_cpp_named_raw_payload_plan_descriptor",
            int(operator.bond),
            tuple(proto.dirs),
            tuple(key for key, _shape in tuple(layout)),
            descriptor_names,
        )
        stats = self.moving_profile_stats
        plan = self._named_raw_payload_plan_cache.get(plan_key)
        try:
            if plan is None:
                build_start = time.perf_counter()
                plan = plan_cls()
                build_elapsed = float(time.perf_counter() - build_start)
                self._named_raw_payload_plan_cache[plan_key] = plan
                stats["cpp_named_raw_payload_plan_builds"] = int(
                    stats.get("cpp_named_raw_payload_plan_builds", 0)
                ) + 1
                stats["cpp_named_raw_payload_plan_build_seconds"] = float(
                    stats.get("cpp_named_raw_payload_plan_build_seconds", 0.0)
                ) + build_elapsed
                stats["cpp_named_raw_payload_plan_last_build_seconds"] = (
                    build_elapsed
                )
            else:
                stats["cpp_named_raw_payload_plan_cache_hits"] = int(
                    stats.get("cpp_named_raw_payload_plan_cache_hits", 0)
                ) + 1
            try:
                before_index_rebuilds = int(
                    dict(plan.stats()).get("index_rebuilds", 0)
                )
            except Exception:
                before_index_rebuilds = None
            info = dict(
                refresh(
                    cached.cpp_table,
                    plan,
                    descriptor_key,
                    int(operator.bond),
                    tuple(layout),
                    int(operator._size(layout)),
                    float(
                        operator._renormalized_operator_table_sparse_density_threshold
                    ),
                )
            )
            plan_stats = dict(info.get("plan_stats", {}) or {})
            payload_seconds = float(info.get("payload_seconds", 0.0) or 0.0)
            table_seconds = float(info.get("table_seconds", 0.0) or 0.0)
            total_seconds = float(
                info.get("seconds", payload_seconds + table_seconds) or 0.0
            )
            if before_index_rebuilds is not None:
                after_index_rebuilds = int(plan_stats.get("index_rebuilds", 0))
                index_rebuild_delta = max(
                    0,
                    after_index_rebuilds - before_index_rebuilds,
                )
                if index_rebuild_delta:
                    stats["cpp_named_raw_payload_plan_index_rebuilds"] = int(
                        stats.get("cpp_named_raw_payload_plan_index_rebuilds", 0)
                    ) + index_rebuild_delta
                    stats["cpp_named_raw_payload_plan_index_rebuild_seconds"] = (
                        float(
                            stats.get(
                                "cpp_named_raw_payload_plan_index_rebuild_seconds",
                                0.0,
                            )
                        )
                        + payload_seconds
                    )
                    stats[
                        "cpp_named_raw_payload_plan_last_index_rebuild_seconds"
                    ] = payload_seconds
            stats["cpp_named_raw_payload_plan_refresh_calls"] = int(
                stats.get("cpp_named_raw_payload_plan_refresh_calls", 0)
            ) + 1
            stats["cpp_named_raw_payload_plan_refresh_seconds"] = float(
                stats.get("cpp_named_raw_payload_plan_refresh_seconds", 0.0)
            ) + payload_seconds
            stats["cpp_named_raw_payload_plan_last_refresh_seconds"] = (
                payload_seconds
            )
            stats["cpp_named_raw_payload_plan_backend_actual"] = (
                "cpp_family_mpo_descriptor_fused_table"
            )
            stats["cpp_named_raw_payload_plan_fused_table_refreshes"] = int(
                stats.get("cpp_named_raw_payload_plan_fused_table_refreshes", 0)
            ) + 1
            stats["cpp_named_raw_payload_plan_fused_table_seconds"] = float(
                stats.get(
                    "cpp_named_raw_payload_plan_fused_table_seconds",
                    0.0,
                )
            ) + total_seconds
            stats["cpp_named_raw_payload_plan_fused_table_last_seconds"] = (
                total_seconds
            )
            for key, value in plan_stats.items():
                stats[f"cpp_named_raw_payload_plan_last_{key}"] = value
            stats["renormalized_operator_payload_collect_calls"] = int(
                stats.get("renormalized_operator_payload_collect_calls", 0)
            ) + 1
            stats["renormalized_operator_payload_collect_seconds"] = float(
                stats.get("renormalized_operator_payload_collect_seconds", 0.0)
            ) + payload_seconds
            stats["renormalized_operator_payload_collect_last_seconds"] = (
                payload_seconds
            )

            entry_count = int(info.get("entries", 0) or 0)
            cached.collected = {
                "raw_builder": None,
                "left": [],
                "right": [],
                "dims": [],
                "in_starts": [],
                "out_starts": [],
                "scales": [],
                "entry_count": entry_count,
                "family_names": descriptor_names,
                "matvec_groups": None,
                "raw_route_plan": None,
                "fused_descriptor_payload": True,
            }
            cached.dim = int(operator._size(layout))
            cached._set_vector_layout(
                layout,
                qns=operator._qns_from_layout_with_proto(layout, proto),
                dirs=proto.dirs,
            )
            cached.bond = int(operator.bond)
            cached.boundary_family_tables = tuple(
                operator._boundary_family_tables()
            )
            cached.storage = str(cached.cpp_table.storage())
            try:
                cached.last_refresh_kind = str(cached.cpp_table.last_refresh_kind())
            except Exception:
                cached.last_refresh_kind = str(
                    info.get("refresh_kind", "unknown")
                )
            cached.block_matrix_elements = int(
                cached.cpp_table.block_matrix_elements()
            )
            cached.block_sparse_nnz = int(cached.cpp_table.block_sparse_nnz())
            cached._diagonal_cache = None
            cached._moving_environment_cpp_renormalized_table = cached.cpp_table
            cached._moving_environment_cpp_renormalized_table_validated = True

            stats["cpp_grouped_renormalized_table_refreshes"] = int(
                stats.get("cpp_grouped_renormalized_table_refreshes", 0)
            ) + 1
            stats["cpp_grouped_renormalized_table_refresh_seconds"] = float(
                stats.get("cpp_grouped_renormalized_table_refresh_seconds", 0.0)
            ) + table_seconds
            stats["cpp_grouped_renormalized_table_slot_reuses"] = int(
                stats.get("cpp_grouped_renormalized_table_slot_reuses", 0)
            ) + 1
            stats["cpp_grouped_renormalized_table_bond_slot_reuses"] = int(
                stats.get("cpp_grouped_renormalized_table_bond_slot_reuses", 0)
            ) + 1
            stats["cpp_grouped_renormalized_table_last_storage"] = (
                cached.storage
            )
            stats["cpp_grouped_renormalized_table_last_blocks"] = int(
                cached.cpp_table.n_blocks()
            )
            stats["cpp_grouped_renormalized_table_last_elements"] = int(
                cached.cpp_table.block_matrix_elements()
            )
            stats["cpp_grouped_renormalized_table_last_sparse_nnz"] = int(
                cached.cpp_table.block_sparse_nnz()
            )
            for info_key, stat_key in (
                (
                    "raw_schedule_hits",
                    "cpp_grouped_renormalized_table_raw_schedule_hits",
                ),
                (
                    "raw_schedule_misses",
                    "cpp_grouped_renormalized_table_raw_schedule_misses",
                ),
                (
                    "raw_schedule_stores",
                    "cpp_grouped_renormalized_table_raw_schedule_stores",
                ),
            ):
                if info_key in info:
                    stats[stat_key] = int(info.get(info_key) or 0)
            self._record_cpp_grouped_renormalized_table_refresh_kind(cached)
            self._record_renormalized_operator_table_refresh(
                operator,
                cached,
                table_seconds,
                cache_kind="slot",
            )
            return cached
        except Exception as exc:
            stats["cpp_named_raw_payload_plan_fused_table_failures"] = int(
                stats.get("cpp_named_raw_payload_plan_fused_table_failures", 0)
            ) + 1
            stats["cpp_named_raw_payload_plan_fused_table_last_error"] = str(exc)
            return None

    def renormalized_operator_table(self, operator, proto, layout):
        layout = tuple(layout)

        def _bind_grouped_owner(table):
            self._install_cpp_moving_environment_grouped_table(
                table,
                operator,
                layout,
            )
            return table

        if self._use_cpp_grouped_renormalized_table(operator):
            collected = None
            route_key = None
            bond_slots_enabled = bool(
                self._option_value(
                    self.matvec_options,
                    "moving_environment_cpp_grouped_bond_slots",
                    False,
                )
            )
            if bond_slots_enabled:
                direct_kind = (
                    "direct"
                    if getattr(
                        operator,
                        "complementary_direct_family_environments",
                        None,
                    )
                    else "named"
                )
                layout_slot_key = tuple((key, tuple(shape)) for key, shape in layout)
                bond_slot_key = (
                    "moving_environment_grouped_renormalized_table_bond_layout_slot",
                    direct_kind,
                    None if operator.bond is None else int(operator.bond),
                    tuple(proto.dirs),
                    layout_slot_key,
                )
                cached = self._grouped_renormalized_table_bond_slots.get(
                    bond_slot_key
                )
                fused = self._refresh_cached_grouped_table_from_family_descriptor(
                    cached,
                    operator,
                    proto,
                    layout,
                )
                if fused is not None:
                    fused._moving_environment_structural_key = bond_slot_key
                    return _bind_grouped_owner(fused)
            if self.compiled_backend.use_cpp_raw_route_plan():
                try:
                    route_key = self.compiled_backend.raw_route_plan_cache_key(
                        operator,
                        proto,
                        layout,
                    )
                except Exception as exc:
                    route_key = None
                    self.moving_profile_stats[
                        "cpp_raw_route_plan_signature_failures"
                    ] = int(
                        self.moving_profile_stats.get(
                            "cpp_raw_route_plan_signature_failures",
                            0,
                        )
                    ) + 1
                    self.moving_profile_stats[
                        "cpp_raw_route_plan_signature_last_error"
                    ] = str(exc)
                route_plan = (
                    None if route_key is None else self._raw_route_plan_cache.get(route_key)
                )
                if route_plan is not None:
                    try:
                        collected = (
                            self.compiled_backend.collect_renormalized_operator_payload_from_route_plan(
                                operator,
                                route_plan,
                                proto,
                                layout,
                            )
                        )
                    except Exception as exc:
                        collected = None
                        self._raw_route_plan_cache.pop(route_key, None)
                        self.moving_profile_stats[
                            "cpp_raw_route_plan_refresh_failures"
                        ] = int(
                            self.moving_profile_stats.get(
                                "cpp_raw_route_plan_refresh_failures",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "cpp_raw_route_plan_refresh_last_error"
                        ] = str(exc)
                    else:
                        if collected is not None:
                            self.moving_profile_stats[
                                "cpp_raw_route_plan_cache_hits"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_raw_route_plan_cache_hits",
                                    0,
                                )
                            ) + 1
                elif route_key is not None:
                    self.moving_profile_stats[
                        "cpp_raw_route_plan_cache_misses"
                    ] = int(
                        self.moving_profile_stats.get(
                            "cpp_raw_route_plan_cache_misses",
                            0,
                        )
                    ) + 1
            if collected is None:
                collected = self.compiled_backend.collect_renormalized_operator_payload(
                    operator,
                    proto,
                    layout,
                )
                route_plan = None if collected is None else collected.get("raw_route_plan")
                if route_plan is not None:
                    if route_key is None:
                        try:
                            route_key = self.compiled_backend.raw_route_plan_cache_key(
                                operator,
                                proto,
                                layout,
                            )
                        except Exception:
                            route_key = None
                    if route_key is not None:
                        self._raw_route_plan_cache[route_key] = route_plan
                        self.moving_profile_stats["cpp_raw_route_plan_builds"] = int(
                            self.moving_profile_stats.get(
                                "cpp_raw_route_plan_builds",
                                0,
                            )
                        ) + 1
            if collected is None:
                return None
            groups = collected.get("matvec_groups")
            raw_payload = (
                groups is None
                and (
                    collected.get("raw_builder") is not None
                    or (
                        bool(collected.get("left"))
                        and bool(collected.get("right"))
                        and "dims_array" in collected
                        and "in_starts_array" in collected
                        and "out_starts_array" in collected
                    )
                )
            )
            if groups is not None or raw_payload:
                structural_key = None

                def _structural_key():
                    nonlocal structural_key
                    if structural_key is None:
                        structural_key = self._renormalized_operator_table_structural_key(
                            operator,
                            proto,
                            layout,
                            collected,
                        )
                    return structural_key

                structural_slots_enabled = bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_grouped_structural_slots",
                        True,
                    )
                )
                single_slot_enabled = bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_grouped_single_slot",
                        False,
                    )
                )
                bond_slots_enabled = bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_grouped_bond_slots",
                        False,
                    )
                )
                direct_kind = (
                    "direct"
                    if getattr(
                        operator,
                        "complementary_direct_family_environments",
                        None,
                    )
                    else "named"
                )
                layout_slot_key = tuple((key, tuple(shape)) for key, shape in layout)
                bond_slot_key = (
                    "moving_environment_grouped_renormalized_table_bond_layout_slot",
                    direct_kind,
                    None if operator.bond is None else int(operator.bond),
                    tuple(proto.dirs),
                    layout_slot_key,
                )
                if bond_slots_enabled:
                    cached = self._grouped_renormalized_table_bond_slots.get(
                        bond_slot_key
                    )
                    if cached is not None:
                        refresh_start = time.perf_counter()
                        try:
                            cached.refresh_from_collected(
                                collected,
                                dim=operator._size(layout),
                                layout=layout,
                                qns=operator._qns_from_layout_with_proto(layout, proto),
                                dirs=proto.dirs,
                                bond=operator.bond,
                                boundary_family_tables=operator._boundary_family_tables(),
                                sparse_density_threshold=(
                                    operator._renormalized_operator_table_sparse_density_threshold
                                ),
                            )
                            cached._moving_environment_structural_key = bond_slot_key
                        except Exception as exc:
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refresh_failures"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_refresh_failures",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refresh_last_error"
                            ] = str(exc)
                        else:
                            refresh_seconds = float(time.perf_counter() - refresh_start)
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refreshes"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_refreshes",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refresh_seconds"
                            ] = float(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_refresh_seconds",
                                    0.0,
                                )
                            ) + refresh_seconds
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_slot_reuses"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_slot_reuses",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_bond_slot_reuses"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_bond_slot_reuses",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_last_storage"
                            ] = str(cached.storage)
                            self._record_cpp_grouped_renormalized_table_refresh_kind(
                                cached
                            )
                            self._record_renormalized_operator_table_refresh(
                                operator,
                                cached,
                                refresh_seconds,
                                cache_kind="slot",
                            )
                            return _bind_grouped_owner(cached)
                if structural_slots_enabled and not bond_slots_enabled:
                    cached = self._incremental_renormalized_operator_table_cache.get(
                        _structural_key()
                    )
                    if cached is not None:
                        refresh_start = time.perf_counter()
                        try:
                            cached.refresh_from_collected(
                                collected,
                                dim=operator._size(layout),
                                layout=layout,
                                qns=operator._qns_from_layout_with_proto(layout, proto),
                                dirs=proto.dirs,
                                bond=operator.bond,
                                boundary_family_tables=operator._boundary_family_tables(),
                                sparse_density_threshold=(
                                    operator._renormalized_operator_table_sparse_density_threshold
                                ),
                            )
                            cached._moving_environment_structural_key = _structural_key()
                        except Exception as exc:
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refresh_failures"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_refresh_failures",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refresh_last_error"
                            ] = str(exc)
                        else:
                            refresh_seconds = float(time.perf_counter() - refresh_start)
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refreshes"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_refreshes",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_refresh_seconds"
                            ] = float(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_refresh_seconds",
                                    0.0,
                                )
                            ) + refresh_seconds
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_slot_reuses"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_slot_reuses",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_structural_slot_reuses"
                            ] = int(
                                self.moving_profile_stats.get(
                                    "cpp_grouped_renormalized_table_structural_slot_reuses",
                                    0,
                                )
                            ) + 1
                            self.moving_profile_stats[
                                "cpp_grouped_renormalized_table_last_storage"
                            ] = str(cached.storage)
                            self._record_cpp_grouped_renormalized_table_refresh_kind(
                                cached
                            )
                            self._record_renormalized_operator_table_refresh(
                                operator,
                                cached,
                                refresh_seconds,
                                cache_kind="slot",
                            )
                            return _bind_grouped_owner(cached)
                if (
                    single_slot_enabled
                    and self._grouped_renormalized_table_slot is not None
                ):
                    cached = self._grouped_renormalized_table_slot
                    refresh_start = time.perf_counter()
                    try:
                        cached.refresh_from_collected(
                            collected,
                            dim=operator._size(layout),
                            layout=layout,
                            qns=operator._qns_from_layout_with_proto(layout, proto),
                            dirs=proto.dirs,
                            bond=operator.bond,
                            boundary_family_tables=operator._boundary_family_tables(),
                            sparse_density_threshold=(
                                operator._renormalized_operator_table_sparse_density_threshold
                            ),
                        )
                        cached._moving_environment_structural_key = (
                            self._grouped_renormalized_table_slot_key
                        )
                    except Exception as exc:
                        self.moving_profile_stats[
                            "cpp_grouped_renormalized_table_refresh_failures"
                        ] = int(
                            self.moving_profile_stats.get(
                                "cpp_grouped_renormalized_table_refresh_failures",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "cpp_grouped_renormalized_table_refresh_last_error"
                        ] = str(exc)
                    else:
                        refresh_seconds = float(time.perf_counter() - refresh_start)
                        self.moving_profile_stats[
                            "cpp_grouped_renormalized_table_refreshes"
                        ] = int(
                            self.moving_profile_stats.get(
                                "cpp_grouped_renormalized_table_refreshes",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "cpp_grouped_renormalized_table_refresh_seconds"
                        ] = float(
                            self.moving_profile_stats.get(
                                "cpp_grouped_renormalized_table_refresh_seconds",
                                0.0,
                            )
                        ) + refresh_seconds
                        self.moving_profile_stats[
                            "cpp_grouped_renormalized_table_slot_reuses"
                        ] = int(
                            self.moving_profile_stats.get(
                                "cpp_grouped_renormalized_table_slot_reuses",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "cpp_grouped_renormalized_table_last_storage"
                        ] = str(cached.storage)
                        self._record_cpp_grouped_renormalized_table_refresh_kind(
                            cached
                        )
                        self._record_renormalized_operator_table_refresh(
                            operator,
                            cached,
                            refresh_seconds,
                            cache_kind="slot",
                        )
                        return _bind_grouped_owner(cached)
                build_start = time.perf_counter()
                table = self.compiled_backend.build_renormalized_operator_table(
                    operator,
                    collected,
                    proto,
                    layout,
                )
                build_seconds = float(time.perf_counter() - build_start)
                if table is None:
                    return None
                if hasattr(table, "refresh_from_collected"):
                    if bond_slots_enabled:
                        table._moving_environment_structural_key = bond_slot_key
                        self._grouped_renormalized_table_bond_slots[
                            bond_slot_key
                        ] = table
                    elif structural_slots_enabled:
                        table._moving_environment_structural_key = _structural_key()
                        self._incremental_renormalized_operator_table_cache[
                            _structural_key()
                        ] = table
                    elif single_slot_enabled:
                        table._moving_environment_structural_key = (
                            self._grouped_renormalized_table_slot_key
                        )
                        self._grouped_renormalized_table_slot = table
                self._record_renormalized_operator_table_build(
                    operator,
                    table,
                    build_seconds,
                )
                return _bind_grouped_owner(table)

        cache_key = self._operator_cache_key(operator, proto, layout)
        if cache_key in self._renormalized_operator_table_cache:
            table = self._renormalized_operator_table_cache[cache_key]
            if table is not None:
                self.moving_profile_stats["renormalized_operator_table_cache_hits"] = int(
                    self.moving_profile_stats.get("renormalized_operator_table_cache_hits", 0)
                ) + 1
            return table
        collected = self.compiled_backend.collect_renormalized_operator_payload(
            operator,
            proto,
            layout,
        )
        if collected is None:
            self._renormalized_operator_table_cache[cache_key] = None
            return None
        build_start = time.perf_counter()
        table = self.compiled_backend.build_renormalized_operator_table(
            operator,
            collected,
            proto,
            layout,
        )
        build_seconds = float(time.perf_counter() - build_start)
        self._renormalized_operator_table_cache[cache_key] = table
        self._record_renormalized_operator_table_build(
            operator,
            table,
            build_seconds,
        )
        return table

    def compiled_flat_matvec(self, operator, proto, layout):
        if operator.complementary_operator_families is None:
            return None
        if not bool(operator._packed_local_family_flat_matvec):
            return None
        if not bool(operator._packed_local_family_flat_direct_matvec):
            return None
        if operator._packed_local_family_flat_direct_matvec_backend != "renormalized_table":
            return None
        layout = tuple(layout)
        dim = int(operator._size(layout))
        min_dim = int(operator._packed_local_family_flat_direct_matvec_min_dim)
        if dim <= 0 or (min_dim > 0 and dim < min_dim):
            return None
        if self._use_cpp_grouped_renormalized_table(operator):
            table = self.renormalized_operator_table(operator, proto, layout)
            if table is None:
                return None
            single_compiled_slot = bool(
                self._option_value(
                    self.matvec_options,
                    "moving_environment_grouped_compiled_flat_matvec_single_slot",
                    True,
                )
            )
            cache_key = (
                "moving_environment_compiled_flat_matvec",
                self._grouped_renormalized_table_slot_key
                if single_compiled_slot
                else getattr(table, "_moving_environment_structural_key", None)
                or id(table),
            )
            cached = self._compiled_flat_matvec_cache.get(cache_key)
            if cached is not None:
                self.moving_profile_stats["compiled_flat_matvec_cache_hits"] = int(
                    self.moving_profile_stats.get("compiled_flat_matvec_cache_hits", 0)
                ) + 1
                return cached.bind_table(operator, table, layout, proto.dirs[:])
            compiled = MovingEnvironmentFlatMatvec(
                self,
                operator,
                table,
                layout,
                proto.dirs[:],
            )
            self._compiled_flat_matvec_cache[cache_key] = compiled
            self.moving_profile_stats["compiled_flat_matvec_builds"] = int(
                self.moving_profile_stats.get("compiled_flat_matvec_builds", 0)
            ) + 1
            return compiled
        cache_key = (
            "moving_environment_compiled_flat_matvec",
            self._operator_cache_key(operator, proto, layout),
        )
        cached = self._compiled_flat_matvec_cache.get(cache_key)
        if cached is not None:
            self.moving_profile_stats["compiled_flat_matvec_cache_hits"] = int(
                self.moving_profile_stats.get("compiled_flat_matvec_cache_hits", 0)
            ) + 1
            return cached.bind_operator(operator)
        table = self.renormalized_operator_table(operator, proto, layout)
        if table is None:
            self._compiled_flat_matvec_cache[cache_key] = None
            return None
        compiled = MovingEnvironmentFlatMatvec(
            self,
            operator,
            table,
            layout,
            proto.dirs[:],
        )
        self._compiled_flat_matvec_cache[cache_key] = compiled
        self.moving_profile_stats["compiled_flat_matvec_builds"] = int(
            self.moving_profile_stats.get("compiled_flat_matvec_builds", 0)
        ) + 1
        return compiled

    def flat_jacobi_diagonal(self, operator, proto, layout):
        compiled = self.compiled_flat_matvec(operator, proto, layout)
        if compiled is None:
            return None
        return compiled.diagonal()

    def _compact_block_table_enabled(self, operator, layout):
        enabled = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_compact_block_table",
                False,
            )
        )
        if not enabled or not bool(operator._packed_local_flat_matvec):
            return False
        max_dim = int(
            self._option_value(
                self.matvec_options,
                "moving_environment_compact_block_table_max_dim",
                0,
            )
        )
        dim = int(operator._size(tuple(layout)))
        return dim > 0 and (max_dim <= 0 or dim <= max_dim)

    def _compact_plan_operator_enabled(self, operator, layout):
        enabled = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_compact_plan",
                False,
            )
        )
        if not enabled or not bool(operator._packed_local_flat_matvec):
            return False
        max_dim = int(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_compact_plan_max_dim",
                0,
            )
        )
        dim = int(operator._size(tuple(layout)))
        return dim > 0 and (max_dim <= 0 or dim <= max_dim)

    @staticmethod
    def _compact_plan_validation_key(plan, layout, proto_dirs):
        return (
            "moving_environment_compact_plan_validation",
            tuple(layout),
            tuple(proto_dirs),
            tuple(plan["r_entries"]),
            tuple(plan["t2_entries"]),
            tuple(plan["t3_entries"]),
            tuple(plan["out_entries"]),
            tuple(plan["r_shapes"]),
            tuple(plan["t2_shapes"]),
            tuple(plan["t3_shapes"]),
            tuple(plan["out_shapes"]),
            tuple(tuple(shape) for shape in plan["a_groups"]["shapes"]),
            tuple(tuple(shape) for shape in plan["r_groups"]["shapes"]),
            tuple(tuple(shape) for shape in plan["t2_groups"]["shapes"]),
            tuple(tuple(shape) for shape in plan["t3_groups"]["shapes"]),
            tuple(tuple(shape) for shape in plan["out_groups"]["shapes"]),
            tuple(int(v) for v in plan["a_groups"]["block_group"]),
            tuple(int(v) for v in plan["a_groups"]["block_pos"]),
            tuple(int(v) for v in plan["out_groups"]["block_group"]),
            tuple(int(v) for v in plan["out_groups"]["block_pos"]),
        )

    @staticmethod
    def _tensor_structure_token(T):
        return tuple(
            (key, tuple(int(dim) for dim in np.asarray(block).shape))
            for key, block in sorted(T.data.items())
        )

    def _compact_renormalized_table_structure_key(self, operator, proto, layout):
        return (
            "moving_environment_compact_renormalized_table",
            None if operator.bond is None else int(operator.bond),
            tuple(layout),
            tuple(proto.dirs),
            self._tensor_structure_token(operator.E),
            self._tensor_structure_token(operator.W[0]),
            self._tensor_structure_token(operator.W[1]),
            self._tensor_structure_token(operator.F),
        )

    @staticmethod
    def _compact_renormalized_table_layout_token(layout):
        return tuple(
            (key, tuple(int(dim) for dim in shape))
            for key, shape in tuple(layout)
        )

    def _compact_renormalized_table_bond_slot_key(self, operator, proto, layout):
        return (
            "moving_environment_compact_renormalized_table_bond_slot",
            None if operator.bond is None else int(operator.bond),
            self._compact_renormalized_table_layout_token(layout),
            tuple(proto.dirs),
        )

    def _compact_renormalized_table_bond_slots_enabled(self):
        return bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_compact_plan_bond_slots",
                False,
            )
        )

    def _record_compact_renormalized_table_cache_hit(self, *, bond_slot=False):
        self.moving_profile_stats["compact_plan_cache_hits"] = int(
            self.moving_profile_stats.get("compact_plan_cache_hits", 0)
        ) + 1
        self.moving_profile_stats["compact_renormalized_table_cache_hits"] = int(
            self.moving_profile_stats.get(
                "compact_renormalized_table_cache_hits",
                0,
            )
        ) + 1
        if bond_slot:
            self.moving_profile_stats["compact_plan_bond_slot_hits"] = int(
                self.moving_profile_stats.get("compact_plan_bond_slot_hits", 0)
            ) + 1
            self.moving_profile_stats[
                "compact_renormalized_table_bond_slot_hits"
            ] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_bond_slot_hits",
                    0,
                )
            ) + 1

    def _record_compact_renormalized_table_refresh(
        self,
        table,
        refresh_seconds,
        *,
        bond_slot=False,
    ):
        self.moving_profile_stats["compact_plan_refreshes"] = int(
            self.moving_profile_stats.get("compact_plan_refreshes", 0)
        ) + 1
        self.moving_profile_stats["compact_renormalized_table_refreshes"] = int(
            self.moving_profile_stats.get(
                "compact_renormalized_table_refreshes",
                0,
            )
        ) + 1
        self.moving_profile_stats["compact_plan_refresh_seconds"] = float(
            self.moving_profile_stats.get("compact_plan_refresh_seconds", 0.0)
        ) + refresh_seconds
        self.moving_profile_stats["compact_renormalized_table_refresh_seconds"] = float(
            self.moving_profile_stats.get(
                "compact_renormalized_table_refresh_seconds",
                0.0,
            )
        ) + refresh_seconds
        self.moving_profile_stats["compact_plan_last_refresh_seconds"] = refresh_seconds
        self.moving_profile_stats[
            "compact_renormalized_table_last_refresh_seconds"
        ] = refresh_seconds
        refresh_backend = str(getattr(table, "last_refresh_backend", "unknown"))
        self.moving_profile_stats[
            "compact_renormalized_table_last_refresh_backend"
        ] = refresh_backend
        if refresh_backend == "cpp_block_refresh":
            self.moving_profile_stats[
                "compact_renormalized_table_cpp_block_refreshes"
            ] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_cpp_block_refreshes",
                    0,
                )
            ) + 1
        elif refresh_backend == "python_stack_refresh":
            self.moving_profile_stats[
                "compact_renormalized_table_python_stack_refreshes"
            ] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_python_stack_refreshes",
                    0,
                )
            ) + 1
        if bond_slot:
            self.moving_profile_stats["compact_plan_bond_slot_refreshes"] = int(
                self.moving_profile_stats.get(
                    "compact_plan_bond_slot_refreshes",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "compact_renormalized_table_bond_slot_reuses"
            ] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_bond_slot_reuses",
                    0,
                )
            ) + 1

    def _forget_compact_renormalized_table_structure(self, table):
        old_key = getattr(table, "structure_key", None)
        if (
            old_key is not None
            and self._compact_renormalized_table_cache.get(old_key) is table
        ):
            self._compact_renormalized_table_cache.pop(old_key, None)

    @staticmethod
    def _cpp_moving_environment_compact_key(operator):
        bond = None if operator.bond is None else int(operator.bond)
        return f"compact-plan-bond:{bond}"

    @staticmethod
    def _cpp_moving_environment_grouped_key(operator):
        bond = None if operator.bond is None else int(operator.bond)
        return f"grouped-table-bond:{bond}"

    def _sync_cpp_moving_environment_stats(self):
        env = self._cpp_moving_environment
        if env is None or not hasattr(env, "stats"):
            return
        try:
            stats = dict(env.stats())
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_last_stats_error"
            ] = str(exc)
            return
        mapping = {
            "compact_plan_records": "cpp_moving_environment_compact_plan_records",
            "compact_plan_installs": "cpp_moving_environment_compact_plan_installs",
            "compact_plan_replacements": "cpp_moving_environment_compact_plan_replacements",
            "compact_plan_matvec_calls": "cpp_moving_environment_compact_plan_matvec_calls",
            "compact_plan_diagonal_calls": "cpp_moving_environment_compact_plan_diagonal_calls",
            "compact_plan_diagonal_cache_hits": "cpp_moving_environment_compact_plan_diagonal_cache_hits",
            "compact_plan_davidson_calls": "cpp_moving_environment_compact_plan_davidson_calls",
            "compact_plan_davidson_workspace_reuses": "cpp_moving_environment_compact_plan_davidson_workspace_reuses",
            "grouped_table_records": "cpp_moving_environment_grouped_table_records",
            "grouped_table_installs": "cpp_moving_environment_grouped_table_installs",
            "grouped_table_replacements": "cpp_moving_environment_grouped_table_replacements",
            "grouped_table_matvec_calls": "cpp_moving_environment_grouped_table_matvec_calls",
            "grouped_table_diagonal_calls": "cpp_moving_environment_grouped_table_diagonal_calls",
            "grouped_table_diagonal_cache_hits": "cpp_moving_environment_grouped_table_diagonal_cache_hits",
            "grouped_table_davidson_calls": "cpp_moving_environment_grouped_table_davidson_calls",
            "grouped_table_davidson_workspace_reuses": "cpp_moving_environment_grouped_table_davidson_workspace_reuses",
            "grouped_table_dense_matvec_batch_rebuilds": "cpp_moving_environment_grouped_table_dense_matvec_batch_rebuilds",
            "grouped_table_dense_matvec_batches": "cpp_moving_environment_grouped_table_dense_matvec_batches",
            "grouped_table_dense_matvec_same_input_batches": "cpp_moving_environment_grouped_table_dense_matvec_same_input_batches",
            "grouped_table_dense_matvec_same_output_batches": "cpp_moving_environment_grouped_table_dense_matvec_same_output_batches",
            "grouped_table_dense_matvec_batched_blocks": "cpp_moving_environment_grouped_table_dense_matvec_batched_blocks",
            "grouped_table_dense_matvec_shape_groups": "cpp_moving_environment_grouped_table_dense_matvec_shape_groups",
            "grouped_table_dense_matvec_shape_group_blocks": "cpp_moving_environment_grouped_table_dense_matvec_shape_group_blocks",
            "grouped_table_dense_matvec_shape_group_elements": "cpp_moving_environment_grouped_table_dense_matvec_shape_group_elements",
            "grouped_table_dense_matvec_singleton_blocks": "cpp_moving_environment_grouped_table_dense_matvec_singleton_blocks",
            "grouped_table_dense_matvec_batched_elements": "cpp_moving_environment_grouped_table_dense_matvec_batched_elements",
            "grouped_table_dense_flat_csr_builds": "cpp_moving_environment_grouped_table_dense_flat_csr_builds",
            "grouped_table_dense_flat_csr_matvecs": "cpp_moving_environment_grouped_table_dense_flat_csr_matvecs",
            "grouped_table_dense_flat_csr_entries": "cpp_moving_environment_grouped_table_dense_flat_csr_entries",
            "grouped_table_dense_flat_csr_active_rows": "cpp_moving_environment_grouped_table_dense_flat_csr_active_rows",
            "grouped_table_dense_flat_csr_value_refreshes": "cpp_moving_environment_grouped_table_dense_flat_csr_value_refreshes",
            "grouped_table_dense_flat_csr_pattern_reuses": "cpp_moving_environment_grouped_table_dense_flat_csr_pattern_reuses",
            "grouped_table_dense_flat_csr_pattern_mismatches": "cpp_moving_environment_grouped_table_dense_flat_csr_pattern_mismatches",
            "site_split_flat_calls": "cpp_moving_environment_site_split_flat_calls",
            "site_split_flat_failures": "cpp_moving_environment_site_split_flat_failures",
            "site_split_flat_blocks": "cpp_moving_environment_site_split_flat_blocks",
            "site_split_flat_sectors": "cpp_moving_environment_site_split_flat_sectors",
            "site_split_flat_rows": "cpp_moving_environment_site_split_flat_rows",
            "site_split_flat_cols": "cpp_moving_environment_site_split_flat_cols",
            "site_split_flat_dim": "cpp_moving_environment_site_split_flat_dim",
            "site_update_flat_calls": "cpp_moving_environment_site_update_flat_calls",
            "site_update_flat_failures": "cpp_moving_environment_site_update_flat_failures",
            "site_update_flat_left_blocks": "cpp_moving_environment_site_update_flat_left_blocks",
            "site_update_flat_right_blocks": "cpp_moving_environment_site_update_flat_right_blocks",
            "site_update_flat_dim": "cpp_moving_environment_site_update_flat_dim",
            "solve_update_flat_calls": "cpp_moving_environment_solve_update_flat_calls",
            "solve_update_flat_accepted": "cpp_moving_environment_solve_update_flat_accepted",
            "solve_update_flat_failures": "cpp_moving_environment_solve_update_flat_failures",
            "sweep_cursor_plan_calls": "cpp_moving_environment_sweep_cursor_plan_calls",
            "sweep_cursor_lr_calls": "cpp_moving_environment_sweep_cursor_lr_calls",
            "sweep_cursor_rl_calls": "cpp_moving_environment_sweep_cursor_rl_calls",
            "sweep_cursor_recenter_calls": "cpp_moving_environment_sweep_cursor_recenter_calls",
            "sweep_cursor_steps": "cpp_moving_environment_sweep_cursor_steps",
            "sweep_cursor_last_n_sites": "cpp_moving_environment_sweep_cursor_last_n_sites",
            "sweep_cursor_last_steps": "cpp_moving_environment_sweep_cursor_last_steps",
            "environment_plan_records": "cpp_moving_environment_environment_plan_records",
            "environment_plan_builds": "cpp_moving_environment_environment_plan_builds",
            "environment_plan_build_seconds": "cpp_moving_environment_environment_plan_build_seconds",
            "environment_plan_replacements": "cpp_moving_environment_environment_plan_replacements",
            "environment_plan_cache_hits": "cpp_moving_environment_environment_plan_cache_hits",
            "environment_plan_advance_calls": "cpp_moving_environment_environment_plan_advance_calls",
            "environment_plan_advance_seconds": "cpp_moving_environment_environment_plan_advance_seconds",
            "environment_plan_failures": "cpp_moving_environment_environment_plan_failures",
            "environment_plan_last_routes": "cpp_moving_environment_environment_plan_last_routes",
            "environment_plan_last_blocks": "cpp_moving_environment_environment_plan_last_blocks",
            "environment_stack_records": "cpp_moving_environment_environment_stack_records",
            "environment_stack_resets": "cpp_moving_environment_environment_stack_resets",
            "environment_stack_pushes": "cpp_moving_environment_environment_stack_pushes",
            "environment_stack_pops": "cpp_moving_environment_environment_stack_pops",
            "environment_stack_apply_calls": "cpp_moving_environment_environment_stack_apply_calls",
            "environment_stack_apply_syncs": "cpp_moving_environment_environment_stack_apply_syncs",
            "environment_stack_apply_pushes": "cpp_moving_environment_environment_stack_apply_pushes",
            "environment_stack_apply_pops": "cpp_moving_environment_environment_stack_apply_pops",
            "environment_stack_apply_replaces": "cpp_moving_environment_environment_stack_apply_replaces",
            "environment_stack_apply_failures": "cpp_moving_environment_environment_stack_apply_failures",
            "environment_stack_failures": "cpp_moving_environment_environment_stack_failures",
            "environment_stack_last_depth": "cpp_moving_environment_environment_stack_last_depth",
            "family_mpo_descriptor_records": "cpp_moving_environment_family_mpo_descriptor_records",
            "family_mpo_descriptor_installs": "cpp_moving_environment_family_mpo_descriptor_installs",
            "family_mpo_descriptor_replacements": "cpp_moving_environment_family_mpo_descriptor_replacements",
            "family_mpo_descriptor_environment_builds": "cpp_moving_environment_family_mpo_descriptor_environment_builds",
            "family_mpo_descriptor_payload_builds": "cpp_moving_environment_family_mpo_descriptor_payload_builds",
            "family_mpo_descriptor_failures": "cpp_moving_environment_family_mpo_descriptor_failures",
            "family_mpo_descriptor_last_families": "cpp_moving_environment_family_mpo_descriptor_last_families",
            "family_mpo_descriptor_last_bond": "cpp_moving_environment_family_mpo_descriptor_last_bond",
            "family_mpo_descriptor_payload_seconds": "cpp_moving_environment_family_mpo_descriptor_payload_seconds",
            "family_mpo_descriptor_last_payload_seconds": "cpp_moving_environment_family_mpo_descriptor_last_payload_seconds",
            "owned_family_mpo_records": "cpp_moving_environment_owned_family_mpo_records",
            "owned_family_mpo_installs": "cpp_moving_environment_owned_family_mpo_installs",
            "owned_family_mpo_replacements": "cpp_moving_environment_owned_family_mpo_replacements",
            "owned_family_mpo_failures": "cpp_moving_environment_owned_family_mpo_failures",
            "owned_family_mpo_last_families": "cpp_moving_environment_owned_family_mpo_last_families",
            "family_mpo_descriptor_from_owned_installs": "cpp_moving_environment_family_mpo_descriptor_from_owned_installs",
            "spatial_qchem_family_descriptor_records": "cpp_moving_environment_spatial_qchem_family_descriptor_records",
            "spatial_qchem_family_descriptor_installs": "cpp_moving_environment_spatial_qchem_family_descriptor_installs",
            "spatial_qchem_family_descriptor_replacements": "cpp_moving_environment_spatial_qchem_family_descriptor_replacements",
            "spatial_qchem_family_descriptor_failures": "cpp_moving_environment_spatial_qchem_family_descriptor_failures",
            "spatial_qchem_family_descriptor_last_families": "cpp_moving_environment_spatial_qchem_family_descriptor_last_families",
            "spatial_qchem_family_descriptor_last_terms": "cpp_moving_environment_spatial_qchem_family_descriptor_last_terms",
            "spatial_qchem_family_descriptor_mpo_builds": "cpp_moving_environment_spatial_qchem_family_descriptor_mpo_builds",
            "spatial_qchem_family_descriptor_mpo_build_failures": "cpp_moving_environment_spatial_qchem_family_descriptor_mpo_build_failures",
            "spatial_qchem_family_descriptor_install_seconds": "cpp_moving_environment_spatial_qchem_family_descriptor_install_seconds",
            "spatial_qchem_family_descriptor_mpo_build_seconds": "cpp_moving_environment_spatial_qchem_family_descriptor_mpo_build_seconds",
            "sweep_environment_step_calls": "cpp_moving_environment_sweep_environment_step_calls",
            "sweep_environment_step_updates": "cpp_moving_environment_sweep_environment_step_updates",
            "sweep_environment_step_pops": "cpp_moving_environment_sweep_environment_step_pops",
            "sweep_environment_step_syncs": "cpp_moving_environment_sweep_environment_step_syncs",
            "sweep_environment_step_failures": "cpp_moving_environment_sweep_environment_step_failures",
            "sweep_environment_step_auto_calls": "cpp_moving_environment_sweep_environment_step_auto_calls",
            "bond_step_transaction_calls": "cpp_moving_environment_bond_step_transaction_calls",
            "bond_step_transaction_accepted": "cpp_moving_environment_bond_step_transaction_accepted",
            "bond_step_transaction_failures": "cpp_moving_environment_bond_step_transaction_failures",
            "bond_step_transaction_environment_updates": "cpp_moving_environment_bond_step_transaction_environment_updates",
            "direct_family_payload_records": "cpp_moving_environment_direct_family_payload_records",
            "direct_family_payload_installs": "cpp_moving_environment_direct_family_payload_installs",
            "direct_family_payload_replacements": "cpp_moving_environment_direct_family_payload_replacements",
            "direct_family_payload_hits": "cpp_moving_environment_direct_family_payload_hits",
            "direct_family_payload_misses": "cpp_moving_environment_direct_family_payload_misses",
            "direct_family_payload_clears": "cpp_moving_environment_direct_family_payload_clears",
            "direct_family_payload_cleared_entries": "cpp_moving_environment_direct_family_payload_cleared_entries",
            "direct_family_route_plan_records": "cpp_moving_environment_direct_family_route_plan_records",
            "direct_family_route_plan_record_hits": "cpp_moving_environment_direct_family_route_plan_record_hits",
            "direct_family_route_plan_record_misses": "cpp_moving_environment_direct_family_route_plan_record_misses",
            "direct_family_route_plan_installs": "cpp_moving_environment_direct_family_route_plan_installs",
            "direct_family_route_plan_payload_builds": "cpp_moving_environment_direct_family_route_plan_payload_builds",
            "direct_family_route_plan_last_entries": "cpp_moving_environment_direct_family_route_plan_last_entries",
            "direct_family_route_plan_payload_seconds": "cpp_moving_environment_direct_family_route_plan_payload_seconds",
            "direct_family_route_plan_last_payload_seconds": "cpp_moving_environment_direct_family_route_plan_last_payload_seconds",
            "direct_family_payload_builder_records": "cpp_moving_environment_direct_family_payload_builder_records",
            "direct_family_payload_builder_installs": "cpp_moving_environment_direct_family_payload_builder_installs",
            "direct_family_payload_builder_replacements": "cpp_moving_environment_direct_family_payload_builder_replacements",
            "direct_family_payload_builder_prepare_calls": "cpp_moving_environment_direct_family_payload_builder_prepare_calls",
            "direct_family_payload_builder_builds": "cpp_moving_environment_direct_family_payload_builder_builds",
            "direct_family_payload_builder_cache_hits": "cpp_moving_environment_direct_family_payload_builder_cache_hits",
            "direct_family_payload_builder_misses": "cpp_moving_environment_direct_family_payload_builder_misses",
            "direct_family_payload_builder_failures": "cpp_moving_environment_direct_family_payload_builder_failures",
            "direct_family_payload_builder_clears": "cpp_moving_environment_direct_family_payload_builder_clears",
            "direct_family_payload_builder_cleared_entries": "cpp_moving_environment_direct_family_payload_builder_cleared_entries",
            "direct_family_payload_builder_entries": "cpp_moving_environment_direct_family_payload_builder_entries",
            "direct_family_payload_builder_last_entries": "cpp_moving_environment_direct_family_payload_builder_last_entries",
            "direct_family_payload_builder_build_seconds": "cpp_moving_environment_direct_family_payload_builder_build_seconds",
            "direct_family_payload_builder_last_build_seconds": "cpp_moving_environment_direct_family_payload_builder_last_build_seconds",
            "direct_family_payload_assembler_calls": "cpp_moving_environment_direct_family_payload_assembler_calls",
            "direct_family_payload_assembler_builds": "cpp_moving_environment_direct_family_payload_assembler_builds",
            "direct_family_payload_assembler_families": "cpp_moving_environment_direct_family_payload_assembler_families",
            "direct_family_payload_assembler_pieces": "cpp_moving_environment_direct_family_payload_assembler_pieces",
            "direct_family_payload_assembler_merges": "cpp_moving_environment_direct_family_payload_assembler_merges",
            "direct_family_payload_assembler_empty_pieces": "cpp_moving_environment_direct_family_payload_assembler_empty_pieces",
            "direct_family_payload_assembler_failures": "cpp_moving_environment_direct_family_payload_assembler_failures",
            "direct_family_payload_assembler_seconds": "cpp_moving_environment_direct_family_payload_assembler_seconds",
            "direct_family_payload_assembler_last_seconds": "cpp_moving_environment_direct_family_payload_assembler_last_seconds",
            "direct_family_piece_builder_plan_calls": "cpp_moving_environment_direct_family_piece_builder_plan_calls",
            "direct_family_piece_builder_plan_builds": "cpp_moving_environment_direct_family_piece_builder_plan_builds",
            "direct_family_piece_builder_plan_families": "cpp_moving_environment_direct_family_piece_builder_plan_families",
            "direct_family_piece_builder_plan_pieces": "cpp_moving_environment_direct_family_piece_builder_plan_pieces",
            "direct_family_piece_builder_plan_entries": "cpp_moving_environment_direct_family_piece_builder_plan_entries",
            "direct_family_piece_builder_plan_empty_pieces": "cpp_moving_environment_direct_family_piece_builder_plan_empty_pieces",
            "direct_family_piece_builder_plan_failures": "cpp_moving_environment_direct_family_piece_builder_plan_failures",
            "direct_family_piece_builder_plan_seconds": "cpp_moving_environment_direct_family_piece_builder_plan_seconds",
            "direct_family_piece_builder_plan_last_seconds": "cpp_moving_environment_direct_family_piece_builder_plan_last_seconds",
            "direct_family_phased_piece_plan_records": "cpp_moving_environment_direct_family_phased_piece_plan_records",
            "direct_family_phased_piece_plan_installs": "cpp_moving_environment_direct_family_phased_piece_plan_installs",
            "direct_family_phased_piece_plan_replacements": "cpp_moving_environment_direct_family_phased_piece_plan_replacements",
            "direct_family_phased_piece_plan_prepare_calls": "cpp_moving_environment_direct_family_phased_piece_plan_prepare_calls",
            "direct_family_phased_piece_plan_cache_hits": "cpp_moving_environment_direct_family_phased_piece_plan_cache_hits",
            "direct_family_phased_piece_plan_misses": "cpp_moving_environment_direct_family_phased_piece_plan_misses",
            "direct_family_phased_piece_plan_failures": "cpp_moving_environment_direct_family_phased_piece_plan_failures",
            "direct_family_phased_family_plan_records": "cpp_moving_environment_direct_family_phased_family_plan_records",
            "direct_family_phased_family_plan_installs": "cpp_moving_environment_direct_family_phased_family_plan_installs",
            "direct_family_phased_family_plan_replacements": "cpp_moving_environment_direct_family_phased_family_plan_replacements",
            "direct_family_phased_family_plan_prepare_calls": "cpp_moving_environment_direct_family_phased_family_plan_prepare_calls",
            "direct_family_phased_family_plan_cache_hits": "cpp_moving_environment_direct_family_phased_family_plan_cache_hits",
            "direct_family_phased_family_plan_misses": "cpp_moving_environment_direct_family_phased_family_plan_misses",
            "direct_family_phased_family_plan_failures": "cpp_moving_environment_direct_family_phased_family_plan_failures",
            "direct_family_phased_family_plan_dispatch_calls": "cpp_moving_environment_direct_family_phased_family_plan_dispatch_calls",
            "direct_family_phased_family_plan_dispatch_families": "cpp_moving_environment_direct_family_phased_family_plan_dispatch_families",
            "direct_family_phased_family_plan_dispatch_pieces": "cpp_moving_environment_direct_family_phased_family_plan_dispatch_pieces",
            "direct_family_phased_family_plan_dispatch_entries": "cpp_moving_environment_direct_family_phased_family_plan_dispatch_entries",
            "direct_family_phased_family_plan_dispatch_empty_pieces": "cpp_moving_environment_direct_family_phased_family_plan_dispatch_empty_pieces",
            "direct_family_two_phase_dispatch_plan_records": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_records",
            "direct_family_two_phase_dispatch_plan_installs": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_installs",
            "direct_family_two_phase_dispatch_plan_replacements": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_replacements",
            "direct_family_two_phase_dispatch_plan_prepare_calls": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_prepare_calls",
            "direct_family_two_phase_dispatch_plan_cache_hits": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_cache_hits",
            "direct_family_two_phase_dispatch_plan_misses": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_misses",
            "direct_family_two_phase_dispatch_plan_failures": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_failures",
            "direct_family_two_phase_dispatch_plan_dispatch_calls": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_calls",
            "direct_family_two_phase_dispatch_plan_dispatch_families": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_families",
            "direct_family_two_phase_dispatch_plan_dispatch_pieces": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_pieces",
            "direct_family_two_phase_dispatch_plan_dispatch_entries": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_entries",
            "direct_family_two_phase_dispatch_plan_dispatch_empty_pieces": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_empty_pieces",
            "direct_family_two_phase_dispatch_plan_factory_calls": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_factory_calls",
            "direct_family_two_phase_dispatch_plan_static_plan_installs": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_installs",
            "direct_family_two_phase_dispatch_plan_static_plan_uses": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_uses",
            "direct_family_two_phase_dispatch_plan_literal_families": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_families",
            "direct_family_two_phase_dispatch_plan_literal_pieces": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_pieces",
            "direct_family_two_phase_dispatch_plan_literal_entries": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_entries",
            "direct_family_two_phase_dispatch_plan_literal_empty_pieces": "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_empty_pieces",
            "planned_direct_family_entry_build_calls": "cpp_moving_environment_planned_direct_family_entry_build_calls",
            "planned_direct_family_entry_build_successes": "cpp_moving_environment_planned_direct_family_entry_build_successes",
            "planned_direct_family_entry_build_failures": "cpp_moving_environment_planned_direct_family_entry_build_failures",
            "planned_direct_family_entry_build_entries": "cpp_moving_environment_planned_direct_family_entry_build_entries",
            "planned_direct_family_entry_build_table_backed": "cpp_moving_environment_planned_direct_family_entry_build_table_backed",
            "planned_direct_family_entry_build_seconds": "cpp_moving_environment_planned_direct_family_entry_build_seconds",
            "planned_direct_family_entry_build_last_seconds": "cpp_moving_environment_planned_direct_family_entry_build_last_seconds",
            "direct_route_plan_build_calls": "cpp_moving_environment_direct_route_plan_build_calls",
            "direct_route_plan_build_successes": "cpp_moving_environment_direct_route_plan_build_successes",
            "direct_route_plan_build_failures": "cpp_moving_environment_direct_route_plan_build_failures",
            "direct_route_plan_build_records": "cpp_moving_environment_direct_route_plan_build_records",
            "direct_route_plan_build_pairs": "cpp_moving_environment_direct_route_plan_build_pairs",
            "direct_route_plan_build_left_keys": "cpp_moving_environment_direct_route_plan_build_left_keys",
            "direct_route_plan_build_right_keys": "cpp_moving_environment_direct_route_plan_build_right_keys",
            "direct_route_plan_build_seconds": "cpp_moving_environment_direct_route_plan_build_seconds",
            "direct_route_plan_build_last_seconds": "cpp_moving_environment_direct_route_plan_build_last_seconds",
            "same_side_route_identity_select_calls": "cpp_moving_environment_same_side_route_identity_select_calls",
            "same_side_route_identity_select_failures": "cpp_moving_environment_same_side_route_identity_select_failures",
            "same_side_route_identity_select_rows": "cpp_moving_environment_same_side_route_identity_select_rows",
            "same_side_route_identity_select_terms": "cpp_moving_environment_same_side_route_identity_select_terms",
            "same_side_route_identity_select_scanned": "cpp_moving_environment_same_side_route_identity_select_scanned",
            "same_side_route_identity_select_skipped_consumed": "cpp_moving_environment_same_side_route_identity_select_skipped_consumed",
            "same_side_route_identity_select_skipped_zero": "cpp_moving_environment_same_side_route_identity_select_skipped_zero",
            "same_side_route_identity_select_seconds": "cpp_moving_environment_same_side_route_identity_select_seconds",
            "same_side_route_identity_select_last_seconds": "cpp_moving_environment_same_side_route_identity_select_last_seconds",
            "same_side_route_identity_info_calls": "cpp_moving_environment_same_side_route_identity_info_calls",
            "same_side_route_identity_info_successes": "cpp_moving_environment_same_side_route_identity_info_successes",
            "same_side_route_identity_info_unsupported": "cpp_moving_environment_same_side_route_identity_info_unsupported",
            "same_side_route_identity_info_failures": "cpp_moving_environment_same_side_route_identity_info_failures",
            "same_side_route_identity_info_records": "cpp_moving_environment_same_side_route_identity_info_records",
            "same_side_route_identity_info_terms": "cpp_moving_environment_same_side_route_identity_info_terms",
            "same_side_route_identity_info_rows": "cpp_moving_environment_same_side_route_identity_info_rows",
            "same_side_route_identity_info_row_map_builds": "cpp_moving_environment_same_side_route_identity_info_row_map_builds",
            "same_side_route_identity_info_row_map_hits": "cpp_moving_environment_same_side_route_identity_info_row_map_hits",
            "same_side_route_identity_info_seconds": "cpp_moving_environment_same_side_route_identity_info_seconds",
            "same_side_route_identity_info_last_seconds": "cpp_moving_environment_same_side_route_identity_info_last_seconds",
            "same_side_route_boundary_batch_calls": "cpp_moving_environment_same_side_route_boundary_batch_calls",
            "same_side_route_boundary_batch_successes": "cpp_moving_environment_same_side_route_boundary_batch_successes",
            "same_side_route_boundary_batch_failures": "cpp_moving_environment_same_side_route_boundary_batch_failures",
            "same_side_route_boundary_batch_keys": "cpp_moving_environment_same_side_route_boundary_batch_keys",
            "same_side_route_boundary_batch_hits": "cpp_moving_environment_same_side_route_boundary_batch_hits",
            "same_side_route_boundary_batch_misses": "cpp_moving_environment_same_side_route_boundary_batch_misses",
            "same_side_route_boundary_batch_complete": "cpp_moving_environment_same_side_route_boundary_batch_complete",
            "same_side_route_boundary_batch_seconds": "cpp_moving_environment_same_side_route_boundary_batch_seconds",
            "same_side_route_boundary_batch_last_seconds": "cpp_moving_environment_same_side_route_boundary_batch_last_seconds",
            "same_side_route_boundary_parent_plan_calls": "cpp_moving_environment_same_side_route_boundary_parent_plan_calls",
            "same_side_route_boundary_parent_plan_successes": "cpp_moving_environment_same_side_route_boundary_parent_plan_successes",
            "same_side_route_boundary_parent_plan_failures": "cpp_moving_environment_same_side_route_boundary_parent_plan_failures",
            "same_side_route_boundary_parent_plan_rows": "cpp_moving_environment_same_side_route_boundary_parent_plan_rows",
            "same_side_route_boundary_parent_plan_unique": "cpp_moving_environment_same_side_route_boundary_parent_plan_unique",
            "same_side_route_boundary_parent_plan_route_layout": "cpp_moving_environment_same_side_route_boundary_parent_plan_route_layout",
            "same_side_route_boundary_parent_plan_fallback": "cpp_moving_environment_same_side_route_boundary_parent_plan_fallback",
            "same_side_route_boundary_parent_plan_seconds": "cpp_moving_environment_same_side_route_boundary_parent_plan_seconds",
            "same_side_route_boundary_parent_plan_last_seconds": "cpp_moving_environment_same_side_route_boundary_parent_plan_last_seconds",
            "same_side_route_boundary_parent_value_calls": "cpp_moving_environment_same_side_route_boundary_parent_value_calls",
            "same_side_route_boundary_parent_value_successes": "cpp_moving_environment_same_side_route_boundary_parent_value_successes",
            "same_side_route_boundary_parent_value_failures": "cpp_moving_environment_same_side_route_boundary_parent_value_failures",
            "same_side_route_boundary_parent_value_rows": "cpp_moving_environment_same_side_route_boundary_parent_value_rows",
            "same_side_route_boundary_parent_value_available": "cpp_moving_environment_same_side_route_boundary_parent_value_available",
            "same_side_route_boundary_parent_value_missing": "cpp_moving_environment_same_side_route_boundary_parent_value_missing",
            "same_side_route_boundary_parent_value_hits": "cpp_moving_environment_same_side_route_boundary_parent_value_hits",
            "same_side_route_boundary_parent_value_misses": "cpp_moving_environment_same_side_route_boundary_parent_value_misses",
            "same_side_route_boundary_parent_value_seconds": "cpp_moving_environment_same_side_route_boundary_parent_value_seconds",
            "same_side_route_boundary_parent_value_last_seconds": "cpp_moving_environment_same_side_route_boundary_parent_value_last_seconds",
            "same_side_route_boundary_parent_advance_calls": "cpp_moving_environment_same_side_route_boundary_parent_advance_calls",
            "same_side_route_boundary_parent_advance_successes": "cpp_moving_environment_same_side_route_boundary_parent_advance_successes",
            "same_side_route_boundary_parent_advance_failures": "cpp_moving_environment_same_side_route_boundary_parent_advance_failures",
            "same_side_route_boundary_parent_advance_rows": "cpp_moving_environment_same_side_route_boundary_parent_advance_rows",
            "same_side_route_boundary_parent_advance_advanced": "cpp_moving_environment_same_side_route_boundary_parent_advance_advanced",
            "same_side_route_boundary_parent_advance_remaining": "cpp_moving_environment_same_side_route_boundary_parent_advance_remaining",
            "same_side_route_boundary_parent_advance_cache_hits": "cpp_moving_environment_same_side_route_boundary_parent_advance_cache_hits",
            "same_side_route_boundary_parent_advance_cache_builds": "cpp_moving_environment_same_side_route_boundary_parent_advance_cache_builds",
            "same_side_route_boundary_parent_advance_none": "cpp_moving_environment_same_side_route_boundary_parent_advance_none",
            "same_side_route_boundary_parent_advance_seconds": "cpp_moving_environment_same_side_route_boundary_parent_advance_seconds",
            "same_side_route_boundary_parent_advance_last_seconds": "cpp_moving_environment_same_side_route_boundary_parent_advance_last_seconds",
            "same_side_route_missing_parent_build_plan_calls": "cpp_moving_environment_same_side_route_missing_parent_build_plan_calls",
            "same_side_route_missing_parent_build_plan_successes": "cpp_moving_environment_same_side_route_missing_parent_build_plan_successes",
            "same_side_route_missing_parent_build_plan_failures": "cpp_moving_environment_same_side_route_missing_parent_build_plan_failures",
            "same_side_route_missing_parent_build_plan_rows": "cpp_moving_environment_same_side_route_missing_parent_build_plan_rows",
            "same_side_route_missing_parent_build_plan_unique": "cpp_moving_environment_same_side_route_missing_parent_build_plan_unique",
            "same_side_route_missing_parent_build_plan_seconds": "cpp_moving_environment_same_side_route_missing_parent_build_plan_seconds",
            "same_side_route_missing_parent_build_plan_last_seconds": "cpp_moving_environment_same_side_route_missing_parent_build_plan_last_seconds",
            "same_side_route_built_parent_advance_plan_calls": "cpp_moving_environment_same_side_route_built_parent_advance_plan_calls",
            "same_side_route_built_parent_advance_plan_successes": "cpp_moving_environment_same_side_route_built_parent_advance_plan_successes",
            "same_side_route_built_parent_advance_plan_failures": "cpp_moving_environment_same_side_route_built_parent_advance_plan_failures",
            "same_side_route_built_parent_advance_plan_rows": "cpp_moving_environment_same_side_route_built_parent_advance_plan_rows",
            "same_side_route_built_parent_advance_plan_available": "cpp_moving_environment_same_side_route_built_parent_advance_plan_available",
            "same_side_route_built_parent_advance_plan_missing": "cpp_moving_environment_same_side_route_built_parent_advance_plan_missing",
            "same_side_route_built_parent_advance_plan_puts": "cpp_moving_environment_same_side_route_built_parent_advance_plan_puts",
            "same_side_route_built_parent_advance_plan_seconds": "cpp_moving_environment_same_side_route_built_parent_advance_plan_seconds",
            "same_side_route_built_parent_advance_plan_last_seconds": "cpp_moving_environment_same_side_route_built_parent_advance_plan_last_seconds",
            "same_side_route_identity_entry_build_calls": "cpp_moving_environment_same_side_route_identity_entry_build_calls",
            "same_side_route_identity_entry_build_failures": "cpp_moving_environment_same_side_route_identity_entry_build_failures",
            "same_side_route_identity_entry_build_rows": "cpp_moving_environment_same_side_route_identity_entry_build_rows",
            "same_side_route_identity_entry_build_terms": "cpp_moving_environment_same_side_route_identity_entry_build_terms",
            "same_side_route_identity_entry_build_seconds": "cpp_moving_environment_same_side_route_identity_entry_build_seconds",
            "same_side_route_identity_entry_build_last_seconds": "cpp_moving_environment_same_side_route_identity_entry_build_last_seconds",
            "owner_bond_step_runner_calls": "cpp_moving_environment_owner_bond_step_runner_calls",
            "owner_bond_step_runner_accepted": "cpp_moving_environment_owner_bond_step_runner_accepted",
            "owner_bond_step_runner_failures": "cpp_moving_environment_owner_bond_step_runner_failures",
            "owner_bond_step_runner_payload_prepares": "cpp_moving_environment_owner_bond_step_runner_payload_prepares",
            "owner_bond_step_runner_environment_moves": "cpp_moving_environment_owner_bond_step_runner_environment_moves",
            "owner_bond_step_runner_environment_fallbacks": "cpp_moving_environment_owner_bond_step_runner_environment_fallbacks",
            "owner_bond_step_runner_assign_calls": "cpp_moving_environment_owner_bond_step_runner_assign_calls",
            "owner_bond_step_runner_assign_skips": "cpp_moving_environment_owner_bond_step_runner_assign_skips",
            "owner_bond_step_runner_seconds": "cpp_moving_environment_owner_bond_step_runner_seconds",
            "owner_bond_step_runner_last_seconds": "cpp_moving_environment_owner_bond_step_runner_last_seconds",
            "owner_bond_step_runner_payload_seconds": "cpp_moving_environment_owner_bond_step_runner_payload_seconds",
            "owner_bond_step_runner_payload_last_seconds": "cpp_moving_environment_owner_bond_step_runner_payload_last_seconds",
            "owner_bond_step_record_records": "cpp_moving_environment_owner_bond_step_record_records",
            "owner_bond_step_record_installs": "cpp_moving_environment_owner_bond_step_record_installs",
            "owner_bond_step_record_replacements": "cpp_moving_environment_owner_bond_step_record_replacements",
            "owner_bond_step_record_hits": "cpp_moving_environment_owner_bond_step_record_hits",
            "owner_bond_step_record_misses": "cpp_moving_environment_owner_bond_step_record_misses",
            "owner_bond_step_record_clears": "cpp_moving_environment_owner_bond_step_record_clears",
            "owner_bond_step_record_cleared_entries": "cpp_moving_environment_owner_bond_step_record_cleared_entries",
            "owner_typed_bond_step_record_records": "cpp_moving_environment_owner_typed_bond_step_record_records",
            "owner_typed_bond_step_record_installs": "cpp_moving_environment_owner_typed_bond_step_record_installs",
            "owner_typed_bond_step_record_replacements": "cpp_moving_environment_owner_typed_bond_step_record_replacements",
            "owner_typed_bond_step_record_hits": "cpp_moving_environment_owner_typed_bond_step_record_hits",
            "owner_typed_bond_step_record_misses": "cpp_moving_environment_owner_typed_bond_step_record_misses",
            "owner_typed_bond_step_record_clears": "cpp_moving_environment_owner_typed_bond_step_record_clears",
            "owner_typed_bond_step_record_cleared_entries": "cpp_moving_environment_owner_typed_bond_step_record_cleared_entries",
            "owner_typed_bond_step_environment_record_prepares": "cpp_moving_environment_owner_typed_bond_step_environment_record_prepares",
            "owner_typed_bond_step_environment_record_consumes": "cpp_moving_environment_owner_typed_bond_step_environment_record_consumes",
            "owner_typed_bond_step_python_prepare_calls": "cpp_moving_environment_owner_typed_bond_step_python_prepare_calls",
            "owner_typed_bond_step_python_move_calls": "cpp_moving_environment_owner_typed_bond_step_python_move_calls",
            "owner_typed_bond_step_direct_plan_provider_record_installs": "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_record_installs",
            "owner_typed_bond_step_direct_plan_provider_calls": "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_calls",
            "owner_typed_bond_step_direct_plan_provider_accepts": "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_accepts",
            "owner_typed_bond_step_direct_plan_provider_empty": "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_empty",
            "owner_typed_bond_step_direct_plan_provider_failures": "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_failures",
            "owner_typed_bond_step_direct_key_updates": "cpp_moving_environment_owner_typed_bond_step_direct_key_updates",
            "owner_typed_bond_step_direct_key_update_misses": "cpp_moving_environment_owner_typed_bond_step_direct_key_update_misses",
            "owner_typed_bond_step_direct_key_update_failures": "cpp_moving_environment_owner_typed_bond_step_direct_key_update_failures",
            "owner_typed_bond_step_direct_key_provider_refresh_calls": "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_calls",
            "owner_typed_bond_step_direct_key_provider_refresh_accepts": "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_accepts",
            "owner_typed_bond_step_direct_key_provider_refresh_empty": "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_empty",
            "owner_typed_bond_step_direct_key_provider_refresh_failures": "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_failures",
            "owner_typed_bond_step_direct_key_successor_refresh_calls": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_calls",
            "owner_typed_bond_step_direct_key_successor_refresh_accepts": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_accepts",
            "owner_typed_bond_step_direct_key_successor_refresh_empty": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_empty",
            "owner_typed_bond_step_direct_key_successor_refresh_failures": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_failures",
            "owner_typed_bond_step_direct_key_successor_chain_calls": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_calls",
            "owner_typed_bond_step_direct_key_successor_chain_accepts": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_accepts",
            "owner_typed_bond_step_direct_key_successor_chain_links": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_links",
            "owner_typed_bond_step_direct_key_successor_chain_failures": "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_failures",
            "direct_family_revision_state_updates": "cpp_moving_environment_direct_family_revision_state_updates",
            "direct_family_revision_state_failures": "cpp_moving_environment_direct_family_revision_state_failures",
            "direct_family_revision_cache_key_builds": "cpp_moving_environment_direct_family_revision_cache_key_builds",
            "direct_family_revision_cache_key_failures": "cpp_moving_environment_direct_family_revision_cache_key_failures",
            "direct_family_cpp_key_bundle_builds": "cpp_moving_environment_direct_family_cpp_key_bundle_builds",
            "direct_family_cpp_key_bundle_failures": "cpp_moving_environment_direct_family_cpp_key_bundle_failures",
            "owner_typed_half_sweep_plan_records": "cpp_moving_environment_owner_typed_half_sweep_plan_records",
            "owner_typed_half_sweep_plan_installs": "cpp_moving_environment_owner_typed_half_sweep_plan_installs",
            "owner_typed_half_sweep_plan_replacements": "cpp_moving_environment_owner_typed_half_sweep_plan_replacements",
            "owner_typed_half_sweep_plan_hits": "cpp_moving_environment_owner_typed_half_sweep_plan_hits",
            "owner_typed_half_sweep_plan_misses": "cpp_moving_environment_owner_typed_half_sweep_plan_misses",
            "owner_typed_half_sweep_plan_runs": "cpp_moving_environment_owner_typed_half_sweep_plan_runs",
            "owner_typed_half_sweep_plan_bonds": "cpp_moving_environment_owner_typed_half_sweep_plan_bonds",
            "owner_typed_half_sweep_template_plan_installs": "cpp_moving_environment_owner_typed_half_sweep_template_plan_installs",
            "owner_typed_half_sweep_template_plan_bonds": "cpp_moving_environment_owner_typed_half_sweep_template_plan_bonds",
            "owner_typed_half_sweep_template_local_records": "cpp_moving_environment_owner_typed_half_sweep_template_local_records",
            "owner_typed_half_sweep_template_step_records": "cpp_moving_environment_owner_typed_half_sweep_template_step_records",
            "owner_sweep_schedule_plan_records": "cpp_moving_environment_owner_sweep_schedule_plan_records",
            "owner_sweep_schedule_plan_installs": "cpp_moving_environment_owner_sweep_schedule_plan_installs",
            "owner_sweep_schedule_plan_replacements": "cpp_moving_environment_owner_sweep_schedule_plan_replacements",
            "owner_sweep_schedule_plan_alternating_installs": "cpp_moving_environment_owner_sweep_schedule_plan_alternating_installs",
            "owner_sweep_schedule_plan_alternating_expanded_halves": "cpp_moving_environment_owner_sweep_schedule_plan_alternating_expanded_halves",
            "owner_sweep_schedule_plan_noise_sets": "cpp_moving_environment_owner_sweep_schedule_plan_noise_sets",
            "owner_sweep_schedule_plan_noise_set_failures": "cpp_moving_environment_owner_sweep_schedule_plan_noise_set_failures",
            "owner_sweep_schedule_plan_hits": "cpp_moving_environment_owner_sweep_schedule_plan_hits",
            "owner_sweep_schedule_plan_misses": "cpp_moving_environment_owner_sweep_schedule_plan_misses",
            "owner_sweep_schedule_plan_runs": "cpp_moving_environment_owner_sweep_schedule_plan_runs",
            "owner_sweep_schedule_plan_halves": "cpp_moving_environment_owner_sweep_schedule_plan_halves",
            "owner_sweep_schedule_plan_converged": "cpp_moving_environment_owner_sweep_schedule_plan_converged",
            "owner_sweep_schedule_plan_history_rows": "cpp_moving_environment_owner_sweep_schedule_plan_history_rows",
            "owner_sweep_schedule_plan_final_recenter_configures": "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_configures",
            "owner_sweep_schedule_plan_final_recenter_runs": "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_runs",
            "owner_sweep_schedule_plan_final_recenter_skips": "cpp_moving_environment_owner_sweep_schedule_plan_final_recenter_skips",
            "owner_sweep_schedule_plan_seconds": "cpp_moving_environment_owner_sweep_schedule_plan_seconds",
            "owner_sweep_schedule_plan_last_seconds": "cpp_moving_environment_owner_sweep_schedule_plan_last_seconds",
            "owner_local_optimize_runner_calls": "cpp_moving_environment_owner_local_optimize_runner_calls",
            "owner_local_optimize_runner_accepted": "cpp_moving_environment_owner_local_optimize_runner_accepted",
            "owner_local_optimize_runner_rejections": "cpp_moving_environment_owner_local_optimize_runner_rejections",
            "owner_local_optimize_runner_failures": "cpp_moving_environment_owner_local_optimize_runner_failures",
            "owner_local_optimize_runner_seconds": "cpp_moving_environment_owner_local_optimize_runner_seconds",
            "owner_local_optimize_runner_last_seconds": "cpp_moving_environment_owner_local_optimize_runner_last_seconds",
            "owner_local_optimize_record_records": "cpp_moving_environment_owner_local_optimize_record_records",
            "owner_local_optimize_record_installs": "cpp_moving_environment_owner_local_optimize_record_installs",
            "owner_local_optimize_record_replacements": "cpp_moving_environment_owner_local_optimize_record_replacements",
            "owner_local_optimize_record_hits": "cpp_moving_environment_owner_local_optimize_record_hits",
            "owner_local_optimize_record_misses": "cpp_moving_environment_owner_local_optimize_record_misses",
            "owner_local_optimize_record_clears": "cpp_moving_environment_owner_local_optimize_record_clears",
            "owner_local_optimize_record_cleared_entries": "cpp_moving_environment_owner_local_optimize_record_cleared_entries",
            "owner_local_optimize_record_noise_sets": "cpp_moving_environment_owner_local_optimize_record_noise_sets",
            "owner_local_optimize_native_merge_calls": "cpp_moving_environment_owner_local_optimize_native_merge_calls",
            "owner_local_optimize_native_merge_accepted": "cpp_moving_environment_owner_local_optimize_native_merge_accepted",
            "owner_local_optimize_native_merge_failures": "cpp_moving_environment_owner_local_optimize_native_merge_failures",
            "owner_local_optimize_native_noise_injections": "cpp_moving_environment_owner_local_optimize_native_noise_injections",
            "owner_local_optimize_native_noise_blocks": "cpp_moving_environment_owner_local_optimize_native_noise_blocks",
            "owner_local_optimize_bridge_merge_calls": "cpp_moving_environment_owner_local_optimize_bridge_merge_calls",
            "owner_local_optimize_boundary_stack_reads": "cpp_moving_environment_owner_local_optimize_boundary_stack_reads",
            "owner_local_optimize_boundary_bridge_calls": "cpp_moving_environment_owner_local_optimize_boundary_bridge_calls",
            "owner_local_problem_bind_owner_calls": "cpp_moving_environment_owner_local_problem_bind_owner_calls",
            "owner_local_problem_bind_set_bond_fallbacks": "cpp_moving_environment_owner_local_problem_bind_set_bond_fallbacks",
            "owner_site_chain_records": "cpp_moving_environment_owner_site_chain_records",
            "owner_site_chain_installs": "cpp_moving_environment_owner_site_chain_installs",
            "owner_site_chain_replacements": "cpp_moving_environment_owner_site_chain_replacements",
            "owner_site_chain_gets": "cpp_moving_environment_owner_site_chain_gets",
            "owner_site_chain_sets": "cpp_moving_environment_owner_site_chain_sets",
            "owner_site_chain_syncs": "cpp_moving_environment_owner_site_chain_syncs",
            "owner_site_chain_sync_sites": "cpp_moving_environment_owner_site_chain_sync_sites",
            "owner_site_chain_failures": "cpp_moving_environment_owner_site_chain_failures",
            "owner_local_grouped_solve_update_calls": "cpp_moving_environment_owner_local_grouped_solve_update_calls",
            "owner_local_grouped_solve_update_accepted": "cpp_moving_environment_owner_local_grouped_solve_update_accepted",
            "owner_local_grouped_solve_update_rejections": "cpp_moving_environment_owner_local_grouped_solve_update_rejections",
            "owner_local_grouped_solve_update_failures": "cpp_moving_environment_owner_local_grouped_solve_update_failures",
            "owner_local_grouped_solve_update_seconds": "cpp_moving_environment_owner_local_grouped_solve_update_seconds",
            "owner_local_grouped_solve_update_last_seconds": "cpp_moving_environment_owner_local_grouped_solve_update_last_seconds",
            "owner_half_sweep_runner_calls": "cpp_moving_environment_owner_half_sweep_runner_calls",
            "owner_half_sweep_runner_accepted": "cpp_moving_environment_owner_half_sweep_runner_accepted",
            "owner_half_sweep_runner_failures": "cpp_moving_environment_owner_half_sweep_runner_failures",
            "owner_half_sweep_runner_bonds": "cpp_moving_environment_owner_half_sweep_runner_bonds",
            "owner_half_sweep_runner_seconds": "cpp_moving_environment_owner_half_sweep_runner_seconds",
            "owner_half_sweep_runner_last_seconds": "cpp_moving_environment_owner_half_sweep_runner_last_seconds",
        }
        for src, dst in mapping.items():
            if src in stats:
                if src.endswith("_seconds"):
                    self.moving_profile_stats[dst] = float(stats[src])
                else:
                    self.moving_profile_stats[dst] = int(stats[src])
        for src, value in stats.items():
            if not str(src).startswith("contextual_"):
                continue
            dst = f"cpp_moving_environment_{src}"
            if value is None:
                self.moving_profile_stats[dst] = None
            elif isinstance(value, bool):
                self.moving_profile_stats[dst] = bool(value)
            elif isinstance(value, str):
                self.moving_profile_stats[dst] = value
            elif src.endswith("_seconds"):
                self.moving_profile_stats[dst] = float(value)
            else:
                try:
                    self.moving_profile_stats[dst] = int(value)
                except (TypeError, ValueError):
                    try:
                        self.moving_profile_stats[dst] = float(value)
                    except (TypeError, ValueError):
                        self.moving_profile_stats[dst] = value
        if "environment_plan_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_environment_plan_last_error"
            ] = stats["environment_plan_last_error"]
        if "environment_stack_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_environment_stack_last_error"
            ] = stats["environment_stack_last_error"]
        if "sweep_environment_step_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_sweep_environment_step_last_error"
            ] = stats["sweep_environment_step_last_error"]
        if "bond_step_transaction_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_bond_step_transaction_last_error"
            ] = stats["bond_step_transaction_last_error"]
        if "direct_family_payload_builder_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_payload_builder_last_error"
            ] = stats["direct_family_payload_builder_last_error"]
        if "direct_family_payload_assembler_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_payload_assembler_last_error"
            ] = stats["direct_family_payload_assembler_last_error"]
        if "direct_family_piece_builder_plan_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_piece_builder_plan_last_error"
            ] = stats["direct_family_piece_builder_plan_last_error"]
        if "direct_family_phased_piece_plan_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_phased_piece_plan_last_error"
            ] = stats["direct_family_phased_piece_plan_last_error"]
        if "direct_family_phased_family_plan_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_phased_family_plan_last_error"
            ] = stats["direct_family_phased_family_plan_last_error"]
        if "direct_family_two_phase_dispatch_plan_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_direct_family_two_phase_dispatch_plan_last_error"
            ] = stats["direct_family_two_phase_dispatch_plan_last_error"]
        if "same_side_route_identity_select_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_same_side_route_identity_select_last_error"
            ] = stats["same_side_route_identity_select_last_error"]
        if "same_side_route_identity_info_last_reason" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_same_side_route_identity_info_last_reason"
            ] = stats["same_side_route_identity_info_last_reason"]
        if "same_side_route_identity_info_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_same_side_route_identity_info_last_error"
            ] = stats["same_side_route_identity_info_last_error"]
        if "same_side_route_boundary_batch_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_same_side_route_boundary_batch_last_error"
            ] = stats["same_side_route_boundary_batch_last_error"]
        if "same_side_route_boundary_parent_plan_last_error" in stats:
            self.moving_profile_stats[
                (
                    "cpp_moving_environment_same_side_route_"
                    "boundary_parent_plan_last_error"
                )
            ] = stats["same_side_route_boundary_parent_plan_last_error"]
        if "same_side_route_boundary_parent_value_last_error" in stats:
            self.moving_profile_stats[
                (
                    "cpp_moving_environment_same_side_route_"
                    "boundary_parent_value_last_error"
                )
            ] = stats["same_side_route_boundary_parent_value_last_error"]
        if "same_side_route_boundary_parent_advance_last_error" in stats:
            self.moving_profile_stats[
                (
                    "cpp_moving_environment_same_side_route_"
                    "boundary_parent_advance_last_error"
                )
            ] = stats["same_side_route_boundary_parent_advance_last_error"]
        if "same_side_route_missing_parent_build_plan_last_error" in stats:
            self.moving_profile_stats[
                (
                    "cpp_moving_environment_same_side_route_"
                    "missing_parent_build_plan_last_error"
                )
            ] = stats["same_side_route_missing_parent_build_plan_last_error"]
        if "same_side_route_built_parent_advance_plan_last_error" in stats:
            self.moving_profile_stats[
                (
                    "cpp_moving_environment_same_side_route_"
                    "built_parent_advance_plan_last_error"
                )
            ] = stats["same_side_route_built_parent_advance_plan_last_error"]
        if "same_side_route_identity_entry_build_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_same_side_route_identity_entry_build_last_error"
            ] = stats["same_side_route_identity_entry_build_last_error"]
        if "owner_bond_step_runner_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_bond_step_runner_last_error"
            ] = stats["owner_bond_step_runner_last_error"]
        if "owner_local_optimize_runner_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_local_optimize_runner_last_error"
            ] = stats["owner_local_optimize_runner_last_error"]
        if "owner_local_optimize_runner_last_reason" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_local_optimize_runner_last_reason"
            ] = stats["owner_local_optimize_runner_last_reason"]
        if "owner_bond_step_record_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_bond_step_record_last_error"
            ] = stats["owner_bond_step_record_last_error"]
        if "owner_typed_bond_step_record_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_typed_bond_step_record_last_error"
            ] = stats["owner_typed_bond_step_record_last_error"]
        if "owner_typed_half_sweep_plan_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_typed_half_sweep_plan_last_error"
            ] = stats["owner_typed_half_sweep_plan_last_error"]
        if "owner_sweep_schedule_plan_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_sweep_schedule_plan_last_error"
            ] = stats["owner_sweep_schedule_plan_last_error"]
        if "owner_local_grouped_solve_update_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_local_grouped_solve_update_last_error"
            ] = stats["owner_local_grouped_solve_update_last_error"]
        if "owner_local_grouped_solve_update_last_reason" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_local_grouped_solve_update_last_reason"
            ] = stats["owner_local_grouped_solve_update_last_reason"]
        if "owner_half_sweep_runner_last_direction" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_half_sweep_runner_last_direction"
            ] = stats["owner_half_sweep_runner_last_direction"]
        if "owner_half_sweep_runner_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_owner_half_sweep_runner_last_error"
            ] = stats["owner_half_sweep_runner_last_error"]
        if "site_split_flat_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_site_split_flat_last_error"
            ] = stats["site_split_flat_last_error"]
        if "site_update_flat_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_site_update_flat_last_error"
            ] = stats["site_update_flat_last_error"]
        if "solve_update_flat_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_solve_update_flat_last_error"
            ] = stats["solve_update_flat_last_error"]
        if "sweep_cursor_last_direction" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_sweep_cursor_last_direction"
            ] = stats["sweep_cursor_last_direction"]
        if "sweep_cursor_last_error" in stats:
            self.moving_profile_stats[
                "cpp_moving_environment_sweep_cursor_last_error"
            ] = stats["sweep_cursor_last_error"]
        self.moving_profile_stats["cpp_moving_environment_enabled"] = True

    def _install_cpp_moving_environment_compact_plan(self, direct, operator, layout):
        env = self._cpp_moving_environment
        if env is None or direct is None:
            return False
        routes = getattr(direct, "_diagonal_routes", None)
        if routes is None:
            return False
        key = self._cpp_moving_environment_compact_key(operator)
        layout_blocks = int(len(tuple(layout)))
        if (
            getattr(direct, "cpp_moving_environment", None) is env
            and getattr(direct, "cpp_moving_environment_key", None) == key
            and int(getattr(direct, "_cpp_moving_environment_layout_blocks", -1))
            == layout_blocks
            and getattr(direct, "_cpp_moving_environment_plan", None)
            is getattr(direct, "cpp_plan", None)
            and getattr(direct, "_cpp_moving_environment_routes", None) is routes
        ):
            self.moving_profile_stats[
                "cpp_moving_environment_compact_plan_install_skips"
            ] = int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_install_skips",
                    0,
                )
            ) + 1
            return True
        try:
            env.install_compact_plan(
                key,
                direct.cpp_plan,
                routes,
                int(direct.dim),
                layout_blocks,
            )
            direct.bind_cpp_moving_environment(env, key)
            direct._cpp_moving_environment_layout_blocks = layout_blocks
            direct._cpp_moving_environment_plan = direct.cpp_plan
            direct._cpp_moving_environment_routes = routes
            return True
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_compact_plan_failures"
            ] = int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "cpp_moving_environment_compact_plan_last_error"
            ] = str(exc)
            return False

    def _install_cpp_moving_environment_grouped_table(self, table, operator, layout):
        env = self._cpp_moving_environment
        if env is None or table is None:
            return False
        cpp_table = getattr(table, "cpp_table", None)
        if cpp_table is None:
            return False
        layout_blocks = int(len(tuple(layout)))
        try:
            auto_install = getattr(env, "install_grouped_table_auto", None)
            if auto_install is not None:
                key = auto_install(
                    cpp_table,
                    int(table.dim),
                    layout_blocks,
                    None if operator.bond is None else int(operator.bond),
                )
            else:
                key = self._cpp_moving_environment_grouped_key(operator)
                env.install_grouped_table(
                    key,
                    cpp_table,
                    int(table.dim),
                    layout_blocks,
                )
            table.bind_cpp_moving_environment(env, key)
            table._cpp_moving_environment_layout_blocks = layout_blocks
            table._cpp_moving_environment_table = cpp_table
            return True
        except Exception as exc:
            self.moving_profile_stats[
                "cpp_moving_environment_grouped_table_failures"
            ] = int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats[
                "cpp_moving_environment_grouped_table_last_error"
            ] = str(exc)
            return False

    @staticmethod
    def _compact_diagonal_routes(plan, offsets, layout_shapes):
        t2_by_r = defaultdict(list)
        for group_index, group in enumerate(plan["batched_t2_entries"]):
            for entry_pos, entry in enumerate(group):
                r_i, _w1_i, _t2_i = (int(value) for value in entry)
                t2_by_r[r_i].append((int(group_index), int(entry_pos), tuple(int(v) for v in entry)))

        t3_by_t2 = defaultdict(list)
        for group_index, group in enumerate(plan["batched_t3_entries"]):
            for entry_pos, entry in enumerate(group):
                t2_i, _w2_i, _t3_i = (int(value) for value in entry)
                t3_by_t2[t2_i].append((int(group_index), int(entry_pos), tuple(int(v) for v in entry)))

        out_by_t3 = defaultdict(list)
        for group_index, group in enumerate(plan["batched_out_entries"]):
            for entry_pos, entry in enumerate(group):
                t3_i, _f_i, _out_i = (int(value) for value in entry)
                out_by_t3[t3_i].append((int(group_index), int(entry_pos), tuple(int(v) for v in entry)))

        rows = []
        for r_group, r_entries in enumerate(plan["batched_r_entries"]):
            for r_pos, r_entry in enumerate(r_entries):
                e_i, a_i, r_i = (int(value) for value in r_entry)
                a_key = plan["a_keys"][a_i]
                if a_key not in offsets:
                    continue
                shape = tuple(int(dim) for dim in layout_shapes[a_key])
                if len(shape) != 4:
                    continue
                e_shape = tuple(int(dim) for dim in plan["e_blocks"][e_i].shape)
                if len(e_shape) != 3 or e_shape[1] != e_shape[2] or e_shape[1] != shape[0]:
                    continue
                for t2_group, t2_pos, t2_entry in t2_by_r.get(r_i, ()):
                    _t2_r_i, w1_i, t2_i = t2_entry
                    w1_shape = tuple(int(dim) for dim in plan["w1_blocks"][w1_i].shape)
                    if (
                        len(w1_shape) != 4
                        or w1_shape[2] != w1_shape[3]
                        or w1_shape[2] != shape[2]
                    ):
                        continue
                    for t3_group, t3_pos, t3_entry in t3_by_t2.get(t2_i, ()):
                        _t3_t2_i, w2_i, t3_i = t3_entry
                        w2_shape = tuple(int(dim) for dim in plan["w2_blocks"][w2_i].shape)
                        if (
                            len(w2_shape) != 4
                            or w2_shape[2] != w2_shape[3]
                            or w2_shape[2] != shape[3]
                        ):
                            continue
                        for out_group, out_pos, out_entry in out_by_t3.get(t3_i, ()):
                            _out_t3_i, f_i, out_i = out_entry
                            out_key = plan["out_keys"][out_i]
                            if out_key != a_key:
                                continue
                            if tuple(plan["out_shapes"][out_i]) != shape:
                                continue
                            f_shape = tuple(int(dim) for dim in plan["f_blocks"][f_i].shape)
                            if (
                                len(f_shape) != 3
                                or f_shape[1] != f_shape[2]
                                or f_shape[1] != shape[1]
                            ):
                                continue
                            flat_start, flat_size = offsets[a_key]
                            if int(flat_size) != int(np.prod(shape, dtype=int)):
                                continue
                            rows.append(
                                (
                                    int(r_group),
                                    int(r_pos),
                                    int(t2_group),
                                    int(t2_pos),
                                    int(t3_group),
                                    int(t3_pos),
                                    int(out_group),
                                    int(out_pos),
                                    int(flat_start),
                                    int(shape[0]),
                                    int(shape[1]),
                                    int(shape[2]),
                                    int(shape[3]),
                                )
                            )
        if not rows:
            return np.zeros((0, 13), dtype=np.int64)
        return np.ascontiguousarray(rows, dtype=np.int64)

    def compact_renormalized_table(self, operator, proto, layout):
        layout = tuple(layout)
        if not self._compact_plan_operator_enabled(operator, layout):
            return None
        if (
            _cpp_davidson is None
            or not getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)
            or getattr(_cpp_davidson, "CompactPlan", None) is None
        ):
            return None
        cache_key = self._compact_renormalized_table_structure_key(operator, proto, layout)
        numeric_token = operator._action_token()
        bond_slots_enabled = self._compact_renormalized_table_bond_slots_enabled()
        bond_slot_key = (
            self._compact_renormalized_table_bond_slot_key(operator, proto, layout)
            if bond_slots_enabled
            else None
        )
        if cache_key in self._compact_renormalized_table_cache:
            cached = self._compact_renormalized_table_cache[cache_key]
            if cached is None:
                return None
            self._record_compact_renormalized_table_cache_hit()
            if bond_slot_key is not None:
                self._compact_renormalized_table_bond_slots[bond_slot_key] = cached
            if getattr(cached, "numeric_token", None) == numeric_token:
                self._install_cpp_moving_environment_compact_plan(
                    cached,
                    operator,
                    layout,
                )
                return cached
            refresh_start = time.perf_counter()
            if cached.refresh_from_operator(operator):
                refresh_seconds = float(time.perf_counter() - refresh_start)
                self._record_compact_renormalized_table_refresh(
                    cached,
                    refresh_seconds,
                )
                self._install_cpp_moving_environment_compact_plan(
                    cached,
                    operator,
                    layout,
                )
                return cached
            self.moving_profile_stats["compact_plan_refresh_failures"] = int(
                self.moving_profile_stats.get("compact_plan_refresh_failures", 0)
            ) + 1
            self.moving_profile_stats["compact_renormalized_table_refresh_failures"] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_refresh_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["compact_plan_last_refresh_error"] = getattr(
                cached,
                "last_refresh_error",
                None,
            )
            self.moving_profile_stats[
                "compact_renormalized_table_last_refresh_error"
            ] = getattr(cached, "last_refresh_error", None)
        elif bond_slot_key is not None and bond_slot_key in self._compact_renormalized_table_bond_slots:
            cached = self._compact_renormalized_table_bond_slots[bond_slot_key]
            if cached is not None:
                self._record_compact_renormalized_table_cache_hit(bond_slot=True)
                if getattr(cached, "numeric_token", None) == numeric_token:
                    self._forget_compact_renormalized_table_structure(cached)
                    cached.structure_key = cache_key
                    self._compact_renormalized_table_cache[cache_key] = cached
                    self._install_cpp_moving_environment_compact_plan(
                        cached,
                        operator,
                        layout,
                    )
                    return cached
                refresh_start = time.perf_counter()
                if cached.refresh_from_operator(operator):
                    refresh_seconds = float(time.perf_counter() - refresh_start)
                    self._forget_compact_renormalized_table_structure(cached)
                    cached.structure_key = cache_key
                    self._compact_renormalized_table_cache[cache_key] = cached
                    self._record_compact_renormalized_table_refresh(
                        cached,
                        refresh_seconds,
                        bond_slot=True,
                    )
                    self._install_cpp_moving_environment_compact_plan(
                        cached,
                        operator,
                        layout,
                    )
                    return cached
                self.moving_profile_stats["compact_plan_bond_slot_refresh_failures"] = int(
                    self.moving_profile_stats.get(
                        "compact_plan_bond_slot_refresh_failures",
                        0,
                    )
                ) + 1
                self.moving_profile_stats["compact_plan_bond_slot_last_refresh_error"] = getattr(
                    cached,
                    "last_refresh_error",
                    None,
                )
                self._compact_renormalized_table_bond_slots.pop(bond_slot_key, None)
        build_start = time.perf_counter()
        try:
            dtype = np.result_type(np.complex128, operator._local_action_dtype(proto))
            if np.dtype(dtype) != np.dtype(np.complex128):
                dtype = np.complex128
            proto_full = operator._zero_proto_from_layout(proto, layout, dtype)
            plan = operator._build_compact_matrix_chain_plan(proto_full)
            if plan is None:
                self._compact_renormalized_table_cache[cache_key] = None
                return None
            if tuple(plan["a_keys"]) != tuple(key for key, _shape in layout):
                self._compact_renormalized_table_cache[cache_key] = None
                return None
            payload = operator._compact_stage_specs_payload(plan)
            if payload is None:
                self._compact_renormalized_table_cache[cache_key] = None
                return None
            info = operator._cython_arena_info(plan)
            offsets, dim = operator._layout_offsets(layout)
            layout_shapes = {key: tuple(shape) for key, shape in layout}
            e_keys = tuple(sorted(operator.E.data))
            w1_keys = tuple(sorted(operator.W[0].data))
            w2_keys = tuple(sorted(operator.W[1].data))
            f_keys = tuple(sorted(operator.F.data))

            a_flat_start = []
            a_flat_size = []
            for block_index, key in enumerate(plan["a_keys"]):
                if key not in offsets:
                    self._compact_renormalized_table_cache[cache_key] = None
                    return None
                start, size = offsets[key]
                if int(size) != int(np.prod(layout_shapes[key], dtype=int)):
                    self._compact_renormalized_table_cache[cache_key] = None
                    return None
                a_flat_start.append(int(start))
                a_flat_size.append(int(size))

            out_flat_start = []
            out_flat_size = []
            for out_index, key in enumerate(plan["out_keys"]):
                if key not in offsets:
                    self._compact_renormalized_table_cache[cache_key] = None
                    return None
                if tuple(plan["out_shapes"][out_index]) != layout_shapes[key]:
                    self._compact_renormalized_table_cache[cache_key] = None
                    return None
                start, size = offsets[key]
                out_flat_start.append(int(start))
                out_flat_size.append(int(size))

            a_offsets, a_sizes, a_total = info["a"]
            r_offsets, r_sizes, r_total = info["r"]
            t2_offsets, t2_sizes, t2_total = info["t2"]
            t3_offsets, t3_sizes, t3_total = info["t3"]
            out_offsets, out_sizes, out_total = info["out"]
            build_backend = "cpp_block_constructor"
            try:
                cpp_plan = _cpp_davidson.CompactPlan(
                    plan["e_blocks"],
                    plan["batched_r_entries"],
                    plan["w1_blocks"],
                    plan["batched_t2_entries"],
                    plan["w2_blocks"],
                    plan["batched_t3_entries"],
                    plan["f_blocks"],
                    plan["batched_out_entries"],
                    payload["r_specs"],
                    payload["t2_specs"],
                    payload["t3_specs"],
                    payload["out_specs"],
                    a_offsets,
                    a_sizes,
                    int(a_total),
                    r_offsets,
                    r_sizes,
                    int(r_total),
                    t2_offsets,
                    t2_sizes,
                    int(t2_total),
                    t3_offsets,
                    t3_sizes,
                    int(t3_total),
                    out_offsets,
                    out_sizes,
                    int(out_total),
                    np.ascontiguousarray(plan["a_groups"]["block_group"], dtype=np.int64),
                    np.ascontiguousarray(plan["a_groups"]["block_pos"], dtype=np.int64),
                    np.ascontiguousarray(a_flat_start, dtype=np.int64),
                    np.ascontiguousarray(a_flat_size, dtype=np.int64),
                    np.ascontiguousarray(plan["out_groups"]["block_group"], dtype=np.int64),
                    np.ascontiguousarray(plan["out_groups"]["block_pos"], dtype=np.int64),
                    np.ascontiguousarray(out_flat_start, dtype=np.int64),
                    np.ascontiguousarray(out_flat_size, dtype=np.int64),
                    int(dim),
                )
            except TypeError:
                stack_payload = operator._cython_compact_payload(plan)
                if stack_payload is None:
                    self._compact_renormalized_table_cache[cache_key] = None
                    return None
                build_backend = "python_stack_constructor"
                cpp_plan = _cpp_davidson.CompactPlan(
                    stack_payload["r_e"],
                    stack_payload["t2_w"],
                    stack_payload["t3_w"],
                    stack_payload["out_f"],
                    stack_payload["r_specs"],
                    stack_payload["t2_specs"],
                    stack_payload["t3_specs"],
                    stack_payload["out_specs"],
                    a_offsets,
                    a_sizes,
                    int(a_total),
                    r_offsets,
                    r_sizes,
                    int(r_total),
                    t2_offsets,
                    t2_sizes,
                    int(t2_total),
                    t3_offsets,
                    t3_sizes,
                    int(t3_total),
                    out_offsets,
                    out_sizes,
                    int(out_total),
                    np.ascontiguousarray(plan["a_groups"]["block_group"], dtype=np.int64),
                    np.ascontiguousarray(plan["a_groups"]["block_pos"], dtype=np.int64),
                    np.ascontiguousarray(a_flat_start, dtype=np.int64),
                    np.ascontiguousarray(a_flat_size, dtype=np.int64),
                    np.ascontiguousarray(plan["out_groups"]["block_group"], dtype=np.int64),
                    np.ascontiguousarray(plan["out_groups"]["block_pos"], dtype=np.int64),
                    np.ascontiguousarray(out_flat_start, dtype=np.int64),
                    np.ascontiguousarray(out_flat_size, dtype=np.int64),
                    int(dim),
                )
            direct = MovingEnvironmentCompactRenormalizedTable(cpp_plan, int(dim), layout)
            direct.build_backend = build_backend
            direct.structure_key = cache_key
            direct.numeric_token = numeric_token
            direct.install_refresh_recipe(
                e_keys=e_keys,
                w1_keys=w1_keys,
                w2_keys=w2_keys,
                f_keys=f_keys,
                entries={
                    "r": plan["batched_r_entries"],
                    "t2": plan["batched_t2_entries"],
                    "t3": plan["batched_t3_entries"],
                    "out": plan["batched_out_entries"],
                },
            )
            diagonal_routes = self._compact_diagonal_routes(
                plan,
                offsets,
                layout_shapes,
            )
            direct.install_diagonal_routes(diagonal_routes)
            direct.n_entries = int(
                len(plan["r_entries"])
                + len(plan["t2_entries"])
                + len(plan["t3_entries"])
                + len(plan["out_entries"])
            )
            direct.n_groups = int(
                len(plan["batched_r_entries"])
                + len(plan["batched_t2_entries"])
                + len(plan["batched_t3_entries"])
                + len(plan["batched_out_entries"])
            )
            direct.n_group_channels = int(direct.n_groups)
            direct.validation_key = self._compact_plan_validation_key(
                plan,
                layout,
                proto.dirs[:],
            )
        except MemoryError:
            self.moving_profile_stats["compact_plan_failures"] = int(
                self.moving_profile_stats.get("compact_plan_failures", 0)
            ) + 1
            self.moving_profile_stats["compact_renormalized_table_failures"] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_failures",
                    0,
                )
            ) + 1
            self._compact_renormalized_table_cache[cache_key] = None
            return None
        except Exception as exc:
            self.moving_profile_stats["compact_plan_failures"] = int(
                self.moving_profile_stats.get("compact_plan_failures", 0)
            ) + 1
            self.moving_profile_stats["compact_renormalized_table_failures"] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["compact_plan_last_error"] = str(exc)
            self.moving_profile_stats["compact_renormalized_table_last_error"] = str(exc)
            self._compact_renormalized_table_cache[cache_key] = None
            return None

        build_seconds = float(time.perf_counter() - build_start)
        self._compact_renormalized_table_cache[cache_key] = direct
        if bond_slot_key is not None:
            self._compact_renormalized_table_bond_slots[bond_slot_key] = direct
            self.moving_profile_stats["compact_plan_bond_slot_stores"] = int(
                self.moving_profile_stats.get("compact_plan_bond_slot_stores", 0)
            ) + 1
            self.moving_profile_stats[
                "compact_renormalized_table_bond_slot_stores"
            ] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_bond_slot_stores",
                    0,
                )
            ) + 1
        self._install_cpp_moving_environment_compact_plan(
            direct,
            operator,
            layout,
        )
        self.moving_profile_stats["compact_plan_builds"] = int(
            self.moving_profile_stats.get("compact_plan_builds", 0)
        ) + 1
        self.moving_profile_stats["compact_renormalized_table_builds"] = int(
            self.moving_profile_stats.get("compact_renormalized_table_builds", 0)
        ) + 1
        build_backend = str(getattr(direct, "build_backend", "unknown"))
        self.moving_profile_stats["compact_renormalized_table_build_backend"] = build_backend
        if build_backend == "cpp_block_constructor":
            self.moving_profile_stats[
                "compact_renormalized_table_cpp_block_constructor_builds"
            ] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_cpp_block_constructor_builds",
                    0,
                )
            ) + 1
        elif build_backend == "python_stack_constructor":
            self.moving_profile_stats[
                "compact_renormalized_table_python_stack_constructor_builds"
            ] = int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_python_stack_constructor_builds",
                    0,
                )
            ) + 1
        self.moving_profile_stats["compact_plan_build_seconds"] = float(
            self.moving_profile_stats.get("compact_plan_build_seconds", 0.0)
        ) + build_seconds
        self.moving_profile_stats["compact_renormalized_table_build_seconds"] = float(
            self.moving_profile_stats.get(
                "compact_renormalized_table_build_seconds",
                0.0,
            )
        ) + build_seconds
        self.moving_profile_stats["compact_plan_last_dimension"] = int(direct.dim)
        self.moving_profile_stats["compact_plan_last_entries"] = int(direct.n_entries)
        self.moving_profile_stats["compact_plan_last_groups"] = int(direct.n_groups)
        self.moving_profile_stats["compact_renormalized_table_last_dimension"] = int(
            direct.dim
        )
        self.moving_profile_stats["compact_renormalized_table_last_entries"] = int(
            direct.n_entries
        )
        self.moving_profile_stats["compact_renormalized_table_last_groups"] = int(
            direct.n_groups
        )
        self.moving_profile_stats["compact_renormalized_table_last_diagonal_routes"] = int(
            getattr(direct, "n_diagonal_routes", 0)
        )
        return direct

    def compact_plan_operator(self, operator, proto, layout):
        return self.compact_renormalized_table(operator, proto, layout)

    @staticmethod
    def _compact_direct_block_matrix(e_blk, w1_blk, w2_blk, f_blk):
        matrix = np.einsum(
            "aij,abux,bcvy,clk->iluvjkxy",
            np.asarray(e_blk, dtype=np.complex128),
            np.asarray(w1_blk, dtype=np.complex128),
            np.asarray(w2_blk, dtype=np.complex128),
            np.asarray(f_blk, dtype=np.complex128),
            optimize=True,
        )
        out_shape = matrix.shape[:4]
        in_shape = matrix.shape[4:]
        return np.ascontiguousarray(
            matrix.reshape(
                int(np.prod(out_shape, dtype=int)),
                int(np.prod(in_shape, dtype=int)),
            )
        )

    def compact_block_table(self, operator, proto, layout):
        layout = tuple(layout)
        if not self._compact_block_table_enabled(operator, layout):
            return None
        cache_key = (
            "moving_environment_compact_block_table",
            operator._action_token(),
            layout,
            tuple(proto.dirs),
        )
        cached = self._compact_block_table_cache.get(cache_key)
        if cached is not None:
            self.moving_profile_stats["compact_block_table_cache_hits"] = int(
                self.moving_profile_stats.get("compact_block_table_cache_hits", 0)
            ) + 1
            return cached
        build_start = time.perf_counter()
        try:
            dtype = np.result_type(np.complex128, operator._local_action_dtype(proto))
            proto_full = operator._zero_proto_from_layout(proto, layout, dtype)
            plan = operator._build_compact_matrix_chain_plan(proto_full)
            if plan is None:
                self._compact_block_table_cache[cache_key] = None
                return None
            if tuple(plan["a_keys"]) != tuple(key for key, _shape in layout):
                self._compact_block_table_cache[cache_key] = None
                return None
            offsets, dim = operator._layout_offsets(layout)
            layout_shapes = {key: tuple(shape) for key, shape in layout}
            t2_by_r = defaultdict(list)
            for r_i, w1_i, t2_i in plan["t2_entries"]:
                t2_by_r[int(r_i)].append((int(w1_i), int(t2_i)))
            t3_by_t2 = defaultdict(list)
            for t2_i, w2_i, t3_i in plan["t3_entries"]:
                t3_by_t2[int(t2_i)].append((int(w2_i), int(t3_i)))
            out_by_t3 = defaultdict(list)
            for t3_i, f_i, out_i in plan["out_entries"]:
                out_by_t3[int(t3_i)].append((int(f_i), int(out_i)))

            estimated_elements = 0
            estimated_blocks = {}
            routes = []
            for e_i, a_i, r_i in plan["r_entries"]:
                a_i = int(a_i)
                a_key = plan["a_keys"][a_i]
                in_offset, in_dim = offsets[a_key]
                if int(np.prod(layout_shapes[a_key], dtype=int)) != int(in_dim):
                    self._compact_block_table_cache[cache_key] = None
                    return None
                for w1_i, t2_i in t2_by_r.get(int(r_i), ()):
                    for w2_i, t3_i in t3_by_t2.get(int(t2_i), ()):
                        for f_i, out_i in out_by_t3.get(int(t3_i), ()):
                            out_i = int(out_i)
                            out_key = plan["out_keys"][out_i]
                            out_offset = offsets.get(out_key)
                            if out_offset is None:
                                self._compact_block_table_cache[cache_key] = None
                                return None
                            out_start, out_dim = out_offset
                            if tuple(plan["out_shapes"][out_i]) != layout_shapes[out_key]:
                                self._compact_block_table_cache[cache_key] = None
                                return None
                            key = (int(out_i), int(a_i), int(out_start), int(in_offset))
                            if key not in estimated_blocks:
                                n_elements = int(out_dim) * int(in_dim)
                                estimated_blocks[key] = n_elements
                                estimated_elements += n_elements
                            routes.append(
                                (
                                    key,
                                    int(e_i),
                                    int(w1_i),
                                    int(w2_i),
                                    int(f_i),
                                    int(out_dim),
                                    int(in_dim),
                                )
                            )

            block_accum = {}
            for key, e_i, w1_i, w2_i, f_i, out_dim, in_dim in routes:
                matrix = self._compact_direct_block_matrix(
                    plan["e_blocks"][int(e_i)],
                    plan["w1_blocks"][int(w1_i)],
                    plan["w2_blocks"][int(w2_i)],
                    plan["f_blocks"][int(f_i)],
                )
                if matrix.shape != (int(out_dim), int(in_dim)):
                    self._compact_block_table_cache[cache_key] = None
                    return None
                existing = block_accum.get(key)
                block_accum[key] = matrix if existing is None else existing + matrix
            if not block_accum:
                self._compact_block_table_cache[cache_key] = None
                return None
            block_matrices = []
            in_starts = []
            out_starts = []
            for (_out_i, _a_i, out_start, in_start), matrix in sorted(block_accum.items()):
                block_matrices.append(np.ascontiguousarray(matrix, dtype=np.complex128))
                in_starts.append(int(in_start))
                out_starts.append(int(out_start))
            table = MovingEnvironmentCompactBlockTable(
                block_matrices,
                in_starts,
                out_starts,
                int(dim),
                layout,
            )
        except MemoryError:
            self.moving_profile_stats["compact_block_table_failures"] = int(
                self.moving_profile_stats.get("compact_block_table_failures", 0)
            ) + 1
            self._compact_block_table_cache[cache_key] = None
            return None
        except Exception as exc:
            self.moving_profile_stats["compact_block_table_failures"] = int(
                self.moving_profile_stats.get("compact_block_table_failures", 0)
            ) + 1
            self.moving_profile_stats["compact_block_table_last_error"] = str(exc)
            self._compact_block_table_cache[cache_key] = None
            return None

        build_seconds = float(time.perf_counter() - build_start)
        self._compact_block_table_cache[cache_key] = table
        self.moving_profile_stats["compact_block_table_builds"] = int(
            self.moving_profile_stats.get("compact_block_table_builds", 0)
        ) + 1
        self.moving_profile_stats["compact_block_table_build_seconds"] = float(
            self.moving_profile_stats.get("compact_block_table_build_seconds", 0.0)
        ) + build_seconds
        self.moving_profile_stats["compact_block_table_last_dimension"] = int(table.dim)
        self.moving_profile_stats["compact_block_table_last_blocks"] = int(
            table.n_block_matrices
        )
        self.moving_profile_stats["compact_block_table_last_elements"] = int(
            table.block_matrix_elements
        )
        return table

    def _validate_compact_block_table(self, table, operator, proto, layout, vector):
        validate = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec",
                True,
            )
        )
        if not validate:
            return True
        validated = getattr(table, "_moving_environment_compact_table_validated", None)
        if validated is not None:
            return bool(validated)
        vector = np.ascontiguousarray(vector, dtype=np.complex128).reshape(table.dim)
        try:
            ref = operator._flat_batched_compact_matrix_chain(vector, proto, layout)
            if ref is None:
                return False
            test = table.matvec(vector)
        except Exception as exc:
            self.moving_profile_stats["compact_block_table_validation_failures"] = int(
                self.moving_profile_stats.get(
                    "compact_block_table_validation_failures",
                    0,
                )
            ) + 1
            self.moving_profile_stats["compact_block_table_validation_last_error"] = str(exc)
            setattr(table, "_moving_environment_compact_table_validated", False)
            return False
        diff = float(np.linalg.norm(test - ref))
        rel = diff / max(1.0, float(np.linalg.norm(ref)))
        tol = float(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec_tol",
                1.0e-10,
            )
        )
        self.moving_profile_stats["compact_block_table_validation_calls"] = int(
            self.moving_profile_stats.get("compact_block_table_validation_calls", 0)
        ) + 1
        self.moving_profile_stats["compact_block_table_validation_last_error_norm"] = diff
        self.moving_profile_stats["compact_block_table_validation_last_relative_error"] = rel
        ok = bool(rel <= tol)
        if not ok:
            self.moving_profile_stats["compact_block_table_validation_failures"] = int(
                self.moving_profile_stats.get(
                    "compact_block_table_validation_failures",
                    0,
                )
            ) + 1
        setattr(table, "_moving_environment_compact_table_validated", ok)
        return ok

    def _validate_compact_plan_operator(self, direct, operator, proto, layout, vector):
        validate = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec",
                True,
            )
        )
        if not validate:
            return True
        validated = getattr(direct, "_moving_environment_compact_plan_validated", None)
        if validated is not None:
            return bool(validated)
        validation_key = getattr(direct, "validation_key", None)
        if validation_key is not None and validation_key in self._compact_plan_validation_cache:
            ok = bool(self._compact_plan_validation_cache[validation_key])
            setattr(direct, "_moving_environment_compact_plan_validated", ok)
            self.moving_profile_stats["compact_plan_validation_cache_hits"] = int(
                self.moving_profile_stats.get("compact_plan_validation_cache_hits", 0)
            ) + 1
            return ok
        vector = np.ascontiguousarray(vector, dtype=np.complex128).reshape(direct.dim)
        test_vectors = [vector]
        random_vectors = int(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec_random_vectors",
                1,
            )
        )
        if random_vectors > 0:
            seed = 9176 + int(direct.dim) * 37 + int(len(layout)) * 101
            rng = np.random.default_rng(seed)
            for _ in range(random_vectors):
                real = rng.standard_normal(int(direct.dim))
                imag = rng.standard_normal(int(direct.dim))
                test_vectors.append(
                    np.ascontiguousarray(real + 1j * imag, dtype=np.complex128)
                )
        try:
            worst_diff = 0.0
            worst_rel = 0.0
            for trial in test_vectors:
                ref = operator._flat_batched_compact_matrix_chain(trial, proto, layout)
                if ref is None:
                    return False
                test = direct.matvec(trial)
                diff_i = float(np.linalg.norm(test - ref))
                rel_i = diff_i / max(1.0, float(np.linalg.norm(ref)))
                worst_diff = max(worst_diff, diff_i)
                worst_rel = max(worst_rel, rel_i)
        except Exception as exc:
            self.moving_profile_stats["compact_plan_validation_failures"] = int(
                self.moving_profile_stats.get("compact_plan_validation_failures", 0)
            ) + 1
            self.moving_profile_stats["compact_plan_validation_last_error"] = str(exc)
            setattr(direct, "_moving_environment_compact_plan_validated", False)
            return False
        diff = float(worst_diff)
        rel = float(worst_rel)
        tol = float(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_validate_matvec_tol",
                1.0e-10,
            )
        )
        self.moving_profile_stats["compact_plan_validation_calls"] = int(
            self.moving_profile_stats.get("compact_plan_validation_calls", 0)
        ) + 1
        self.moving_profile_stats["compact_plan_validation_last_error_norm"] = diff
        self.moving_profile_stats["compact_plan_validation_last_relative_error"] = rel
        ok = bool(rel <= tol)
        if not ok:
            self.moving_profile_stats["compact_plan_validation_failures"] = int(
                self.moving_profile_stats.get("compact_plan_validation_failures", 0)
            ) + 1
        setattr(direct, "_moving_environment_compact_plan_validated", ok)
        if validation_key is not None:
            self._compact_plan_validation_cache[validation_key] = ok
        return ok

    def solve_cpp_davidson(
        self,
        operator,
        proto,
        layout,
        v_flat,
        *,
        tol,
        max_iter,
        restart_dim,
        accept_unconverged=False,
    ):
        if _cpp_davidson is None or not getattr(
            _cpp_davidson,
            "CPP_DAVIDSON_AVAILABLE",
            False,
        ):
            return None
        if not bool(getattr(operator, "_moving_environment_cpp_davidson", False)):
            return None
        direct = None
        tried_direct = False
        solver = None
        diagonal = None
        table_source = "renormalized_table"
        if bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_compact_plan",
                False,
            )
        ):
            tried_direct = True
            direct = self.compact_renormalized_table(operator, proto, layout)
            table_source = "compact_renormalized_table"
            if direct is not None and not self._validate_compact_plan_operator(
                direct,
                operator,
                proto,
                layout,
                v_flat,
            ):
                direct = None
            if direct is not None:
                solver = direct
                if getattr(direct, "cpp_moving_environment", None) is not None:
                    diagonal = np.empty(0, dtype=np.complex128)
                    self.moving_profile_stats[
                        "compact_renormalized_table_diagonal_calls"
                    ] = int(
                        self.moving_profile_stats.get(
                            "compact_renormalized_table_diagonal_calls",
                            0,
                        )
                    ) + 1
                    self.moving_profile_stats[
                        "compact_renormalized_table_last_diagonal_seconds"
                    ] = 0.0
                    self.moving_profile_stats[
                        "compact_renormalized_table_diagonal_backend"
                    ] = "cpp_moving_environment_routes"
                else:
                    diagonal_start = time.perf_counter()
                    diagonal = direct.diagonal_flat()
                    if diagonal is not None:
                        diagonal_seconds = float(time.perf_counter() - diagonal_start)
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_calls"
                        ] = int(
                            self.moving_profile_stats.get(
                                "compact_renormalized_table_diagonal_calls",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_seconds"
                        ] = float(
                            self.moving_profile_stats.get(
                                "compact_renormalized_table_diagonal_seconds",
                                0.0,
                            )
                        ) + diagonal_seconds
                        self.moving_profile_stats[
                            "compact_renormalized_table_last_diagonal_seconds"
                        ] = diagonal_seconds
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_backend"
                        ] = "cpp_routes"
                    else:
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_fallbacks"
                        ] = int(
                            self.moving_profile_stats.get(
                                "compact_renormalized_table_diagonal_fallbacks",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "compact_renormalized_table_last_diagonal_error"
                        ] = getattr(direct, "last_diagonal_error", None)
                        diagonal = operator._flat_jacobi_diagonal(proto, layout)

        compiled = None
        if solver is None:
            compiled = self.compiled_flat_matvec(operator, proto, layout)
        table = None if compiled is None else compiled.table
        if solver is None and table is not None:
            table_source = "renormalized_table"
            if (
                getattr(table, "cpp_moving_environment", None) is not None
                and hasattr(table, "davidson")
            ):
                solver = table
                table_source = "grouped_renormalized_table_owner"
                diagonal = np.empty(0, dtype=np.complex128)
            else:
                cpp_table = self.cpp_renormalized_table(
                    table,
                    validation_vector=v_flat,
                )
                if cpp_table is None:
                    return None
                solver = cpp_table
            if diagonal is None and hasattr(solver, "diagonal"):
                diagonal_start = time.perf_counter()
                try:
                    diagonal = np.asarray(
                        solver.diagonal(),
                        dtype=np.complex128,
                    ).reshape(int(table.dim))
                except Exception as exc:
                    self.moving_profile_stats[
                        "cpp_renormalized_table_last_diagonal_error"
                    ] = str(exc)
                    diagonal = None
                else:
                    diagonal_seconds = float(time.perf_counter() - diagonal_start)
                    self.moving_profile_stats[
                        "cpp_renormalized_table_diagonal_calls"
                    ] = int(
                        self.moving_profile_stats.get(
                            "cpp_renormalized_table_diagonal_calls",
                            0,
                        )
                    ) + 1
                    self.moving_profile_stats[
                        "cpp_renormalized_table_diagonal_seconds"
                    ] = float(
                        self.moving_profile_stats.get(
                            "cpp_renormalized_table_diagonal_seconds",
                            0.0,
                        )
                    ) + diagonal_seconds
                    self.moving_profile_stats[
                        "cpp_renormalized_table_last_diagonal_seconds"
                    ] = diagonal_seconds
            if diagonal is None:
                diagonal = compiled.diagonal()
        elif solver is None:
            if not tried_direct:
                direct = self.compact_renormalized_table(operator, proto, layout)
            table_source = "compact_renormalized_table"
            if direct is not None and not self._validate_compact_plan_operator(
                direct,
                operator,
                proto,
                layout,
                v_flat,
            ):
                direct = None
            if direct is not None:
                solver = direct
                if getattr(direct, "cpp_moving_environment", None) is not None:
                    diagonal = np.empty(0, dtype=np.complex128)
                    self.moving_profile_stats[
                        "compact_renormalized_table_diagonal_calls"
                    ] = int(
                        self.moving_profile_stats.get(
                            "compact_renormalized_table_diagonal_calls",
                            0,
                        )
                    ) + 1
                    self.moving_profile_stats[
                        "compact_renormalized_table_last_diagonal_seconds"
                    ] = 0.0
                    self.moving_profile_stats[
                        "compact_renormalized_table_diagonal_backend"
                    ] = "cpp_moving_environment_routes"
                else:
                    diagonal_start = time.perf_counter()
                    diagonal = direct.diagonal_flat()
                    if diagonal is not None:
                        diagonal_seconds = float(time.perf_counter() - diagonal_start)
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_calls"
                        ] = int(
                            self.moving_profile_stats.get(
                                "compact_renormalized_table_diagonal_calls",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_seconds"
                        ] = float(
                            self.moving_profile_stats.get(
                                "compact_renormalized_table_diagonal_seconds",
                                0.0,
                            )
                        ) + diagonal_seconds
                        self.moving_profile_stats[
                            "compact_renormalized_table_last_diagonal_seconds"
                        ] = diagonal_seconds
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_backend"
                        ] = "cpp_routes"
                    else:
                        self.moving_profile_stats[
                            "compact_renormalized_table_diagonal_fallbacks"
                        ] = int(
                            self.moving_profile_stats.get(
                                "compact_renormalized_table_diagonal_fallbacks",
                                0,
                            )
                        ) + 1
                        self.moving_profile_stats[
                            "compact_renormalized_table_last_diagonal_error"
                        ] = getattr(direct, "last_diagonal_error", None)
                        diagonal = operator._flat_jacobi_diagonal(proto, layout)

        if solver is None:
            table = self.compact_block_table(operator, proto, layout)
            table_source = "compact_block_table"
            if table is not None and not self._validate_compact_block_table(
                table,
                operator,
                proto,
                layout,
                v_flat,
            ):
                return None
            if table is None:
                return None
            cpp_table = self.cpp_block_table(table, validation_vector=v_flat)
            if cpp_table is None:
                return None
            solver = cpp_table
            diagonal = table.diagonal_flat()
        if diagonal is None:
            return None
        start = time.perf_counter()
        self.moving_profile_stats["cpp_davidson_attempts"] = int(
            self.moving_profile_stats.get("cpp_davidson_attempts", 0)
        ) + 1
        try:
            result = solver.davidson(
                np.ascontiguousarray(diagonal, dtype=np.complex128),
                np.ascontiguousarray(v_flat, dtype=np.complex128),
                float(tol),
                int(max_iter),
                int(restart_dim),
                bool(accept_unconverged),
            )
        except Exception as exc:
            self.moving_profile_stats["cpp_davidson_failures"] = int(
                self.moving_profile_stats.get("cpp_davidson_failures", 0)
            ) + 1
            self.moving_profile_stats["cpp_davidson_last_error"] = str(exc)
            return None
        elapsed = float(time.perf_counter() - start)
        self.moving_profile_stats["cpp_davidson_calls"] = int(
            self.moving_profile_stats.get("cpp_davidson_calls", 0)
        ) + 1
        self.moving_profile_stats["cpp_davidson_table_source"] = str(table_source)
        self.moving_profile_stats["cpp_davidson_seconds"] = float(
            self.moving_profile_stats.get("cpp_davidson_seconds", 0.0)
        ) + elapsed
        self.moving_profile_stats["cpp_davidson_last_seconds"] = elapsed
        result["table_source"] = str(table_source)
        try:
            self.moving_profile_stats["cpp_davidson_last_solver_calls"] = int(
                solver.davidson_calls()
            )
        except Exception:
            pass
        try:
            self.moving_profile_stats[
                "cpp_davidson_last_solver_workspace_reuses"
            ] = int(solver.davidson_workspace_reuses())
        except Exception:
            pass
        if bool(result.get("workspace_reused", False)):
            self.moving_profile_stats["cpp_davidson_workspace_reuses"] = int(
                self.moving_profile_stats.get("cpp_davidson_workspace_reuses", 0)
            ) + 1
        if not bool(result.get("accepted", False)):
            self.moving_profile_stats["cpp_davidson_rejected"] = int(
                self.moving_profile_stats.get("cpp_davidson_rejected", 0)
            ) + 1
            self.moving_profile_stats["cpp_davidson_last_residual"] = float(
                result.get("residual_norm", math.inf)
            )
        try:
            iterations = int(result.get("iterations", 0))
        except Exception:
            iterations = 0
        self.moving_profile_stats["compiled_flat_matvec_calls"] = int(
            self.moving_profile_stats.get("compiled_flat_matvec_calls", 0)
        ) + iterations
        self.moving_profile_stats["compiled_flat_matvec_seconds"] = float(
            self.moving_profile_stats.get("compiled_flat_matvec_seconds", 0.0)
        ) + elapsed
        return result

    def prepare_owner_grouped_update(self, operator, proto, layout):
        env = self._cpp_moving_environment
        stats = self.moving_profile_stats
        stats["owner_local_grouped_direct_prepare_calls"] = int(
            stats.get("owner_local_grouped_direct_prepare_calls", 0)
        ) + 1
        if (
            env is None
            or not hasattr(env, "grouped_davidson_split_flat_two_site_update_auto")
            or not hasattr(env, "bond_step_update_and_environment_auto")
        ):
            return None
        if not bool(getattr(operator, "_moving_environment_cpp_davidson", False)):
            return None
        try:
            compiled = self.compiled_flat_matvec(operator, proto, layout)
            table = None if compiled is None else compiled.table
            if (
                table is None
                or getattr(table, "cpp_moving_environment", None) is not env
            ):
                stats["owner_local_grouped_direct_prepare_last_error"] = (
                    "grouped_table_unavailable"
                )
                return None
            packed_layout, sector_decoder = (
                _pack_two_site_split_layout_integer_sector_ids(layout)
            )
            qns = tuple(tuple(axis) for axis in getattr(proto, "qns", ()) or ())
            dirs = tuple(int(d) for d in getattr(proto, "dirs", ()) or ())
        except Exception as exc:
            stats["owner_local_grouped_direct_prepare_failures"] = int(
                stats.get("owner_local_grouped_direct_prepare_failures", 0)
            ) + 1
            stats["owner_local_grouped_direct_prepare_last_error"] = str(exc)
            return None
        stats["owner_local_grouped_direct_prepare_accepts"] = int(
            stats.get("owner_local_grouped_direct_prepare_accepts", 0)
        ) + 1
        stats["owner_local_grouped_direct_prepare_last_error"] = None
        return (
            packed_layout,
            qns,
            dirs,
            AbelianSiteTensorData,
            sector_decoder,
        )

    def solve_cpp_davidson_update(
        self,
        operator,
        proto,
        layout,
        v_flat,
        *,
        tol,
        max_iter,
        restart_dim,
        accept_unconverged=False,
        direction="right",
        m_max=None,
    ):
        env = self._cpp_moving_environment
        enabled = bool(
            self._option_value(
                self.matvec_options,
                "moving_environment_cpp_solve_site_update_owner",
                bool(
                    self._option_value(
                        self.matvec_options,
                        "moving_environment_cpp_state_owner",
                        False,
                    )
                ),
            )
        )
        if (
            not enabled
            or env is None
            or (
                not hasattr(env, "grouped_davidson_split_flat_two_site_update_auto")
                and not hasattr(env, "grouped_davidson_split_flat_two_site_update")
            )
        ):
            return None
        if _cpp_davidson is None or not getattr(
            _cpp_davidson,
            "CPP_DAVIDSON_AVAILABLE",
            False,
        ):
            return None
        if not bool(getattr(operator, "_moving_environment_cpp_davidson", False)):
            return None
        start = time.perf_counter()
        self.moving_profile_stats["cpp_solve_update_attempts"] = int(
            self.moving_profile_stats.get("cpp_solve_update_attempts", 0)
        ) + 1
        use_transaction = False
        try:
            compiled = self.compiled_flat_matvec(operator, proto, layout)
            table = None if compiled is None else compiled.table
            if (
                table is None
                or getattr(table, "cpp_moving_environment", None) is not env
            ):
                self.moving_profile_stats[
                    "cpp_moving_environment_solve_update_backend"
                ] = "unsupported"
                return None
            packed_layout, sector_decoder = (
                _pack_two_site_split_layout_integer_sector_ids(layout)
            )
            qns = tuple(tuple(axis) for axis in getattr(proto, "qns", ()) or ())
            dirs = tuple(int(d) for d in getattr(proto, "dirs", ()) or ())
            update_auto = getattr(
                env,
                "grouped_davidson_split_flat_two_site_update_auto",
                None,
            )
            update_args = (
                np.ascontiguousarray(v_flat, dtype=np.complex128),
                float(tol),
                int(max_iter),
                int(restart_dim),
                bool(accept_unconverged),
                packed_layout,
                qns,
                dirs,
                str(direction),
                AbelianSiteTensorData,
                m_max,
                sector_decoder,
            )
            pending_step = self._pending_cpp_bond_environment_step
            transaction = getattr(
                env,
                "bond_step_update_and_environment_auto",
                None,
            )
            use_transaction = (
                transaction is not None
                and pending_step is not None
                and int(pending_step.get("bond", -1)) == int(operator.bond)
            )
            if use_transaction:
                bond_hint = None if operator.bond is None else int(operator.bond)
                self.moving_profile_stats[
                    "cpp_bond_step_transaction_attempts"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_bond_step_transaction_attempts",
                        0,
                    )
                ) + 1
                self.moving_profile_stats[
                    "cpp_bond_step_transaction_calls"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_bond_step_transaction_calls",
                        0,
                    )
                ) + 1
                self.moving_profile_stats[
                    "cpp_moving_environment_solve_update_auto_calls"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_moving_environment_solve_update_auto_calls",
                        0,
                    )
                ) + 1
                (
                    result,
                    left,
                    right,
                    s_data,
                    bond_qns,
                    trunc,
                    kept,
                    _native_stats,
                    env_updates,
                    env_pops,
                    env_syncs,
                    env_failures,
                ) = transaction(
                    bond_hint,
                    *update_args,
                    str(pending_step["environment_direction"]),
                    pending_step["update_rows"],
                    pending_step["pop_rows"],
                )
                self._pending_cpp_bond_environment_step = None
                moved = (
                    bool(result.get("accepted", False))
                    and int(env_failures) == 0
                    and (int(env_updates) > 0 or int(env_pops) > 0)
                )
                if moved:
                    self._last_cpp_bond_environment_step = {
                        "sweep_direction": pending_step["sweep_direction"],
                        "bond": int(pending_step["bond"]),
                        "updates": int(env_updates),
                        "pops": int(env_pops),
                        "syncs": int(env_syncs),
                        "failures": int(env_failures),
                        "environment_direction": pending_step[
                            "environment_direction"
                        ],
                        "update_records": pending_step.get(
                            "update_records",
                            (),
                        ),
                        "pop_records": pending_step.get("pop_records", ()),
                    }
                    self.moving_profile_stats[
                        "cpp_bond_step_transaction_accepted"
                    ] = int(
                        self.moving_profile_stats.get(
                            "cpp_bond_step_transaction_accepted",
                            0,
                        )
                    ) + 1
                    self.moving_profile_stats[
                        "cpp_bond_step_transaction_environment_updates"
                    ] = int(
                        self.moving_profile_stats.get(
                            "cpp_bond_step_transaction_environment_updates",
                            0,
                        )
                    ) + int(env_updates)
                    self.moving_profile_stats[
                        "cpp_bond_step_transaction_backend_actual"
                    ] = "cpp_moving_environment"
            elif update_auto is not None:
                bond_hint = None if operator.bond is None else int(operator.bond)
                self.moving_profile_stats[
                    "cpp_moving_environment_solve_update_auto_calls"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_moving_environment_solve_update_auto_calls",
                        0,
                    )
                ) + 1
                (
                    result,
                    left,
                    right,
                    s_data,
                    bond_qns,
                    trunc,
                    kept,
                    _native_stats,
                ) = update_auto(bond_hint, *update_args)
            else:
                key = getattr(table, "cpp_moving_environment_key", None)
                if key is None:
                    self.moving_profile_stats[
                        "cpp_moving_environment_solve_update_backend"
                    ] = "missing_key"
                    return None
                (
                    result,
                    left,
                    right,
                    s_data,
                    bond_qns,
                    trunc,
                    kept,
                    _native_stats,
                ) = env.grouped_davidson_split_flat_two_site_update(
                    str(key),
                    *update_args,
                )
        except Exception as exc:
            self.moving_profile_stats["cpp_solve_update_failures"] = int(
                self.moving_profile_stats.get("cpp_solve_update_failures", 0)
            ) + 1
            if use_transaction:
                self.moving_profile_stats[
                    "cpp_bond_step_transaction_failures"
                ] = int(
                    self.moving_profile_stats.get(
                        "cpp_bond_step_transaction_failures",
                        0,
                    )
                ) + 1
                self.moving_profile_stats[
                    "cpp_bond_step_transaction_last_error"
                ] = str(exc)
            self.moving_profile_stats[
                "cpp_moving_environment_solve_update_flat_last_error"
            ] = str(exc)
            self.moving_profile_stats[
                "cpp_moving_environment_solve_update_backend"
            ] = "failed"
            self._sync_cpp_moving_environment_stats()
            return None
        finally:
            elapsed = float(time.perf_counter() - start)
            self.moving_profile_stats["cpp_solve_update_seconds"] = float(
                self.moving_profile_stats.get("cpp_solve_update_seconds", 0.0)
            ) + elapsed
            self.moving_profile_stats["cpp_solve_update_last_seconds"] = elapsed
            if use_transaction:
                self.moving_profile_stats[
                    "cpp_bond_step_transaction_seconds"
                ] = float(
                    self.moving_profile_stats.get(
                        "cpp_bond_step_transaction_seconds",
                        0.0,
                    )
                ) + elapsed
                self.moving_profile_stats[
                    "cpp_bond_step_transaction_last_seconds"
                ] = elapsed
        self._sync_cpp_moving_environment_stats()
        result = dict(result)
        result["table_source"] = "grouped_renormalized_table_owner_fused_update"
        self.moving_profile_stats[
            "cpp_moving_environment_solve_update_backend"
        ] = "cpp_moving_environment"
        self.moving_profile_stats[
            "cpp_moving_environment_site_split_backend"
        ] = "cpp_moving_environment"
        self.moving_profile_stats[
            "cpp_moving_environment_site_update_backend"
        ] = "cpp_moving_environment"
        if not bool(result.get("accepted", False)) or left is None or right is None:
            return result, None
        update = AbelianTwoSiteUpdateData(
            left,
            right,
            OrderedDict(
                (key, np.asarray(block)) for key, block in (s_data or {}).items()
            ),
            tuple(bond_qns or ()),
            float(trunc),
            int(kept),
        )
        return result, update

    def update_left(self, W, A, E, B):
        start = time.perf_counter()
        self._cpp_environment_stack_seed_direct("left", E)
        self._environment_advance_slot_key = ("left", "direct", 0)
        try:
            updated = self.compiled_backend.update_left_environment(W, A, E, B)
            self._cpp_environment_stack_replace_direct("left", updated)
            return updated
        finally:
            self._environment_advance_slot_key = None
            self.moving_profile_stats["environment_update_backend"] = getattr(
                self,
                "_last_environment_update_backend",
                "python_contract",
            )
            self._record_environment_update("update_left", time.perf_counter() - start)

    def update_right(self, W, A, F, B):
        start = time.perf_counter()
        self._cpp_environment_stack_seed_direct("right", F)
        self._environment_advance_slot_key = ("right", "direct", 0)
        try:
            updated = self.compiled_backend.update_right_environment(W, A, F, B)
            self._cpp_environment_stack_replace_direct("right", updated)
            return updated
        finally:
            self._environment_advance_slot_key = None
            self.moving_profile_stats["environment_update_backend"] = getattr(
                self,
                "_last_environment_update_backend",
                "python_contract",
            )
            self._record_environment_update("update_right", time.perf_counter() - start)

    def _record_environment_update(self, phase, elapsed):
        updates = self.moving_profile_stats.setdefault("environment_updates", {})
        entry = updates.setdefault(
            str(phase),
            {"calls": 0, "seconds": 0.0, "last_seconds": 0.0},
        )
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["seconds"] = float(entry.get("seconds", 0.0)) + float(elapsed)
        entry["last_seconds"] = float(elapsed)

    def flush_compiled_flat_matvec_profiles(self):
        for compiled in self._compiled_flat_matvec_cache.values():
            if compiled is not None:
                compiled.flush_profile()

    def profile_summary(self):
        self.flush_compiled_flat_matvec_profiles()
        self._sync_cpp_moving_environment_stats()
        cpp_contextual_owner_stats = {
            key[len("cpp_moving_environment_") :]: value
            for key, value in self.moving_profile_stats.items()
            if key.startswith("cpp_moving_environment_contextual_")
            or key.startswith(
                "cpp_moving_environment_planned_direct_family_"
            )
            or key.startswith("cpp_moving_environment_direct_route_plan_")
        }
        if self._operatorless_local_problem_active:
            paths = self._local_profile_stats.get("paths", {})
            dominant = None
            if paths:
                dominant = max(
                    paths.items(),
                    key=lambda item: float(item[1].get("seconds", 0.0)),
                )[0]
            summary = {
                "bond": self.bond,
                "matvec_calls": int(
                    self._local_profile_stats.get("matvec_calls", 0)
                ),
                "matvec_seconds": float(
                    self._local_profile_stats.get("matvec_seconds", 0.0)
                ),
                "dominant_path": dominant,
                "paths": dict(paths),
                "plan_builds": self._local_profile_stats.get("plan_builds", {}),
                "local_solver": self._local_profile_stats.get("local_solver", {}),
                "packed_local_davidson": self._local_profile_stats.get(
                    "packed_local_davidson",
                    {},
                ),
                "preconditioner": self._local_profile_stats.get(
                    "preconditioner",
                    {},
                ),
            }
        elif self._dense_operatorless_local_problem_active:
            paths = self._dense_local_profile_stats.get("paths", {})
            dominant = None
            if paths:
                dominant = max(
                    paths.items(),
                    key=lambda item: float(item[1].get("seconds", 0.0)),
                )[0]
            summary = {
                "bond": self.bond,
                "matvec_calls": int(
                    self._dense_local_profile_stats.get("matvec_calls", 0)
                ),
                "matvec_seconds": float(
                    self._dense_local_profile_stats.get("matvec_seconds", 0.0)
                ),
                "dominant_path": dominant,
                "paths": dict(paths),
                "local_solver": dict(
                    self._dense_local_profile_stats.get("local_solver", {})
                ),
                "cpp_dense_davidson": dict(
                    self._dense_local_profile_stats.get("cpp_dense_davidson", {})
                ),
                "operatorless": True,
            }
        else:
            summary = self.operator.profile_summary()
        summary["moving_environment"] = {
            "bond": self.bond,
            "local_operator_builds": int(
                self.moving_profile_stats.get("local_operator_builds", 0)
            ),
            "local_operator_reuses": int(
                self.moving_profile_stats.get("local_operator_reuses", 0)
            ),
            "solve_local_calls": int(
                self.moving_profile_stats.get("solve_local_calls", 0)
            ),
            "solve_local_accepts": int(
                self.moving_profile_stats.get("solve_local_accepts", 0)
            ),
            "solve_local_rejections": int(
                self.moving_profile_stats.get("solve_local_rejections", 0)
            ),
            "solve_local_seconds": float(
                self.moving_profile_stats.get("solve_local_seconds", 0.0)
            ),
            "solve_local_last_seconds": float(
                self.moving_profile_stats.get("solve_local_last_seconds", 0.0)
            ),
            "packed_local_setup_seconds": float(
                self.moving_profile_stats.get("packed_local_setup_seconds", 0.0)
            ),
            "packed_local_initial_pack_seconds": float(
                self.moving_profile_stats.get(
                    "packed_local_initial_pack_seconds",
                    0.0,
                )
            ),
            "packed_local_final_unpack_seconds": float(
                self.moving_profile_stats.get(
                    "packed_local_final_unpack_seconds",
                    0.0,
                )
            ),
            "cpp_davidson_total_seconds": float(
                self.moving_profile_stats.get("cpp_davidson_total_seconds", 0.0)
            ),
            "solve_local_rejected_reason": self.moving_profile_stats.get(
                "solve_local_rejected_reason"
            ),
            "renormalized_operator_table_builds": int(
                self.moving_profile_stats.get("renormalized_operator_table_builds", 0)
            ),
            "renormalized_operator_table_build_seconds": float(
                self.moving_profile_stats.get(
                    "renormalized_operator_table_build_seconds",
                    0.0,
                )
            ),
            "renormalized_operator_table_cache_hits": int(
                self.moving_profile_stats.get("renormalized_operator_table_cache_hits", 0)
            ),
            "renormalized_operator_table_refreshes": int(
                self.moving_profile_stats.get("renormalized_operator_table_refreshes", 0)
            ),
            "renormalized_operator_table_refresh_seconds": float(
                self.moving_profile_stats.get(
                    "renormalized_operator_table_refresh_seconds",
                    0.0,
                )
            ),
            "renormalized_operator_table_structural_cache_hits": int(
                self.moving_profile_stats.get(
                    "renormalized_operator_table_structural_cache_hits",
                    0,
                )
            ),
            "renormalized_operator_table_slot_reuses": int(
                self.moving_profile_stats.get(
                    "renormalized_operator_table_slot_reuses",
                    0,
                )
            ),
            "renormalized_operator_payload_collect_calls": int(
                self.moving_profile_stats.get(
                    "renormalized_operator_payload_collect_calls",
                    0,
                )
            ),
            "renormalized_operator_payload_collect_seconds": float(
                self.moving_profile_stats.get(
                    "renormalized_operator_payload_collect_seconds",
                    0.0,
                )
            ),
            "renormalized_operator_payload_collect_last_seconds": float(
                self.moving_profile_stats.get(
                    "renormalized_operator_payload_collect_last_seconds",
                    0.0,
                )
            ),
            "renormalized_operator_payload_collect_direct_seconds": float(
                self.moving_profile_stats.get(
                    "renormalized_operator_payload_collect_direct_seconds",
                    0.0,
                )
            ),
            "renormalized_operator_payload_collect_named_seconds": float(
                self.moving_profile_stats.get(
                    "renormalized_operator_payload_collect_named_seconds",
                    0.0,
                )
            ),
            "renormalized_operator_payload_collect_group_seconds": float(
                self.moving_profile_stats.get(
                    "renormalized_operator_payload_collect_group_seconds",
                    0.0,
                )
            ),
            "compiled_flat_matvec_builds": int(
                self.moving_profile_stats.get("compiled_flat_matvec_builds", 0)
            ),
            "compiled_flat_matvec_cache_hits": int(
                self.moving_profile_stats.get("compiled_flat_matvec_cache_hits", 0)
            ),
            "compiled_flat_matvec_calls": int(
                self.moving_profile_stats.get("compiled_flat_matvec_calls", 0)
            ),
            "compiled_flat_matvec_seconds": float(
                self.moving_profile_stats.get("compiled_flat_matvec_seconds", 0.0)
            ),
            "compact_plan_builds": int(
                self.moving_profile_stats.get("compact_plan_builds", 0)
            ),
            "compact_plan_build_seconds": float(
                self.moving_profile_stats.get("compact_plan_build_seconds", 0.0)
            ),
            "compact_plan_cache_hits": int(
                self.moving_profile_stats.get("compact_plan_cache_hits", 0)
            ),
            "cpp_moving_environment_enabled": bool(
                self.moving_profile_stats.get("cpp_moving_environment_enabled", False)
            ),
            "cpp_moving_environment_contextual_owner": cpp_contextual_owner_stats,
            "cpp_moving_environment_compact_plan_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_records",
                    0,
                )
            ),
            "cpp_moving_environment_compact_plan_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_installs",
                    0,
                )
            ),
            "cpp_moving_environment_compact_plan_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_compact_plan_davidson_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_davidson_calls",
                    0,
                )
            ),
            "cpp_moving_environment_compact_plan_davidson_workspace_reuses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_davidson_workspace_reuses",
                    0,
                )
            ),
            "cpp_moving_environment_compact_plan_diagonal_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_diagonal_calls",
                    0,
                )
            ),
            "cpp_moving_environment_compact_plan_diagonal_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_compact_plan_diagonal_cache_hits",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_records",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_installs",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_davidson_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_davidson_calls",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_davidson_workspace_reuses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_davidson_workspace_reuses",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_matvec_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_matvec_calls",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_diagonal_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_diagonal_calls",
                    0,
                )
            ),
            "cpp_moving_environment_grouped_table_diagonal_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_grouped_table_diagonal_cache_hits",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_flat_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_calls",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_flat_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_failures",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_flat_blocks": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_blocks",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_flat_sectors": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_sectors",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_flat_rows": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_rows",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_flat_cols": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_cols",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_flat_dim": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_dim",
                    0,
                )
            ),
            "cpp_moving_environment_site_split_backend": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_backend"
                )
            ),
            "cpp_moving_environment_site_split_flat_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_split_flat_last_error"
                )
            ),
            "cpp_moving_environment_site_update_flat_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_update_flat_calls",
                    0,
                )
            ),
            "cpp_moving_environment_site_update_flat_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_update_flat_failures",
                    0,
                )
            ),
            "cpp_moving_environment_site_update_flat_left_blocks": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_update_flat_left_blocks",
                    0,
                )
            ),
            "cpp_moving_environment_site_update_flat_right_blocks": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_update_flat_right_blocks",
                    0,
                )
            ),
            "cpp_moving_environment_site_update_flat_dim": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_update_flat_dim",
                    0,
                )
            ),
            "cpp_moving_environment_site_update_backend": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_update_backend"
                )
            ),
            "cpp_moving_environment_site_update_flat_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_site_update_flat_last_error"
                )
            ),
            "cpp_moving_environment_solve_update_flat_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_solve_update_flat_calls",
                    0,
                )
            ),
            "cpp_moving_environment_solve_update_flat_accepted": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_solve_update_flat_accepted",
                    0,
                )
            ),
            "cpp_moving_environment_solve_update_flat_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_solve_update_flat_failures",
                    0,
                )
            ),
            "cpp_moving_environment_solve_update_auto_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_solve_update_auto_calls",
                    0,
                )
            ),
            "cpp_moving_environment_solve_update_backend": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_solve_update_backend"
                )
            ),
            "cpp_moving_environment_solve_update_flat_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_solve_update_flat_last_error"
                )
            ),
            "cpp_solve_update_seconds": float(
                self.moving_profile_stats.get("cpp_solve_update_seconds", 0.0)
            ),
            "cpp_solve_update_last_seconds": float(
                self.moving_profile_stats.get("cpp_solve_update_last_seconds", 0.0)
            ),
            "cpp_moving_environment_sweep_cursor_plan_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_plan_calls",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_lr_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_lr_calls",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_rl_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_rl_calls",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_recenter_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_recenter_calls",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_steps": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_steps",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_last_n_sites": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_last_n_sites",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_last_steps": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_last_steps",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_failures",
                    0,
                )
            ),
            "cpp_moving_environment_sweep_cursor_backend": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_backend"
                )
            ),
            "cpp_moving_environment_sweep_cursor_last_direction": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_last_direction"
                )
            ),
            "cpp_moving_environment_sweep_cursor_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_cursor_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_payload_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_records",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_installs",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_hits",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_misses",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_clears": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_clears",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_cleared_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_cleared_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_records",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_installs",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_prepare_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_prepare_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_builds": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_builds",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_cache_hits",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_misses",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_clears": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_clears",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_cleared_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_cleared_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_last_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_last_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_build_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_build_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_last_build_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_last_build_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_builder_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_builder_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_builds": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_builds",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_families": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_families",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_merges": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_merges",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_empty_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_empty_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_direct_family_payload_assembler_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_payload_assembler_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_builds": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_builds",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_families": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_families",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_empty_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_empty_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_direct_family_piece_builder_plan_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_piece_builder_plan_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_records",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_installs",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_prepare_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_prepare_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_cache_hits",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_misses",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_piece_plan_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_piece_plan_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_records",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_installs",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_prepare_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_prepare_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_cache_hits",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_misses",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_dispatch_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_families": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_dispatch_families",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_dispatch_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_dispatch_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_dispatch_empty_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_dispatch_empty_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_phased_family_plan_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_phased_family_plan_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_records",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_installs",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_prepare_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_prepare_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_cache_hits",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_misses",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_families": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_families",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_empty_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_dispatch_empty_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_factory_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_factory_calls",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_installs",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_uses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_static_plan_uses",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_families": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_families",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_entries": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_entries",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_empty_pieces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_literal_empty_pieces",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_calls",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_failures",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_rows": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_rows",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_terms": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_terms",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_scanned": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_scanned",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_skipped_consumed": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_skipped_consumed",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_skipped_zero": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_skipped_zero",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_select_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_select_last_error"
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_calls",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_successes": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_successes",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_unsupported": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_unsupported",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_failures",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_records",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_terms": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_terms",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_rows": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_rows",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_row_map_builds": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_identity_"
                        "info_row_map_builds"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_row_map_hits": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_identity_"
                        "info_row_map_hits"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_info_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_last_seconds": float(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_identity_"
                        "info_last_seconds"
                    ),
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_last_reason": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_identity_"
                        "info_last_reason"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_identity_info_last_error": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_identity_"
                        "info_last_error"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_boundary_batch_calls",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_successes": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_batch_successes"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_failures": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_batch_failures"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_keys": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_boundary_batch_keys",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_boundary_batch_hits",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_boundary_batch_misses",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_complete": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_boundary_batch_complete",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_boundary_batch_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_last_seconds": float(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_batch_last_seconds"
                    ),
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_batch_last_error": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_batch_last_error"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_calls": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_calls"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_successes": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_successes"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_failures": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_failures"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_rows": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_rows"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_unique": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_unique"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_route_layout": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_route_layout"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_fallback": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_fallback"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_seconds": float(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_seconds"
                    ),
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_plan_last_error": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_plan_last_error"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_calls": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_calls"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_successes": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_successes"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_failures": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_failures"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_rows": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_rows"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_available": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_available"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_missing": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_missing"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_hits": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_hits"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_misses": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_misses"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_seconds": float(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_seconds"
                    ),
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_value_last_error": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_value_last_error"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_calls": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_calls"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_successes": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_successes"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_failures": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_failures"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_rows": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_rows"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_advanced": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_advanced"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_remaining": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_remaining"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_cache_hits": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_cache_hits"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_cache_builds": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_cache_builds"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_none": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_none"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_seconds": float(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_seconds"
                    ),
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_boundary_parent_advance_last_error": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "boundary_parent_advance_last_error"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_missing_parent_build_plan_calls": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "missing_parent_build_plan_calls"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_missing_parent_build_plan_successes": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "missing_parent_build_plan_successes"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_missing_parent_build_plan_failures": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "missing_parent_build_plan_failures"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_missing_parent_build_plan_rows": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "missing_parent_build_plan_rows"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_missing_parent_build_plan_unique": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "missing_parent_build_plan_unique"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_missing_parent_build_plan_seconds": float(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "missing_parent_build_plan_seconds"
                    ),
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_missing_parent_build_plan_last_error": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "missing_parent_build_plan_last_error"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_calls": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_calls"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_successes": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_successes"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_failures": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_failures"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_rows": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_rows"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_available": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_available"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_missing": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_missing"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_puts": int(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_puts"
                    ),
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_seconds": float(
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_seconds"
                    ),
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_built_parent_advance_plan_last_error": (
                self.moving_profile_stats.get(
                    (
                        "cpp_moving_environment_same_side_route_"
                        "built_parent_advance_plan_last_error"
                    )
                )
            ),
            "cpp_moving_environment_same_side_route_identity_entry_build_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_entry_build_calls",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_entry_build_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_entry_build_failures",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_entry_build_rows": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_entry_build_rows",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_entry_build_terms": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_entry_build_terms",
                    0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_entry_build_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_entry_build_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_entry_build_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_entry_build_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_same_side_route_identity_entry_build_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_same_side_route_identity_entry_build_last_error"
                )
            ),
            "cpp_moving_environment_direct_family_two_phase_dispatch_plan_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_two_phase_dispatch_plan_last_error"
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_accepted": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_accepted",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_payload_prepares": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_payload_prepares",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_environment_moves": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_environment_moves",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_environment_fallbacks": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_environment_fallbacks",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_payload_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_payload_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_payload_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_payload_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_runner_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_runner_last_error"
                )
            ),
            "cpp_moving_environment_owner_bond_step_record_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_record_records",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_record_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_record_installs",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_record_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_record_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_record_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_record_hits",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_record_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_record_misses",
                    0,
                )
            ),
            "cpp_moving_environment_owner_bond_step_record_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_bond_step_record_last_error"
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_record_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_record_records",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_record_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_record_installs",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_record_replacements": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_record_replacements",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_record_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_record_hits",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_record_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_record_misses",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_environment_record_prepares": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_environment_record_prepares",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_environment_record_consumes": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_environment_record_consumes",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_python_prepare_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_python_prepare_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_python_move_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_python_move_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_record_installs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_record_installs",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_accepts": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_accepts",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_empty": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_empty",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_plan_provider_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_updates": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_updates",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_update_misses": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_update_misses",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_update_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_update_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_accepts": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_accepts",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_empty": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_empty",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_provider_refresh_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_accepts": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_accepts",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_empty": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_empty",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_refresh_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_accepts": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_accepts",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_links": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_links",
                    0,
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_direct_key_successor_chain_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_revision_state_updates": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_revision_state_updates",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_revision_state_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_revision_state_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_revision_cache_key_builds": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_revision_cache_key_builds",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_revision_cache_key_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_revision_cache_key_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_cpp_key_bundle_builds": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_cpp_key_bundle_builds",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_cpp_key_bundle_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_cpp_key_bundle_failures",
                    0,
                )
            ),
            "cpp_moving_environment_direct_family_revision_state_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_direct_family_revision_state_last_error"
                )
            ),
            "cpp_moving_environment_owner_typed_bond_step_record_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_typed_bond_step_record_last_error"
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_accepted": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_accepted",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_rejections": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_rejections",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_last_error"
                )
            ),
            "cpp_moving_environment_owner_local_optimize_runner_last_reason": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_optimize_runner_last_reason"
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_accepted": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_accepted",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_rejections": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_rejections",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_last_error"
                )
            ),
            "cpp_moving_environment_owner_local_grouped_solve_update_last_reason": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_local_grouped_solve_update_last_reason"
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_calls",
                    0,
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_accepted": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_accepted",
                    0,
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_failures",
                    0,
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_bonds": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_bonds",
                    0,
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_last_seconds",
                    0.0,
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_last_direction": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_last_direction"
                )
            ),
            "cpp_moving_environment_owner_half_sweep_runner_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_owner_half_sweep_runner_last_error"
                )
            ),
            "cpp_moving_environment_environment_plan_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_plan_records",
                    0,
                )
            ),
            "cpp_moving_environment_environment_plan_builds": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_plan_builds",
                    0,
                )
            ),
            "cpp_moving_environment_environment_plan_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_plan_cache_hits",
                    0,
                )
            ),
            "cpp_moving_environment_environment_plan_advance_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_plan_advance_calls",
                    0,
                )
            ),
            "cpp_moving_environment_environment_plan_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_plan_failures",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_records": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_records",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_resets": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_resets",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_pushes": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_pushes",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_pops": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_pops",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_apply_calls": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_apply_calls",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_apply_syncs": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_apply_syncs",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_apply_pushes": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_apply_pushes",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_apply_pops": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_apply_pops",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_apply_replaces": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_apply_replaces",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_apply_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_apply_failures",
                    0,
                )
            ),
            "cpp_moving_environment_environment_stack_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_failures",
                    0,
                )
            ),
            "cpp_environment_update_calls": int(
                self.moving_profile_stats.get("cpp_environment_update_calls", 0)
            ),
            "cpp_environment_update_left_calls": int(
                self.moving_profile_stats.get("cpp_environment_update_left_calls", 0)
            ),
            "cpp_environment_update_right_calls": int(
                self.moving_profile_stats.get("cpp_environment_update_right_calls", 0)
            ),
            "cpp_environment_update_seconds": float(
                self.moving_profile_stats.get("cpp_environment_update_seconds", 0.0)
            ),
            "cpp_environment_update_failures": int(
                self.moving_profile_stats.get("cpp_environment_update_failures", 0)
            ),
            "cpp_environment_update_last_error": self.moving_profile_stats.get(
                "cpp_environment_update_last_error"
            ),
            "cpp_environment_update_backend_actual": self.moving_profile_stats.get(
                "cpp_environment_update_backend_actual"
            ),
            "cpp_environment_plan_builds": int(
                self.moving_profile_stats.get("cpp_environment_plan_builds", 0)
            ),
            "cpp_environment_plan_build_seconds": float(
                self.moving_profile_stats.get("cpp_environment_plan_build_seconds", 0.0)
            ),
            "cpp_environment_plan_cache_hits": int(
                self.moving_profile_stats.get("cpp_environment_plan_cache_hits", 0)
            ),
            "cpp_environment_plan_advance_calls": int(
                self.moving_profile_stats.get("cpp_environment_plan_advance_calls", 0)
            ),
            "cpp_environment_plan_advance_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_environment_plan_advance_seconds",
                    0.0,
                )
            ),
            "cpp_environment_plan_failures": int(
                self.moving_profile_stats.get("cpp_environment_plan_failures", 0)
            ),
            "cpp_environment_plan_owner_failures": int(
                self.moving_profile_stats.get(
                    "cpp_environment_plan_owner_failures",
                    0,
                )
            ),
            "cpp_environment_plan_owner_records": int(
                self.moving_profile_stats.get("cpp_environment_plan_owner_records", 0)
            ),
            "cpp_environment_plan_backend_actual": self.moving_profile_stats.get(
                "cpp_environment_plan_backend_actual"
            ),
            "cpp_environment_stack_backend_actual": self.moving_profile_stats.get(
                "cpp_environment_stack_backend_actual"
            ),
            "cpp_environment_stack_resets": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_resets",
                            0,
                        )
                    ),
                    int(self.moving_profile_stats.get("cpp_environment_stack_resets", 0)),
                )
            ),
            "cpp_environment_stack_pushes": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_pushes",
                            0,
                        )
                    ),
                    int(self.moving_profile_stats.get("cpp_environment_stack_pushes", 0)),
                )
            ),
            "cpp_environment_stack_pops": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_pops",
                            0,
                        )
                    ),
                    int(self.moving_profile_stats.get("cpp_environment_stack_pops", 0)),
                )
            ),
            "cpp_environment_stack_apply_calls": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_apply_calls",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_environment_stack_apply_calls",
                            0,
                        )
                    ),
                )
            ),
            "cpp_environment_stack_apply_syncs": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_apply_syncs",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_environment_stack_apply_syncs",
                            0,
                        )
                    ),
                )
            ),
            "cpp_environment_stack_apply_pushes": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_apply_pushes",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_environment_stack_apply_pushes",
                            0,
                        )
                    ),
                )
            ),
            "cpp_environment_stack_apply_pops": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_apply_pops",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_environment_stack_apply_pops",
                            0,
                        )
                    ),
                )
            ),
            "cpp_environment_stack_apply_replaces": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_apply_replaces",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_environment_stack_apply_replaces",
                            0,
                        )
                    ),
                )
            ),
            "cpp_environment_stack_apply_failures": int(
                self.moving_profile_stats.get(
                    "cpp_moving_environment_environment_stack_apply_failures",
                    0,
                )
            ),
            "cpp_environment_stack_failures": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_environment_stack_failures",
                            0,
                        )
                    ),
                    int(self.moving_profile_stats.get("cpp_environment_stack_failures", 0)),
                )
            ),
            "cpp_sweep_environment_step_calls": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_sweep_environment_step_calls",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_sweep_environment_step_calls",
                            0,
                        )
                    ),
                )
            ),
            "cpp_sweep_environment_step_updates": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_sweep_environment_step_updates",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_sweep_environment_step_updates",
                            0,
                        )
                    ),
                )
            ),
            "cpp_sweep_environment_step_pops": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_sweep_environment_step_pops",
                            0,
                        )
                    ),
                    int(self.moving_profile_stats.get("cpp_sweep_environment_step_pops", 0)),
                )
            ),
            "cpp_sweep_environment_step_syncs": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_sweep_environment_step_syncs",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_sweep_environment_step_syncs",
                            0,
                        )
                    ),
                )
            ),
            "cpp_sweep_environment_step_auto_calls": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_sweep_environment_step_auto_calls",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_sweep_environment_step_auto_calls",
                            0,
                        )
                    ),
                )
            ),
            "owner_bond_step_calls": int(
                self.moving_profile_stats.get("owner_bond_step_calls", 0)
            ),
            "owner_bond_step_accepts": int(
                self.moving_profile_stats.get("owner_bond_step_accepts", 0)
            ),
            "owner_bond_step_failures": int(
                self.moving_profile_stats.get("owner_bond_step_failures", 0)
            ),
            "owner_bond_step_environment_moves": int(
                self.moving_profile_stats.get("owner_bond_step_environment_moves", 0)
            ),
            "owner_bond_step_environment_fallbacks": int(
                self.moving_profile_stats.get(
                    "owner_bond_step_environment_fallbacks",
                    0,
                )
            ),
            "owner_bond_step_payload_prepares": int(
                self.moving_profile_stats.get(
                    "owner_bond_step_payload_prepares",
                    0,
                )
            ),
            "owner_bond_step_payload_prepare_seconds": float(
                self.moving_profile_stats.get(
                    "owner_bond_step_payload_prepare_seconds",
                    0.0,
                )
            ),
            "owner_bond_step_payload_prepare_last_seconds": float(
                self.moving_profile_stats.get(
                    "owner_bond_step_payload_prepare_last_seconds",
                    0.0,
                )
            ),
            "owner_bond_step_seconds": float(
                self.moving_profile_stats.get("owner_bond_step_seconds", 0.0)
            ),
            "owner_bond_step_last_seconds": float(
                self.moving_profile_stats.get("owner_bond_step_last_seconds", 0.0)
            ),
            "owner_bond_step_backend_actual": self.moving_profile_stats.get(
                "owner_bond_step_backend_actual"
            ),
            "owner_bond_step_orchestrator_actual": self.moving_profile_stats.get(
                "owner_bond_step_orchestrator_actual"
            ),
            "owner_bond_step_last_error": self.moving_profile_stats.get(
                "owner_bond_step_last_error"
            ),
            "owner_local_optimize_calls": int(
                self.moving_profile_stats.get("owner_local_optimize_calls", 0)
            ),
            "owner_local_optimize_accepts": int(
                self.moving_profile_stats.get("owner_local_optimize_accepts", 0)
            ),
            "owner_local_optimize_rejections": int(
                self.moving_profile_stats.get("owner_local_optimize_rejections", 0)
            ),
            "owner_local_optimize_failures": int(
                self.moving_profile_stats.get("owner_local_optimize_failures", 0)
            ),
            "owner_local_optimize_seconds": float(
                self.moving_profile_stats.get("owner_local_optimize_seconds", 0.0)
            ),
            "owner_local_optimize_last_seconds": float(
                self.moving_profile_stats.get(
                    "owner_local_optimize_last_seconds",
                    0.0,
                )
            ),
            "owner_local_optimize_backend_actual": self.moving_profile_stats.get(
                "owner_local_optimize_backend_actual"
            ),
            "owner_local_optimize_rejected_reason": self.moving_profile_stats.get(
                "owner_local_optimize_rejected_reason"
            ),
            "owner_local_optimize_last_error": self.moving_profile_stats.get(
                "owner_local_optimize_last_error"
            ),
            "owner_local_optimize_site_commits": int(
                self.moving_profile_stats.get("owner_local_optimize_site_commits", 0)
            ),
            "owner_local_optimize_guess_cache_sets": int(
                self.moving_profile_stats.get(
                    "owner_local_optimize_guess_cache_sets",
                    0,
                )
            ),
            "owner_local_optimize_direct_cache_invalidations": int(
                self.moving_profile_stats.get(
                    "owner_local_optimize_direct_cache_invalidations",
                    0,
                )
            ),
            "owner_local_optimize_direct_payload_key_hits": int(
                self.moving_profile_stats.get(
                    "owner_local_optimize_direct_payload_key_hits",
                    0,
                )
            ),
            "owner_local_optimize_commit_actual": self.moving_profile_stats.get(
                "owner_local_optimize_commit_actual"
            ),
            "owner_local_optimize_solve_actual": self.moving_profile_stats.get(
                "owner_local_optimize_solve_actual"
            ),
            "owner_local_optimize_update_payload_actual": (
                self.moving_profile_stats.get(
                    "owner_local_optimize_update_payload_actual"
                )
            ),
            "owner_local_grouped_solve_update_calls": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_calls",
                    0,
                )
            ),
            "owner_local_grouped_solve_update_accepts": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_accepts",
                    0,
                )
            ),
            "owner_local_grouped_solve_update_rejections": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_rejections",
                    0,
                )
            ),
            "owner_local_grouped_solve_update_failures": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_failures",
                    0,
                )
            ),
            "owner_local_grouped_solve_update_seconds": float(
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_seconds",
                    0.0,
                )
            ),
            "owner_local_grouped_solve_update_last_seconds": float(
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_last_seconds",
                    0.0,
                )
            ),
            "owner_local_grouped_solve_update_backend_actual": (
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_backend_actual"
                )
            ),
            "owner_local_grouped_solve_update_rejected_reason": (
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_rejected_reason"
                )
            ),
            "owner_local_grouped_solve_update_last_error": (
                self.moving_profile_stats.get(
                    "owner_local_grouped_solve_update_last_error"
                )
            ),
            "owner_local_grouped_direct_prepare_calls": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_prepare_calls",
                    0,
                )
            ),
            "owner_local_grouped_direct_prepare_accepts": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_prepare_accepts",
                    0,
                )
            ),
            "owner_local_grouped_direct_prepare_failures": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_prepare_failures",
                    0,
                )
            ),
            "owner_local_grouped_direct_solve_update_calls": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_solve_update_calls",
                    0,
                )
            ),
            "owner_local_grouped_direct_solve_update_accepts": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_solve_update_accepts",
                    0,
                )
            ),
            "owner_local_grouped_direct_raw_update_accepts": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_raw_update_accepts",
                    0,
                )
            ),
            "owner_local_grouped_direct_solve_update_failures": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_solve_update_failures",
                    0,
                )
            ),
            "owner_local_grouped_direct_solve_update_fallbacks": int(
                self.moving_profile_stats.get(
                    "owner_local_grouped_direct_solve_update_fallbacks",
                    0,
                )
            ),
            "owner_half_sweep_calls": int(
                self.moving_profile_stats.get("owner_half_sweep_calls", 0)
            ),
            "owner_half_sweep_accepts": int(
                self.moving_profile_stats.get("owner_half_sweep_accepts", 0)
            ),
            "owner_half_sweep_failures": int(
                self.moving_profile_stats.get("owner_half_sweep_failures", 0)
            ),
            "owner_half_sweep_bonds": int(
                self.moving_profile_stats.get("owner_half_sweep_bonds", 0)
            ),
            "owner_half_sweep_seconds": float(
                self.moving_profile_stats.get("owner_half_sweep_seconds", 0.0)
            ),
            "owner_half_sweep_last_seconds": float(
                self.moving_profile_stats.get(
                    "owner_half_sweep_last_seconds",
                    0.0,
                )
            ),
            "owner_half_sweep_last_direction": self.moving_profile_stats.get(
                "owner_half_sweep_last_direction"
            ),
            "owner_half_sweep_backend_actual": self.moving_profile_stats.get(
                "owner_half_sweep_backend_actual"
            ),
            "owner_half_sweep_last_error": self.moving_profile_stats.get(
                "owner_half_sweep_last_error"
            ),
            "owner_direct_family_environment_calls": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_calls",
                    0,
                )
            ),
            "owner_direct_family_environment_builds": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_builds",
                    0,
                )
            ),
            "owner_direct_family_environment_cache_hits": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_cache_hits",
                    0,
                )
            ),
            "owner_direct_family_environment_cache_misses": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_cache_misses",
                    0,
                )
            ),
            "owner_direct_family_environment_entries": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_entries",
                    0,
                )
            ),
            "owner_direct_family_environment_seconds": float(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_seconds",
                    0.0,
                )
            ),
            "owner_direct_family_environment_last_seconds": float(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_last_seconds",
                    0.0,
                )
            ),
            "owner_direct_family_environment_cache_size": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_cache_size",
                    0,
                )
            ),
            "owner_direct_family_environment_last_bond": (
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_last_bond"
                )
            ),
            "owner_direct_family_environment_last_error": (
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_last_error"
                )
            ),
            "owner_direct_family_environment_prepared_payloads": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_prepared_payloads",
                    0,
                )
            ),
            "owner_direct_family_environment_prepared_hits": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_prepared_hits",
                    0,
                )
            ),
            "owner_direct_family_environment_prepared_misses": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_prepared_misses",
                    0,
                )
            ),
            "owner_direct_family_environment_prepared_cache_size": int(
                self.moving_profile_stats.get(
                    "owner_direct_family_environment_prepared_cache_size",
                    0,
                )
            ),
            "cpp_bond_step_transaction_calls": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_bond_step_transaction_calls",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_bond_step_transaction_attempts",
                            0,
                        )
                    ),
                )
            ),
            "cpp_bond_step_transaction_accepted": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_bond_step_transaction_accepted",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_bond_step_transaction_accepted",
                            0,
                        )
                    ),
                )
            ),
            "cpp_bond_step_transaction_failures": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_bond_step_transaction_failures",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_bond_step_transaction_failures",
                            0,
                        )
                    ),
                )
            ),
            "cpp_bond_step_transaction_environment_updates": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_bond_step_transaction_environment_updates",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_bond_step_transaction_environment_updates",
                            0,
                        )
                    ),
                )
            ),
            "cpp_bond_step_transaction_record_builds": int(
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_record_builds",
                    0,
                )
            ),
            "cpp_bond_step_transaction_record_prepares": int(
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_record_prepares",
                    0,
                )
            ),
            "cpp_bond_step_transaction_record_consumes": int(
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_record_consumes",
                    0,
                )
            ),
            "cpp_bond_step_transaction_commits": int(
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_commits",
                    0,
                )
            ),
            "cpp_bond_step_transaction_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_seconds",
                    0.0,
                )
            ),
            "cpp_bond_step_transaction_backend_actual": (
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_backend_actual"
                )
            ),
            "cpp_bond_step_transaction_commit_backend_actual": (
                self.moving_profile_stats.get(
                    "cpp_bond_step_transaction_commit_backend_actual"
                )
            ),
            "cpp_bond_step_transaction_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_bond_step_transaction_last_error",
                    self.moving_profile_stats.get(
                        "cpp_bond_step_transaction_last_error"
                    ),
                )
            ),
            "cpp_sweep_environment_step_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_sweep_environment_step_seconds",
                    0.0,
                )
            ),
            "cpp_sweep_environment_step_failures": int(
                max(
                    int(
                        self.moving_profile_stats.get(
                            "cpp_moving_environment_sweep_environment_step_failures",
                            0,
                        )
                    ),
                    int(
                        self.moving_profile_stats.get(
                            "cpp_sweep_environment_step_failures",
                            0,
                        )
                    ),
                )
            ),
            "cpp_sweep_environment_step_backend_actual": (
                self.moving_profile_stats.get(
                    "cpp_sweep_environment_step_backend_actual"
                )
            ),
            "cpp_sweep_environment_step_last_error": (
                self.moving_profile_stats.get(
                    "cpp_moving_environment_sweep_environment_step_last_error",
                    self.moving_profile_stats.get(
                        "cpp_sweep_environment_step_last_error"
                    ),
                )
            ),
            "cpp_environment_plan_last_routes": int(
                self.moving_profile_stats.get("cpp_environment_plan_last_routes", 0)
            ),
            "cpp_environment_plan_last_blocks": int(
                self.moving_profile_stats.get("cpp_environment_plan_last_blocks", 0)
            ),
            "cpp_environment_plan_last_error": self.moving_profile_stats.get(
                "cpp_environment_plan_last_error"
            ),
            "abelian_environment_advance_payloads": (
                abelian_environment_advance_payload_stats()
            ),
            "compact_plan_bond_slot_stores": int(
                self.moving_profile_stats.get("compact_plan_bond_slot_stores", 0)
            ),
            "compact_plan_bond_slot_hits": int(
                self.moving_profile_stats.get("compact_plan_bond_slot_hits", 0)
            ),
            "compact_plan_bond_slot_refreshes": int(
                self.moving_profile_stats.get(
                    "compact_plan_bond_slot_refreshes",
                    0,
                )
            ),
            "compact_plan_bond_slot_refresh_failures": int(
                self.moving_profile_stats.get(
                    "compact_plan_bond_slot_refresh_failures",
                    0,
                )
            ),
            "compact_plan_bond_slot_last_refresh_error": self.moving_profile_stats.get(
                "compact_plan_bond_slot_last_refresh_error"
            ),
            "compact_plan_refreshes": int(
                self.moving_profile_stats.get("compact_plan_refreshes", 0)
            ),
            "compact_plan_refresh_seconds": float(
                self.moving_profile_stats.get("compact_plan_refresh_seconds", 0.0)
            ),
            "compact_plan_refresh_failures": int(
                self.moving_profile_stats.get("compact_plan_refresh_failures", 0)
            ),
            "compact_plan_last_refresh_seconds": float(
                self.moving_profile_stats.get("compact_plan_last_refresh_seconds", 0.0)
            ),
            "compact_plan_last_refresh_error": self.moving_profile_stats.get(
                "compact_plan_last_refresh_error"
            ),
            "compact_plan_failures": int(
                self.moving_profile_stats.get("compact_plan_failures", 0)
            ),
            "compact_renormalized_table_builds": int(
                self.moving_profile_stats.get("compact_renormalized_table_builds", 0)
            ),
            "compact_renormalized_table_build_seconds": float(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_build_seconds",
                    0.0,
                )
            ),
            "compact_renormalized_table_cache_hits": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_cache_hits",
                    0,
                )
            ),
            "compact_renormalized_table_bond_slot_stores": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_bond_slot_stores",
                    0,
                )
            ),
            "compact_renormalized_table_bond_slot_hits": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_bond_slot_hits",
                    0,
                )
            ),
            "compact_renormalized_table_bond_slot_reuses": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_bond_slot_reuses",
                    0,
                )
            ),
            "compact_renormalized_table_build_backend": self.moving_profile_stats.get(
                "compact_renormalized_table_build_backend"
            ),
            "compact_renormalized_table_cpp_block_constructor_builds": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_cpp_block_constructor_builds",
                    0,
                )
            ),
            "compact_renormalized_table_python_stack_constructor_builds": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_python_stack_constructor_builds",
                    0,
                )
            ),
            "compact_renormalized_table_refreshes": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_refreshes",
                    0,
                )
            ),
            "compact_renormalized_table_refresh_seconds": float(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_refresh_seconds",
                    0.0,
                )
            ),
            "compact_renormalized_table_refresh_failures": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_refresh_failures",
                    0,
                )
            ),
            "compact_renormalized_table_cpp_block_refreshes": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_cpp_block_refreshes",
                    0,
                )
            ),
            "compact_renormalized_table_python_stack_refreshes": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_python_stack_refreshes",
                    0,
                )
            ),
            "compact_renormalized_table_last_refresh_backend": self.moving_profile_stats.get(
                "compact_renormalized_table_last_refresh_backend"
            ),
            "compact_renormalized_table_last_refresh_seconds": float(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_last_refresh_seconds",
                    0.0,
                )
            ),
            "compact_renormalized_table_last_refresh_error": self.moving_profile_stats.get(
                "compact_renormalized_table_last_refresh_error"
            ),
            "compact_renormalized_table_failures": int(
                self.moving_profile_stats.get("compact_renormalized_table_failures", 0)
            ),
            "compact_renormalized_table_last_error": self.moving_profile_stats.get(
                "compact_renormalized_table_last_error"
            ),
            "compact_renormalized_table_last_dimension": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_last_dimension",
                    0,
                )
            ),
            "compact_renormalized_table_last_entries": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_last_entries",
                    0,
                )
            ),
            "compact_renormalized_table_last_groups": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_last_groups",
                    0,
                )
            ),
            "compact_renormalized_table_last_diagonal_routes": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_last_diagonal_routes",
                    0,
                )
            ),
            "compact_renormalized_table_diagonal_calls": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_diagonal_calls",
                    0,
                )
            ),
            "compact_renormalized_table_diagonal_seconds": float(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_diagonal_seconds",
                    0.0,
                )
            ),
            "compact_renormalized_table_last_diagonal_seconds": float(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_last_diagonal_seconds",
                    0.0,
                )
            ),
            "compact_renormalized_table_diagonal_fallbacks": int(
                self.moving_profile_stats.get(
                    "compact_renormalized_table_diagonal_fallbacks",
                    0,
                )
            ),
            "compact_renormalized_table_diagonal_backend": self.moving_profile_stats.get(
                "compact_renormalized_table_diagonal_backend"
            ),
            "compact_renormalized_table_last_diagonal_error": self.moving_profile_stats.get(
                "compact_renormalized_table_last_diagonal_error"
            ),
            "compact_plan_matvec_calls": int(
                self.moving_profile_stats.get("compact_plan_matvec_calls", 0)
            ),
            "compact_plan_matvec_seconds": float(
                self.moving_profile_stats.get("compact_plan_matvec_seconds", 0.0)
            ),
            "compact_plan_matvec_last_seconds": float(
                self.moving_profile_stats.get("compact_plan_matvec_last_seconds", 0.0)
            ),
            "compact_plan_validation_calls": int(
                self.moving_profile_stats.get("compact_plan_validation_calls", 0)
            ),
            "compact_plan_validation_cache_hits": int(
                self.moving_profile_stats.get("compact_plan_validation_cache_hits", 0)
            ),
            "compact_plan_validation_failures": int(
                self.moving_profile_stats.get("compact_plan_validation_failures", 0)
            ),
            "compact_plan_validation_last_error_norm": float(
                self.moving_profile_stats.get(
                    "compact_plan_validation_last_error_norm",
                    0.0,
                )
            ),
            "compact_plan_validation_last_relative_error": float(
                self.moving_profile_stats.get(
                    "compact_plan_validation_last_relative_error",
                    0.0,
                )
            ),
            "compact_plan_last_error": self.moving_profile_stats.get(
                "compact_plan_last_error"
            ),
            "compact_plan_validation_last_error": self.moving_profile_stats.get(
                "compact_plan_validation_last_error"
            ),
            "compact_plan_last_dimension": int(
                self.moving_profile_stats.get("compact_plan_last_dimension", 0)
            ),
            "compact_plan_last_entries": int(
                self.moving_profile_stats.get("compact_plan_last_entries", 0)
            ),
            "compact_plan_last_groups": int(
                self.moving_profile_stats.get("compact_plan_last_groups", 0)
            ),
            "cpp_block_table_builds": int(
                self.moving_profile_stats.get("cpp_block_table_builds", 0)
            ),
            "compact_block_table_builds": int(
                self.moving_profile_stats.get("compact_block_table_builds", 0)
            ),
            "compact_block_table_build_seconds": float(
                self.moving_profile_stats.get("compact_block_table_build_seconds", 0.0)
            ),
            "compact_block_table_cache_hits": int(
                self.moving_profile_stats.get("compact_block_table_cache_hits", 0)
            ),
            "compact_block_table_failures": int(
                self.moving_profile_stats.get("compact_block_table_failures", 0)
            ),
            "compact_block_table_validation_calls": int(
                self.moving_profile_stats.get("compact_block_table_validation_calls", 0)
            ),
            "compact_block_table_validation_failures": int(
                self.moving_profile_stats.get(
                    "compact_block_table_validation_failures",
                    0,
                )
            ),
            "compact_block_table_validation_last_error_norm": float(
                self.moving_profile_stats.get(
                    "compact_block_table_validation_last_error_norm",
                    0.0,
                )
            ),
            "compact_block_table_validation_last_relative_error": float(
                self.moving_profile_stats.get(
                    "compact_block_table_validation_last_relative_error",
                    0.0,
                )
            ),
            "compact_block_table_last_error": self.moving_profile_stats.get(
                "compact_block_table_last_error"
            ),
            "compact_block_table_validation_last_error": self.moving_profile_stats.get(
                "compact_block_table_validation_last_error"
            ),
            "cpp_renormalized_table_builds": int(
                self.moving_profile_stats.get("cpp_renormalized_table_builds", 0)
            ),
            "cpp_grouped_renormalized_table_builds": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_builds",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_build_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_build_seconds",
                    0.0,
                )
            ),
            "cpp_grouped_renormalized_table_failures": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_failures",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_refreshes": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_refreshes",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_refresh_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_refresh_seconds",
                    0.0,
                )
            ),
            "cpp_grouped_renormalized_table_refresh_failures": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_refresh_failures",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_slot_reuses": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_slot_reuses",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_fast_refreshes": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_fast_refreshes",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_rebuild_refreshes": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_rebuild_refreshes",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_rebuild_in_place_refreshes": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_rebuild_in_place_refreshes",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_bond_slot_reuses": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_bond_slot_reuses",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_structural_slot_reuses": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_structural_slot_reuses",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_last_refresh_kind": (
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_last_refresh_kind"
                )
            ),
            "cpp_grouped_renormalized_table_refresh_last_error": (
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_refresh_last_error"
                )
            ),
            "cpp_grouped_renormalized_table_last_error": self.moving_profile_stats.get(
                "cpp_grouped_renormalized_table_last_error"
            ),
            "cpp_grouped_renormalized_table_last_storage": self.moving_profile_stats.get(
                "cpp_grouped_renormalized_table_last_storage"
            ),
            "cpp_grouped_renormalized_table_last_blocks": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_last_blocks",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_last_elements": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_last_elements",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_last_sparse_nnz": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_last_sparse_nnz",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_index_cache_entries": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_index_cache_entries",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_index_cache_hits": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_index_cache_hits",
                    0,
                )
            ),
            "cpp_grouped_renormalized_table_index_cache_misses": int(
                self.moving_profile_stats.get(
                    "cpp_grouped_renormalized_table_index_cache_misses",
                    0,
                )
            ),
            "cpp_sparse_renormalized_table_builds": int(
                self.moving_profile_stats.get(
                    "cpp_sparse_renormalized_table_builds",
                    0,
                )
            ),
            "cpp_renormalized_table_storage": self.moving_profile_stats.get(
                "cpp_renormalized_table_storage"
            ),
            "cpp_renormalized_table_failures": int(
                self.moving_profile_stats.get("cpp_renormalized_table_failures", 0)
            ),
            "cpp_renormalized_table_last_error": self.moving_profile_stats.get(
                "cpp_renormalized_table_last_error"
            ),
            "cpp_renormalized_table_validation_calls": int(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_validation_calls",
                    0,
                )
            ),
            "cpp_renormalized_table_validation_failures": int(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_validation_failures",
                    0,
                )
            ),
            "cpp_renormalized_table_validation_last_error_norm": float(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_validation_last_error_norm",
                    0.0,
                )
            ),
            "cpp_renormalized_table_validation_last_relative_error": float(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_validation_last_relative_error",
                    0.0,
                )
            ),
            "cpp_renormalized_table_validation_last_error": self.moving_profile_stats.get(
                "cpp_renormalized_table_validation_last_error"
            ),
            "cpp_renormalized_table_diagonal_calls": int(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_diagonal_calls",
                    0,
                )
            ),
            "cpp_renormalized_table_diagonal_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_diagonal_seconds",
                    0.0,
                )
            ),
            "cpp_renormalized_table_last_diagonal_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_last_diagonal_seconds",
                    0.0,
                )
            ),
            "cpp_renormalized_table_last_diagonal_error": self.moving_profile_stats.get(
                "cpp_renormalized_table_last_diagonal_error"
            ),
            "cpp_renormalized_table_matvec_calls": int(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_matvec_calls",
                    0,
                )
            ),
            "cpp_renormalized_table_matvec_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_matvec_seconds",
                    0.0,
                )
            ),
            "cpp_renormalized_table_matvec_last_seconds": float(
                self.moving_profile_stats.get(
                    "cpp_renormalized_table_matvec_last_seconds",
                    0.0,
                )
            ),
            "cpp_block_table_failures": int(
                self.moving_profile_stats.get("cpp_block_table_failures", 0)
            ),
            "cpp_block_table_validation_calls": int(
                self.moving_profile_stats.get("cpp_block_table_validation_calls", 0)
            ),
            "cpp_block_table_validation_failures": int(
                self.moving_profile_stats.get(
                    "cpp_block_table_validation_failures",
                    0,
                )
            ),
            "cpp_block_table_validation_last_error_norm": float(
                self.moving_profile_stats.get(
                    "cpp_block_table_validation_last_error_norm",
                    0.0,
                )
            ),
            "cpp_block_table_validation_last_relative_error": float(
                self.moving_profile_stats.get(
                    "cpp_block_table_validation_last_relative_error",
                    0.0,
                )
            ),
            "cpp_block_table_validation_last_error": self.moving_profile_stats.get(
                "cpp_block_table_validation_last_error"
            ),
            "cpp_block_table_last_error": self.moving_profile_stats.get(
                "cpp_block_table_last_error"
            ),
            "cpp_block_matvec_calls": int(
                self.moving_profile_stats.get("cpp_block_matvec_calls", 0)
            ),
            "cpp_block_matvec_seconds": float(
                self.moving_profile_stats.get("cpp_block_matvec_seconds", 0.0)
            ),
            "cpp_block_matvec_last_seconds": float(
                self.moving_profile_stats.get("cpp_block_matvec_last_seconds", 0.0)
            ),
            "cpp_block_matvec_failures": int(
                self.moving_profile_stats.get("cpp_block_matvec_failures", 0)
            ),
            "cpp_block_matvec_last_error": self.moving_profile_stats.get(
                "cpp_block_matvec_last_error"
            ),
            "cpp_davidson_calls": int(
                self.moving_profile_stats.get("cpp_davidson_calls", 0)
            ),
            "cpp_davidson_last_solver_calls": int(
                self.moving_profile_stats.get("cpp_davidson_last_solver_calls", 0)
            ),
            "cpp_davidson_workspace_reuses": int(
                self.moving_profile_stats.get("cpp_davidson_workspace_reuses", 0)
            ),
            "cpp_davidson_last_solver_workspace_reuses": int(
                self.moving_profile_stats.get(
                    "cpp_davidson_last_solver_workspace_reuses",
                    0,
                )
            ),
            "cpp_davidson_attempts": int(
                self.moving_profile_stats.get("cpp_davidson_attempts", 0)
            ),
            "cpp_davidson_failures": int(
                self.moving_profile_stats.get("cpp_davidson_failures", 0)
            ),
            "cpp_davidson_rejected": int(
                self.moving_profile_stats.get("cpp_davidson_rejected", 0)
            ),
            "cpp_davidson_seconds": float(
                self.moving_profile_stats.get("cpp_davidson_seconds", 0.0)
            ),
            "cpp_davidson_last_seconds": float(
                self.moving_profile_stats.get("cpp_davidson_last_seconds", 0.0)
            ),
            "cpp_davidson_last_error": self.moving_profile_stats.get(
                "cpp_davidson_last_error"
            ),
            "cpp_davidson_table_source": self.moving_profile_stats.get(
                "cpp_davidson_table_source"
            ),
            "cpp_davidson_last_residual": self.moving_profile_stats.get(
                "cpp_davidson_last_residual"
            ),
            "cpp_solution_validation_calls": int(
                self.moving_profile_stats.get("cpp_solution_validation_calls", 0)
            ),
            "cpp_solution_validation_failures": int(
                self.moving_profile_stats.get("cpp_solution_validation_failures", 0)
            ),
            "cpp_solution_validation_last_residual": self.moving_profile_stats.get(
                "cpp_solution_validation_last_residual"
            ),
            "cpp_solution_validation_last_energy": self.moving_profile_stats.get(
                "cpp_solution_validation_last_energy"
            ),
            "cpp_solution_validation_last_limit": self.moving_profile_stats.get(
                "cpp_solution_validation_last_limit"
            ),
            "environment_updates": self.moving_profile_stats.get("environment_updates", {}),
            "environment_update_backend": self.moving_profile_stats.get(
                "environment_update_backend"
            ),
            "environment_stack_updates": self.moving_profile_stats.get(
                "environment_stack_updates",
                {},
            ),
            "direct_family_cache_invalidations": int(
                self.moving_profile_stats.get(
                    "direct_family_cache_invalidations",
                    0,
                )
            ),
            "direct_family_cache_revision": int(
                self.moving_profile_stats.get(
                    "direct_family_cache_revision",
                    self.direct_family_revision,
                )
            ),
            "direct_family_cache_maps_cleared": int(
                self.moving_profile_stats.get(
                    "direct_family_cache_maps_cleared",
                    0,
                )
            ),
            "direct_family_boundary_revisions": self.moving_profile_stats.get(
                "direct_family_boundary_revisions",
                {},
            ),
            "sweep_stack_bindings": int(
                self.moving_profile_stats.get("sweep_stack_bindings", 0)
            ),
            "sweep_stack_left_bound": bool(
                self.moving_profile_stats.get("sweep_stack_left_bound", False)
            ),
            "sweep_stack_right_bound": bool(
                self.moving_profile_stats.get("sweep_stack_right_bound", False)
            ),
            "sweep_stack_family_count": int(
                self.moving_profile_stats.get("sweep_stack_family_count", 0)
            ),
        }
        for key in (
            "dense_local_operator_builds",
            "dense_local_operator_reuses",
            "dense_solve_local_calls",
            "dense_solve_local_accepts",
            "dense_solve_local_rejections",
            "dense_solve_local_seconds",
            "dense_solve_local_last_seconds",
            "dense_cpp_sweep_workspace_enabled",
            "dense_cpp_sweep_workspace_creates",
            "dense_cpp_sweep_workspace_records",
            "dense_cpp_sweep_workspace_binds",
            "dense_cpp_sweep_workspace_bind_seconds",
            "dense_cpp_sweep_workspace_boundary_binds",
            "dense_cpp_sweep_workspace_boundary_bind_seconds",
            "dense_cpp_sweep_workspace_static_w_hits",
            "dense_cpp_sweep_workspace_bind_cache_hits",
            "dense_cpp_sweep_workspace_solve_calls",
            "dense_cpp_sweep_workspace_solve_seconds",
            "dense_cpp_sweep_workspace_two_site_solve_calls",
            "dense_cpp_sweep_workspace_two_site_solve_accepts",
            "dense_cpp_sweep_workspace_two_site_solve_rejections",
            "dense_cpp_sweep_workspace_two_site_solve_seconds",
            "dense_cpp_sweep_workspace_two_site_static_w_reuses",
            "dense_cpp_sweep_workspace_two_site_mpo_builds",
            "dense_cpp_sweep_workspace_two_site_mps_builds",
            "dense_cpp_sweep_workspace_failures",
            "dense_cpp_sweep_workspace_last_error",
            "dense_cpp_tensor_primitive_calls",
            "dense_cpp_tensor_primitive_seconds",
            "dense_cpp_tensor_primitive_failures",
            "dense_cpp_tensor_primitive_last_error",
            "dense_cpp_coarse_grain_mpo_calls",
            "dense_cpp_coarse_grain_mpo_cache_hits",
            "dense_cpp_coarse_grain_mps_calls",
            "dense_cpp_environment_update_calls",
            "dense_cpp_environment_update_seconds",
            "dense_cpp_environment_update_failures",
            "dense_operatorless_local_problem_binds",
            "dense_operatorless_local_problem_solve_calls",
            "dense_operatorless_local_problem_solve_accepts",
            "dense_operatorless_local_problem_solve_rejections",
            "dense_operatorless_local_problem_solve_seconds",
            "dense_operatorless_local_problem_solve_last_seconds",
            "dense_operatorless_local_problem_last_error",
            "dense_cpp_split_calls",
            "dense_cpp_split_accepts",
            "dense_cpp_split_failures",
            "dense_cpp_split_seconds",
            "dense_cpp_split_last_seconds",
            "dense_cpp_split_last_error",
        ):
            summary["moving_environment"][key] = self.moving_profile_stats.get(
                key,
                False if key.endswith("_enabled") else 0,
            )
        return summary


_abelian_local_engine.MovingEnvironment = MovingEnvironment
