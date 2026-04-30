from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class LayerConfigError(ValueError):
    pass


class LegacyLayerConfigError(LayerConfigError):
    pass


class UnknownAllocationLayerError(LayerConfigError):
    pass


_LEGACY_UNIVERSE_KEYS = frozenset(
    {
        "core_etfs",
        "industry_etfs",
        "style_etfs",
        "stock_satellite",
    }
)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    asset_type: str
    include: tuple[str, ...] = ()
    enabled: bool = True
    max_count: int | None = None
    exposure_group: str = ""


def validate_no_legacy_layer_keys(config: Mapping[str, Any]) -> None:
    universe_config = _section(config, "universe")
    validate_no_legacy_universe_keys(universe_config)


def validate_no_legacy_universe_keys(universe_config: Mapping[str, Any]) -> None:
    legacy_keys = [key for key in _LEGACY_UNIVERSE_KEYS if key in universe_config]
    if legacy_keys:
        ordered_keys = sorted(legacy_keys)
        raise LegacyLayerConfigError(
            "AX1 layer registry does not support legacy universe keys: "
            + ", ".join(ordered_keys)
            + ". Use universe.layers.<layer_name> instead."
        )


class LayerRegistry:
    def __init__(self, layer_specs: Iterable[LayerSpec]):
        self._layers: dict[str, LayerSpec] = {}
        self._id_to_layer: dict[str, str] = {}
        self._default_by_asset_type: dict[str, str] = {}

        for spec in layer_specs:
            if spec.name in self._layers:
                raise LayerConfigError(f"Duplicate layer name: {spec.name}")
            self._layers[spec.name] = spec
            if not spec.enabled:
                continue

            asset_key = _normalize_asset_type(spec.asset_type)
            self._default_by_asset_type.setdefault(asset_key, spec.name)
            for order_book_id in spec.include:
                existing_layer = self._id_to_layer.get(order_book_id)
                if existing_layer is not None:
                    raise LayerConfigError(
                        f"order_book_id {order_book_id!r} appears in multiple layers: "
                        f"{existing_layer}, {spec.name}"
                    )
                self._id_to_layer[order_book_id] = spec.name

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LayerRegistry":
        validate_no_legacy_layer_keys(config)
        universe_layers = _layers_section(_section(config, "universe"))
        allocation = _section(config, "allocation")
        if "layers" in allocation:
            raise UnknownAllocationLayerError("allocation.layers is no longer supported; use allocation.exposure_groups")
        return cls.from_layer_sections(universe_layers)

    @classmethod
    def from_universe_config(cls, universe_config: Mapping[str, Any]) -> "LayerRegistry":
        validate_no_legacy_universe_keys(universe_config)
        universe_layers = _layers_section(universe_config)
        return cls.from_layer_sections(universe_layers)

    @classmethod
    def from_layer_sections(cls, universe_layers: Mapping[str, Mapping[str, Any]]) -> "LayerRegistry":
        specs = []
        for name, universe_spec in universe_layers.items():
            specs.append(_parse_universe_layer_spec(name, universe_spec))
        return cls(specs)

    def spec(self, layer_name: str) -> LayerSpec:
        return self._layers[layer_name]

    def layer_names(self) -> list[str]:
        return list(self._layers)

    def enabled_layer_names(self) -> list[str]:
        return [name for name, spec in self._layers.items() if spec.enabled]

    def layer_name_for_id(self, order_book_id: str) -> str | None:
        return self._id_to_layer.get(order_book_id)

    def default_layer_for_asset_type(self, asset_type: str) -> str | None:
        return self._default_by_asset_type.get(_normalize_asset_type(asset_type))

    def infer_layer_name(self, order_book_id: str, asset_type: str) -> str | None:
        return self.layer_name_for_id(order_book_id) or self.default_layer_for_asset_type(asset_type)

    def exposure_group_for_layer(self, layer_name: str) -> str:
        return self.spec(layer_name).exposure_group


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = config.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise LayerConfigError(f"{key} must be a mapping")
    return section


def _layers_section(config: Mapping[str, Any], *, required: bool = True) -> dict[str, Mapping[str, Any]]:
    layers = config.get("layers")
    if layers is None:
        if required:
            raise LayerConfigError("universe.layers is required")
        return {}
    if isinstance(layers, Mapping):
        return {
            str(name): _require_mapping(layer_config, f"layers.{name}")
            for name, layer_config in layers.items()
        }
    if isinstance(layers, list):
        parsed_layers = {}
        for index, layer_config in enumerate(layers):
            layer_mapping = _require_mapping(layer_config, f"layers[{index}]")
            name = layer_mapping.get("name")
            if not name:
                raise LayerConfigError(f"layers[{index}].name is required")
            parsed_layers[str(name)] = layer_mapping
        return parsed_layers
    raise LayerConfigError("layers must be a mapping or a list of layer specs")


def _parse_universe_layer_spec(
    layer_name: str,
    universe_spec: Mapping[str, Any],
) -> LayerSpec:
    explicit_name = universe_spec.get("name")
    if explicit_name is not None and explicit_name != layer_name:
        raise LayerConfigError(
            f"Layer key {layer_name!r} does not match layer spec name {explicit_name!r}"
        )

    if "asset_type" not in universe_spec:
        raise LayerConfigError(f"universe.layers.{layer_name}.asset_type is required")

    return LayerSpec(
        name=layer_name,
        asset_type=str(universe_spec["asset_type"]),
        include=tuple(str(order_book_id) for order_book_id in universe_spec.get("include", ())),
        enabled=bool(universe_spec.get("enabled", True)),
        max_count=_optional_int(universe_spec.get("max_count"), f"{layer_name}.max_count"),
        exposure_group=str(universe_spec.get("exposure_group") or layer_name),
    )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayerConfigError(f"{label} must be a mapping")
    return value


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    converted = int(value)
    if converted < 0:
        raise LayerConfigError(f"{label} must be non-negative")
    return converted


def _normalize_asset_type(asset_type: str) -> str:
    return str(asset_type).strip().lower()
