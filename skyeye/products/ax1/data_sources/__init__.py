"""AX1 data source registry."""

from __future__ import annotations

from skyeye.products.ax1.data_sources.base import AX1DataSource, DataSourceCapability, DataSourceRegistry
from skyeye.products.ax1.data_sources.flow import FlowDataSource
from skyeye.products.ax1.data_sources.fundamental import FundamentalDataSource
from skyeye.products.ax1.data_sources.macro import MacroDataSource
from skyeye.products.ax1.data_sources.price_volume import PriceVolumeDataSource
from skyeye.products.ax1.data_sources.technical import TechnicalIndicatorDataSource


def build_default_data_source_registry() -> DataSourceRegistry:
    sources: list[AX1DataSource] = [
        PriceVolumeDataSource(),
        FundamentalDataSource(),
        FlowDataSource(),
        MacroDataSource(),
        TechnicalIndicatorDataSource(),
    ]
    capabilities: list[DataSourceCapability] = []
    for source in sources:
        capabilities.extend(source.capabilities())
    return DataSourceRegistry(capabilities)


__all__ = [
    "AX1DataSource",
    "DataSourceCapability",
    "DataSourceRegistry",
    "PriceVolumeDataSource",
    "FundamentalDataSource",
    "FlowDataSource",
    "MacroDataSource",
    "TechnicalIndicatorDataSource",
    "build_default_data_source_registry",
]
