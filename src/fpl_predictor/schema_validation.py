"""Strict production-to-live feature schema parity checks."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SchemaReport:
    required: list[str]
    matching: list[str]
    missing: list[str]
    unexpected: list[str]
    dtype_mismatches: list[str]

    @property
    def passed(self) -> bool:
        return not self.missing and not self.dtype_mismatches


def audit_live_schema(frame: pd.DataFrame, required: list[str]) -> SchemaReport:
    missing = sorted(set(required) - set(frame.columns))
    matching = [name for name in required if name in frame]
    numeric_expected = [name for name in matching if name != "position"]
    mismatches = [name for name in numeric_expected if not pd.api.types.is_numeric_dtype(frame[name])]
    unexpected = sorted(set(frame.columns) - set(required))
    return SchemaReport(required, matching, missing, unexpected, mismatches)


def require_live_schema(frame: pd.DataFrame, required: list[str]) -> SchemaReport:
    report = audit_live_schema(frame, required)
    if not report.passed:
        raise ValueError(f"Live feature schema mismatch: missing={report.missing}, dtype_mismatches={report.dtype_mismatches}")
    return report
