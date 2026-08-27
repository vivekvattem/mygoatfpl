#!/usr/bin/env python3
"""Focused transfer-analysis entry point; uses the Phase 6 decision pipeline."""

from optimize_squad import parser, run


if __name__ == "__main__":
    arguments = parser().parse_args()
    summary = run(arguments)
    print("\nTRANSFER ANALYSIS COMPLETE")
    print(f"Decision: {summary['transfer_decision']}")
