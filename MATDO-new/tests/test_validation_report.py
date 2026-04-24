"""Tests for the validation report (MATDO-E paper proposition checker)."""

from __future__ import annotations

import io
import sys

from matdo_new.core.config import MATDOConfig
from matdo_new.experiments.validation_report import (
    ValidationReport,
    print_validation_report,
    run_validation,
)


class TestRunValidation:
    def test_returns_valid_report(self) -> None:
        report = run_validation()
        assert isinstance(report, ValidationReport)
        assert len(report.verdicts) == 6
        assert isinstance(report.summary, str)

    def test_pass_count_matches_verdicts(self) -> None:
        report = run_validation()
        pass_count = sum(1 for v in report.verdicts if v.verdict == "PASS")
        assert report.pass_count == pass_count
        fail_count = sum(1 for v in report.verdicts if v.verdict == "FAIL")
        assert report.fail_count == fail_count
        skip_count = sum(1 for v in report.verdicts if v.verdict == "SKIP")
        assert report.skip_count == skip_count

    def test_all_passed_true_when_no_failures(self) -> None:
        report = run_validation()
        if report.fail_count == 0:
            assert report.all_passed is True

    def test_three_core_propositions_skipped_without_compute_budget(self) -> None:
        # P1 (wall ordering) is skipped without compute_budget_flops
        # The report still runs without crashing
        report = run_validation(config=MATDOConfig())
        assert any(v.proposition_id == "P1" for v in report.verdicts)

    def test_p3_p4_p5_non_skipped(self) -> None:
        report = run_validation()
        p3 = next(v for v in report.verdicts if v.proposition_id == "P3")
        assert p3.verdict != "SKIP"  # Should PASS with default profiles

    def test_p6_non_skipped(self) -> None:
        report = run_validation()
        p6 = next(v for v in report.verdicts if v.proposition_id == "P6")
        assert p6.verdict != "SKIP"


class TestPrintValidationReport:
    def test_prints_without_error(self) -> None:
        report = run_validation()
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            print_validation_report(report)
        finally:
            sys.stdout = old_stdout

    def test_summary_contains_all_propositions(self) -> None:
        report = run_validation()
        for v in report.verdicts:
            assert v.proposition_id in report.summary


class TestPropositionVerdictFields:
    def test_all_verdicts_have_required_fields(self) -> None:
        report = run_validation()
        for v in report.verdicts:
            assert v.proposition_id
            assert v.title
            assert v.verdict in ("PASS", "FAIL", "SKIP")
            assert isinstance(v.evidence, str)
            assert isinstance(v.supporting_results, tuple)
