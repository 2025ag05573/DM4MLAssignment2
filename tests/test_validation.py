import pytest

from src.validation.validate import DataQualityError, enforce_gate, validate_tables
from tests.conftest import NOW


def checks_failed(out, table, severity=None):
    r = out["results"]
    r = r[(r.table == table) & (r.failed_rows > 0)]
    if severity:
        r = r[r.severity == severity]
    return set(r.check)


def test_clean_data_passes_everything(clean_raw, cfg):
    out = validate_tables(clean_raw, cfg, now_ts=NOW)
    for t in ("users", "movies", "ratings"):
        assert out["summary"][t]["rows_quarantined"] == 0
    # ZIP+4 is the only finding, and it is a warning
    assert checks_failed(out, "users") == {"zip_format"}


def test_rating_defects_are_quarantined_with_reasons(dirty_raw, cfg):
    out = validate_tables(dirty_raw, cfg, now_ts=NOW)
    errs = out["errors"]["ratings"]
    reasons = ";".join(errs["_error_checks"])
    for check in ("rating_in_scale", "timestamp_in_range", "orphan_user", "orphan_movie",
                  "missing_key", "bad_dtype"):
        assert check in reasons, check
    assert len(errs) == 7                              # the 7 bad rating rows, nothing else
    assert len(out["validated"]["ratings"]) == 5       # 4 clean + the duplicate (kept, warned)
    assert "duplicate_interaction" in checks_failed(out, "ratings", "warning")


def test_user_defects(dirty_raw, cfg):
    out = validate_tables(dirty_raw, cfg, now_ts=NOW)
    assert checks_failed(out, "users", "error") == {"duplicate_primary_key", "bad_dtype", "missing_key"}
    assert {"gender_valid", "age_code_valid", "occupation_valid", "zip_format"} <= checks_failed(
        out, "users", "warning")
    assert out["summary"]["users"]["rows_quarantined"] == 3


def test_movie_warnings_do_not_quarantine(dirty_raw, cfg):
    out = validate_tables(dirty_raw, cfg, now_ts=NOW)
    assert {"title_has_year", "genre_vocabulary"} <= checks_failed(out, "movies", "warning")
    assert out["summary"]["movies"]["rows_quarantined"] == 0


def test_validated_tables_are_typed(clean_raw, cfg):
    out = validate_tables(clean_raw, cfg, now_ts=NOW)
    assert str(out["validated"]["ratings"]["rating"].dtype) == "Int64"
    assert str(out["validated"]["users"]["zip_code"].dtype) == "string"


def test_quality_gate():
    enforce_gate({"ratings": {"error_rate": 0.005}}, 0.01)
    with pytest.raises(DataQualityError):
        enforce_gate({"ratings": {"error_rate": 0.02}}, 0.01)


def test_quarantine_evidence_lists_every_quarantined_row(dirty_raw, cfg):
    from src.validation.validate import quarantine_evidence
    out = validate_tables(dirty_raw, cfg, now_ts=NOW)
    ev = quarantine_evidence(out)
    n_quarantined = sum(s["rows_quarantined"] for s in out["summary"].values())
    assert len(ev) == n_quarantined > 0
    assert list(ev.columns) == ["table", "source_line", "user_id", "movie_id", "failed_checks"]
    assert (ev.loc[ev.table == "ratings", "failed_checks"].str.len() > 0).all()


def test_source_checksum_policy():
    from src.ingestion.ingest import check_content
    md5 = {"ratings": "abc"}
    assert check_content(md5, {})["status"] == {"ratings": "not pinned"}
    assert check_content(md5, {"expected_file_md5": {"ratings": "abc"}})["status"] == {"ratings": "match"}
    warn = check_content(md5, {"expected_file_md5": {"ratings": "xyz"}, "checksum_policy": "warn"})
    assert warn["status"] == {"ratings": "MISMATCH"}
    with pytest.raises(ValueError, match="checksum mismatch"):
        check_content(md5, {"expected_file_md5": {"ratings": "xyz"}, "checksum_policy": "fail"})
