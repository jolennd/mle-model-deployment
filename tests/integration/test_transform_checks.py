"""Data rule checks on the built tables: one pytest case per check, zero bad rows means pass.

The transform variant provides tests/integration/transform_checks.py with two functions:
    cases() -> list[str]        names of the checks (cheap: no database work)
    bad_rows(name: str) -> int  runs one check and returns the number of rows that break it
Without a transform variant the tests are skipped. They need the built tables, so they carry
the `transform` marker and are not part of test-fast. Run them where .env is loaded:
`make transform-test` (variants whose checks run through pytest) or `make upgrade-check`.
"""

import pytest

adapter = pytest.importorskip("transform_checks", reason="no transform variant chosen")


@pytest.mark.transform
@pytest.mark.parametrize("name", adapter.cases())
def test_transform_check_has_no_bad_rows(name):
    bad = adapter.bad_rows(name)
    assert bad == 0, f"{name}: {bad} bad rows"
