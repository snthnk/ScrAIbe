import pytest

from agent_module.constants import ALL_TOOL_NAMES, MAX_TOTAL_SCORE, OFFICIAL_MAX, normalize_criterion_id
from agent_module.tools import ALL_TOOLS


def test_official_max_sums_to_22():
    assert MAX_TOTAL_SCORE == 22
    assert sum(OFFICIAL_MAX.values()) == 22


@pytest.mark.parametrize(
    "raw, expected",
    [("K2", "K2"), ("к2", "K2"), ("К10", "K10"), (" k4 ", "K4"), ("К 3", "K3"), ("К11", None), ("критерий", None), ("", None)],
)
def test_normalize_criterion_id(raw, expected):
    assert normalize_criterion_id(raw) == expected


def test_tool_registry_matches_constants():
    assert {t.name for t in ALL_TOOLS} == set(ALL_TOOL_NAMES)
    assert len(ALL_TOOLS) == len(ALL_TOOL_NAMES)
