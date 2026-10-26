"""Ответы агента не должны быть зашиты под конкретное произведение (раньше «Муму» жило в nodes.py и web_search.py)."""

from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent_module"


def test_agent_package_has_no_work_specific_branches():
    offenders = []
    for path in AGENT_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for needle in ("муму", "mumu", "тургенев"):
            if needle in text:
                offenders.append(f"{path.relative_to(AGENT_DIR)}: {needle}")
    assert not offenders, offenders


def test_no_mock_names_left_in_tools():
    assert not list(AGENT_DIR.rglob("mocks.py"))
    for path in AGENT_DIR.rglob("*.py"):
        assert "_mock" not in path.read_text(encoding="utf-8"), path
