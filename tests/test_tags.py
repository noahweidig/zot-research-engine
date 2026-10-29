from pipeline.tags import normalize_tags


def test_lowercase_whitespace_dedupe():
    assert normalize_tags(["  Fire   Ecology ", "fire ecology", "FIRE ECOLOGY"]) == [
        "fire ecology"
    ]


def test_drops_noise():
    raw = ["AI Suggested", "Score 4", "score5", "N/A", "TODO", "truncated tag...", "ok"]
    assert normalize_tags(raw) == ["ok"]


def test_drops_long_tags():
    assert normalize_tags(["one two three four five", "one two three four five six"]) == [
        "one two three four five"
    ]


def test_caps_at_six():
    assert normalize_tags([f"t{i}" for i in range(10)]) == [f"t{i}" for i in range(6)]


def test_empty():
    assert normalize_tags([]) == []
    assert normalize_tags(["", "   "]) == []
