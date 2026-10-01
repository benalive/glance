import pytest

from glance.data.benchmarks import parse_mmstar


def test_parse_mmstar_handles_commas_in_options():
    q = "What is shown?\nOptions: A: A cat, sitting on a mat., B: Two dogs, C: 3, 4, and 5, D: None"
    stem, opts = parse_mmstar(q)
    assert stem == "What is shown?"
    assert opts == ["A cat, sitting on a mat.", "Two dogs", "3, 4, and 5", "None"]


def test_parse_mmstar_two_options_and_newlines():
    stem, opts = parse_mmstar("Is it day?\nOptions: A: Yes\nB: No")
    assert opts == ["Yes", "No"]


def test_parse_mmstar_rejects_missing_marker():
    with pytest.raises(ValueError):
        parse_mmstar("no options here")


def _cached(fn):
    try:
        return fn()
    except Exception as e:  # not downloaded
        pytest.skip(f"benchmark not cached: {type(e).__name__}")


def test_real_benchmarks_load_with_expected_shapes():
    from glance.data import benchmarks as b

    pope = _cached(lambda: b.pope("adversarial"))
    assert len(pope) == 3000 and len({d.image_id for d in pope}) == 500
    assert sum(d.label == 0 for d in pope) == 1500
    ok = _cached(b.aokvqa_val)
    assert len(ok) == 1145 and all(len(d.candidates) == 4 for d in ok)
    ms = _cached(b.mmstar)
    assert len(ms) >= 1480 and all(2 <= len(d.candidates) <= 6 for d in ms)
    for sub in ["replace_rel", "swap_att"]:
        sc = _cached(lambda: b.sugarcrepe(sub))
        assert len(sc) >= 600 and all(len(d.candidates) == 2 for d in sc)
        assert 0.4 < sum(d.label == 0 for d in sc) / len(sc) < 0.6  # order randomised
    kq = _cached(b.koniq_test)
    assert len(kq) == 2015 and all(len(d.target) == 5 for d in kq)


def test_parse_mmstar_mathvista_layout():
    q = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: How many snowmen are there?\nChoices:\n(A) 10\n(B) 15\n(C) 12\n(D) 20"
    assert parse_mmstar(q) == ("How many snowmen are there?", ["10", "15", "12", "20"])
