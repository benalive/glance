import random
from collections import Counter

from scripts.build_clean_mix import _distractors, _named_alternative, flickr_id, is_calib


def test_named_alternative_of_or_questions():
    assert _named_alternative("Is the sky blue or white?", "white") == "blue"
    assert _named_alternative("Is the sky blue or white?", "blue") == "white"
    assert _named_alternative("Is the shirt light brown or dark blue?", "light brown") == "dark blue"
    # phrase options it cannot align are left alone (the question then gets ordinary distractors)
    assert _named_alternative("Is the table on the left or on the right?", "right") is None
    assert _named_alternative("What color is the sky?", "blue") is None


def test_flickr_id_from_static_urls_and_photo_pages():
    assert flickr_id("http://farm3.staticflickr.com/2383/2203235497_7a4d5e3d8c_z.jpg") == "2203235497"
    assert flickr_id("https://www.flickr.com/photos/118815643@N04/15340259497/") == "15340259497"
    assert flickr_id("https://www.flickr.com/photos/118815643@N04/15340259497") == "15340259497"


def test_distractors_skip_gold_plurals_synonyms_and_named_alternative():
    pool = Counter({"dog": 50, "dogs": 40, "cat": 30, "grey": 20, "gray": 20, "horse": 10, "bird": 5, "cow": 5})
    for seed in range(20):
        out = _distractors(pool, "dog", 3, random.Random(seed), exclude=("cat",))
        assert out is not None and len(set(out)) == 3
        assert not {"dog", "dogs", "cat"} & set(out)
        grey = _distractors(pool, "gray", 3, random.Random(seed))
        assert "grey" not in grey
    assert _distractors(Counter({"dog": 1, "dogs": 1}), "dog", 3, random.Random(0)) is None


def test_derived_copies_share_their_photo_split():
    from glance.train import _is_dev

    ids = [f"coco:{i}" for i in range(2000)]
    assert all(is_calib(i) == is_calib(i + "#quality") for i in ids)
    # train.R1Data groups by the part before '#', so a copy's dev membership is its photo's
    assert sum(_is_dev(i) for i in ids) > 0


def test_strict_named_alternative_rejects_phrases_and_strips_articles():
    known = {"blue", "white", "train", "giraffe", "horse"}
    assert _named_alternative("Is it a giraffe or a horse?", "horse", known) == "giraffe"
    assert _named_alternative("Is that a train or a bus?", "bus", known) == "train"
    assert _named_alternative("Is the pizza to the left or to the right of the table the pizza is on?", "left", known) is None


def test_weighted_vqav2_distractors_follow_answer_frequency():
    from glance.data.train_mix import vqav2_decision

    vocab = {("other", "what color"): Counter({"red": 1000, "blue": 900, "green": 800, **{f"rare{i}": 1 for i in range(20)}})}
    r = {"question_id": 1, "question": "What color is it?", "answer_type": "other", "question_type": "what color",
         "multiple_choice_answer": "red", "answers": [{"answer": "red"}] * 10}
    picks = Counter()
    for seed in range(200):
        d, _ = vqav2_decision(r, vocab, random.Random(seed), b"", "i", weighted=True)
        picks.update(c for c in d.candidates if c != "red")
    rare = sum(v for k, v in picks.items() if k.startswith("rare"))
    assert picks["blue"] > 180 and picks["green"] > 180 and rare < 250  # uniform draws would give ~ 55 / 55 / 490
