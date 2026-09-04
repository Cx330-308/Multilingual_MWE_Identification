from mwe_roberta.analysis import extract_mwe_spans, parse_official_breakdown
from mwe_roberta.cupt import CuptSentence, CuptToken


def _token(token_id: str, form: str, mwe: str) -> CuptToken:
    return CuptToken([token_id, form, "_", "_", "_", "_", "_", "_", "_", "_", mwe])


def test_parse_official_breakdown_extracts_counts_and_dimensions():
    text = """## Global evaluation
* MWE-based: P=3/4=0.7500 R=3/5=0.6000 F=0.6667
* Tok-based: P=5/6=0.8333 R=5/8=0.6250 F=0.7143

## MWE continuity (partition of Global)
* Continuous: MWE-proportion: gold=4/5=80% pred=3/4=75%
* Continuous: MWE-based: P=2/3=0.6667 R=2/4=0.5000 F=0.5714
"""

    scores = parse_official_breakdown(text)

    assert len(scores) == 2
    assert scores[0].dimension == "global"
    assert scores[0].group == "All"
    assert scores[0].correct == 3
    assert scores[0].false_positives == 1
    assert scores[0].false_negatives == 2
    assert scores[1].dimension == "continuity"
    assert scores[1].group == "Continuous"


def test_extract_mwe_spans_handles_discontinuous_and_overlapping_ids():
    sentence = CuptSentence(
        comments=[],
        tokens=[
            _token("1", "take", "1:LVC.full"),
            _token("2", "a", "2:VID"),
            _token("3", "walk", "1;2"),
        ],
    )

    assert extract_mwe_spans(sentence) == {
        (1, 3): "LVC.full",
        (2, 3): "VID",
    }


def test_extract_mwe_spans_absorbs_contraction_range_annotations():
    sentence = CuptSentence(
        comments=[],
        tokens=[
            _token("1-2", "can't", "1:VID"),
            _token("1", "ca", "*"),
            _token("2", "n't", "*"),
            _token("3", "go", "1"),
        ],
    )

    assert extract_mwe_spans(sentence) == {(1, 2, 3): "VID"}
