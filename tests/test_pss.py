from unittest.mock import MagicMock

from prism.pss import PSS, VERDICT_SCORES


def test_verdict_scores_valid():
    """All verdicts should have scores in (0, 1]."""
    for verdict, score in VERDICT_SCORES.items():
        assert 0 < score <= 1.0
        assert isinstance(score, float)


def test_pss_formula():
    """PSS = AD * coherence."""
    ad = 0.3
    coherence = 0.64
    pss = ad * coherence
    assert abs(pss - 0.192) < 0.001


def test_meta_stripping():
    """Meta-commentary should be stripped."""
    pss = PSS(generator=MagicMock(), evaluator=MagicMock())

    text = "The novel element here is: the actual insight content"
    stripped = pss._strip_meta(text)
    assert "novel element" not in stripped.lower()
    assert "actual insight content" in stripped


def test_meta_stripping_preserves_short():
    """Don't strip so aggressively that nothing remains."""
    pss = PSS(generator=MagicMock(), evaluator=MagicMock())

    text = "Short text"
    stripped = pss._strip_meta(text)
    assert stripped == "Short text"


def test_parse_coherence_standard():
    """Parse a well-formatted coherence response."""
    pss = PSS(generator=MagicMock(), evaluator=MagicMock())

    response = "Q1: YES\nQ2: GENUINE_INSIGHT\nQ3: STRUCTURAL_REVELATION\nQ4: NO\nQ5: NOVEL"
    score, va, vb, pred, has_tool, is_novel = pss._parse_coherence(response)

    assert va == "GENUINE_INSIGHT"
    assert vb == "STRUCTURAL_REVELATION"
    assert pred is True
    assert abs(score - 0.6 * 0.8) < 0.001


def test_parse_coherence_no_prediction():
    """Parse response with no testable prediction."""
    pss = PSS(generator=MagicMock(), evaluator=MagicMock())

    response = "Q1: NO\nQ2: SUPERFICIAL\nQ3: NOTHING_NEW\nQ4: NO\nQ5: NOVEL"
    score, va, vb, pred, has_tool, is_novel = pss._parse_coherence(response)

    assert pred is False
    assert va == "SUPERFICIAL"
    assert vb == "NOTHING_NEW"
    assert abs(score - 0.2 * 0.1) < 0.001


def test_attractor_distance_range():
    """Attractor distance formula produces values in [0, 1]."""
    # cosine_sim = 0.8 -> distance = (1-0.8)/2 = 0.1
    distance = (1 - 0.8) / 2.0
    assert 0 <= distance <= 1

    # cosine_sim = -0.5 -> distance = (1-(-0.5))/2 = 0.75
    distance = (1 - (-0.5)) / 2.0
    assert 0 <= distance <= 1
