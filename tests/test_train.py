"""T2.3: false-negative masking — masked logits are -inf; with no collisions
the masked loss equals the unmasked reference exactly."""
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from esmc_retrieval.train import false_negative_mask, nce_logits, symmetric_nce  # noqa: E402


def _scores(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, b, generator=g)


def test_masked_logits_are_neg_inf():
    s = _scores()
    mask = false_negative_mask(["a", "a", "b", "c", "c", "b"])
    assert mask[0, 1] and mask[1, 0] and mask[3, 4] and mask[4, 3]
    assert not mask.diagonal().any()
    logits = nce_logits(s, tau=0.5, forbidden=mask)
    assert (logits[mask] == float("-inf")).all()
    assert torch.equal(logits[~mask], (s / 0.5)[~mask])


def test_no_collisions_equals_unmasked():
    s = _scores()
    labels = [f"sf{i}" for i in range(6)]  # all distinct
    mask = false_negative_mask(labels)
    assert not mask.any()
    assert torch.equal(symmetric_nce(s, tau=0.1, forbidden=mask),
                       symmetric_nce(s, tau=0.1, forbidden=None))


def test_masking_excludes_false_negative_from_denominator():
    # A huge off-diagonal "negative" that is actually a same-sf pair: masked
    # loss must equal the loss where that score is suppressed instead.
    s = _scores()
    s[0, 1] = s[1, 0] = 1e4
    mask = false_negative_mask(["x", "x", "a", "b", "c", "d"])
    masked = symmetric_nce(s, tau=1.0, forbidden=mask)
    ref = s.clone()
    ref[0, 1] = ref[1, 0] = -100.0  # suppressed, not -inf (loses ~0 mass)
    assert torch.isfinite(masked)
    assert masked < symmetric_nce(s, tau=1.0)  # unmasked is hurt by the spike
    assert abs(masked - symmetric_nce(ref, tau=1.0)) < 1e-3
