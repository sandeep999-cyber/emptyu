import pytest
import torch

from src.training.losses.masked_modeling import MaskGenerator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_generator_device_matches_mask():
    """Regression: a CPU generator must not be used with CUDA masks."""
    gen = MaskGenerator(mask_ratio=0.15, device=torch.device("cuda"))
    mask = torch.ones(8, 512, dtype=torch.bool, device="cuda")
    out = gen(mask)
    assert out.device.type == "cuda"

    gen_span = MaskGenerator(
        mask_ratio=0.15, mask_mode="span", span_len=16, device=torch.device("cuda")
    )
    out_span = gen_span(mask)
    assert out_span.device.type == "cuda"


def test_set_state_cross_device_fallback():
    """Restoring an incompatible RNG state must re-seed, not crash."""
    gen = MaskGenerator(mask_ratio=0.15, seed=42)
    gen.set_state(torch.zeros(0, dtype=torch.uint8))  # wrong-size state
    out = gen(torch.ones(4, 32, dtype=torch.bool))
    assert out.shape == (4, 32)


class TestMaskGenerator:
    @pytest.fixture
    def mask(self):
        m = torch.ones(100, 512, dtype=torch.bool)
        m[:, -10:] = False
        return m

    def test_mask_rate(self, mask):
        gen = MaskGenerator(mask_ratio=0.15)
        masked = gen(mask)
        rate = masked.sum().item() / mask[:, :-10].sum().item()
        assert 0.10 <= rate <= 0.20

    def test_mask_ratio_very_small(self, mask):
        gen = MaskGenerator(mask_ratio=0.0001)
        masked = gen(mask)
        assert masked.sum().item() >= 1  # clamp(min=1) ensures at least 1

    def test_mask_ratio_one(self, mask):
        gen = MaskGenerator(mask_ratio=1.0)
        masked = gen(mask)
        assert masked[:, :-10].all()

    def test_output_shape(self, mask):
        gen = MaskGenerator(0.15)
        masked = gen(mask)
        assert masked.shape == (100, 512)
        assert masked.dtype == torch.bool

    def test_deterministic(self):
        m = torch.ones(10, 128, dtype=torch.bool)
        gen = MaskGenerator(0.15, seed=42)
        m1 = gen(m.clone())
        gen2 = MaskGenerator(0.15, seed=42)
        m2 = gen2(m.clone())
        assert torch.allclose(m1, m2)

    def test_clamp_at_least_one(self):
        m = torch.ones(1, 10, dtype=torch.bool)
        gen = MaskGenerator(0.01)
        masked = gen(m)
        assert masked.sum().item() >= 1

    def test_only_valid_positions_masked(self):
        m = torch.ones(10, 100, dtype=torch.bool)
        m[:, -50:] = False
        gen = MaskGenerator(0.5)
        masked = gen(m)
        assert (~masked[:, -50:]).all()


class TestMaskGeneratorSpan:
    def test_span_contiguity(self):
        m = torch.ones(20, 512, dtype=torch.bool)
        gen = MaskGenerator(mask_ratio=0.15, seed=42, mask_mode="span", span_len=16)
        masked = gen(m)
        # Masked positions must form contiguous runs of length <= span_len.
        for row in masked:
            idxs = row.nonzero(as_tuple=False).view(-1)
            if len(idxs) == 0:
                continue
            breaks = (idxs[1:] - idxs[:-1] != 1).nonzero(as_tuple=False).view(-1).tolist()
            split_points = [0] + [i + 1 for i in breaks] + [len(idxs)]
            runs = [idxs[split_points[i]:split_points[i + 1]] for i in range(len(split_points) - 1)]
            assert runs, "expected at least one masked run"
            assert max(len(r) for r in runs) <= 16, "each run must be at most span_len long"

    def test_span_only_valid_positions(self):
        m = torch.ones(10, 100, dtype=torch.bool)
        m[:, -40:] = False
        gen = MaskGenerator(mask_ratio=0.3, seed=7, mask_mode="span", span_len=16)
        masked = gen(m)
        assert (~masked[:, -40:]).all()

    def test_span_rate(self):
        m = torch.ones(50, 512, dtype=torch.bool)
        gen = MaskGenerator(mask_ratio=0.15, seed=1, mask_mode="span", span_len=16)
        masked = gen(m)
        rate = masked.sum().item() / m.sum().item()
        assert 0.05 <= rate <= 0.30

    def test_span_deterministic(self):
        m = torch.ones(10, 128, dtype=torch.bool)
        g1 = MaskGenerator(0.15, seed=42, mask_mode="span", span_len=16)
        g2 = MaskGenerator(0.15, seed=42, mask_mode="span", span_len=16)
        assert torch.equal(g1(m.clone()), g2(m.clone()))
