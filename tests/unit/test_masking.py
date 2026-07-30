import pytest
import torch

from src.training.losses.masked_modeling import MaskGenerator


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
