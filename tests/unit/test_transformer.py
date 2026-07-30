import pytest
import torch

from src.models.teacher.transformer import TransformerBlock
from src.models.teacher.positional_encoding import RotaryPositionalEncoding


class TestTransformerBlock:
    @pytest.fixture
    def block_512(self):
        rope = RotaryPositionalEncoding(d_model=512)
        return TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.1, rope=rope)

    @pytest.fixture
    def block_128(self):
        rope = RotaryPositionalEncoding(d_model=128)
        return TransformerBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.1, rope=rope)

    @pytest.fixture
    def block_deterministic(self):
        rope = RotaryPositionalEncoding(d_model=512)
        return TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.0, rope=rope)

    def test_output_shape(self, block_512):
        x = torch.randn(2, 128, 512)
        mask = torch.ones(2, 128, dtype=torch.bool)
        pos = torch.zeros(2, 128, dtype=torch.long)
        out = block_512(x, key_padding_mask=mask, positions=pos)
        assert out.shape == (2, 128, 512)

    def test_no_nan(self, block_128):
        x = torch.randn(1, 64, 128)
        mask = torch.ones(1, 64, dtype=torch.bool)
        pos = torch.zeros(1, 64, dtype=torch.long)
        out = block_128(x, key_padding_mask=mask, positions=pos)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, block_128):
        x = torch.randn(2, 32, 128, requires_grad=True)
        mask = torch.ones(2, 32, dtype=torch.bool)
        pos = torch.zeros(2, 32, dtype=torch.long)
        out = block_128(x, key_padding_mask=mask, positions=pos)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_deterministic(self, block_deterministic):
        x = torch.randn(2, 32, 512)
        mask = torch.ones(2, 32, dtype=torch.bool)
        pos = torch.zeros(2, 32, dtype=torch.long)
        torch.manual_seed(0)
        out1 = block_deterministic(x, key_padding_mask=mask, positions=pos)
        torch.manual_seed(0)
        out2 = block_deterministic(x, key_padding_mask=mask, positions=pos)
        assert torch.allclose(out1, out2)

    def test_with_padding_mask(self, block_128):
        x = torch.randn(2, 16, 128)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[0, -1] = False
        pos = torch.zeros(2, 16, dtype=torch.long)
        out = block_128(x, key_padding_mask=mask, positions=pos)
        assert out.shape == x.shape


class TestRotaryPositionalEncoding:
    @pytest.fixture
    def rope_64(self):
        return RotaryPositionalEncoding(d_model=64)

    def test_apply_rope_shape(self, rope_64):
        q = torch.randn(2, 128, 64)
        k = torch.randn(2, 128, 64)
        pos = torch.arange(128).unsqueeze(0).expand(2, -1)
        q_out, k_out = rope_64.apply_rope(q, k, positions=pos)
        assert q_out.shape == (2, 128, 64)
        assert k_out.shape == (2, 128, 64)

    def test_rotation_property(self, rope_64):
        q = torch.randn(1, 10, 64)
        k = torch.randn(1, 10, 64)
        pos = torch.arange(10).unsqueeze(0)
        q_out, k_out = rope_64.apply_rope(q, k, positions=pos)
        assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5)

    def test_zero_position_no_change(self, rope_64):
        q = torch.randn(1, 5, 64)
        k = torch.randn(1, 5, 64)
        pos = torch.zeros(1, 5, dtype=torch.long)
        q_out, k_out = rope_64.apply_rope(q, k, positions=pos)
        assert torch.allclose(q, q_out)
        assert torch.allclose(k, k_out)

    def test_position_sensitivity(self, rope_64):
        q = torch.randn(1, 1, 64)
        k = torch.randn(1, 1, 64)
        pos0 = torch.zeros(1, 1, dtype=torch.long)
        pos1 = torch.ones(1, 1, dtype=torch.long)
        q0, k0 = rope_64.apply_rope(q, k, positions=pos0)
        q1, k1 = rope_64.apply_rope(q, k, positions=pos1)
        assert not torch.allclose(q0, q1)

    def test_deterministic(self, rope_64):
        q = torch.randn(1, 10, 64)
        k = torch.randn(1, 10, 64)
        pos = torch.arange(10).unsqueeze(0)
        q1, k1 = rope_64.apply_rope(q, k, positions=pos)
        q2, k2 = rope_64.apply_rope(q, k, positions=pos)
        assert torch.allclose(q1, q2)
        assert torch.allclose(k1, k2)
