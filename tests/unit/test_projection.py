import pytest
import torch

from src.models.teacher.projection import FeatureProjection, ReconstructionHead


CALENDAR_SPEC = {
    "hour_sin": {"index": 5, "classes": 24, "offset": 0},
    "hour_cos": {"index": 5, "classes": 24, "offset": 0},
    "minute_sin": {"index": 7, "classes": 60, "offset": 0},
    "minute_cos": {"index": 7, "classes": 60, "offset": 0},
    "day_of_week_sin": {"index": 9, "classes": 7, "offset": 0},
    "day_of_week_cos": {"index": 9, "classes": 7, "offset": 0},
    "month_sin": {"index": 11, "classes": 12, "offset": 1},
    "month_cos": {"index": 11, "classes": 12, "offset": 1},
}


class TestFeatureProjection:
    def test_output_shape(self):
        proj = FeatureProjection(feature_dim=15, d_model=512)
        x = torch.randn(2, 512, 15)
        out = proj(x)
        assert out.shape == (2, 512, 512)

    def test_different_batch_sizes(self):
        proj = FeatureProjection(15, 512)
        for b in [1, 4, 16]:
            x = torch.randn(b, 128, 15)
            assert proj(x).shape == (b, 128, 512)

    def test_gradient_flow(self):
        proj = FeatureProjection(15, 512)
        x = torch.randn(2, 64, 15, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_deterministic(self):
        proj = FeatureProjection(15, 512)
        x = torch.randn(2, 32, 15)
        torch.manual_seed(0)
        out1 = proj(x)
        torch.manual_seed(0)
        out2 = proj(x)
        assert torch.allclose(out1, out2)


class TestReconstructionHead:
    def test_price_output(self):
        head = ReconstructionHead(d_model=512, calendar_spec=CALENDAR_SPEC)
        x = torch.randn(2, 512, 512)
        out = head(x)
        assert "price" in out
        assert out["price"].shape == (2, 512, 5)

    def test_funding_oi_output(self):
        head = ReconstructionHead(512, CALENDAR_SPEC)
        x = torch.randn(2, 512, 512)
        out = head(x)
        assert "funding_oi" in out
        assert out["funding_oi"].shape == (2, 512, 2)

    def test_calendar_outputs(self):
        head = ReconstructionHead(512, CALENDAR_SPEC)
        x = torch.randn(2, 512, 512)
        out = head(x)
        assert "calendar" in out
        for field in CALENDAR_SPEC:
            assert field in out["calendar"]
            assert out["calendar"][field].shape == (2, 512, CALENDAR_SPEC[field]["classes"])

    def test_deterministic(self):
        head = ReconstructionHead(512, CALENDAR_SPEC)
        x = torch.randn(2, 32, 512)
        torch.manual_seed(0)
        out1 = head(x)
        torch.manual_seed(0)
        out2 = head(x)
        for k in out1:
            if k == "calendar":
                for f in out1[k]:
                    assert torch.allclose(out1[k][f], out2[k][f])
            else:
                assert torch.allclose(out1[k], out2[k])

    def test_gradient_flow(self):
        head = ReconstructionHead(512, CALENDAR_SPEC)
        x = torch.randn(2, 64, 512, requires_grad=True)
        out = head(x)
        loss = out["price"].sum()
        loss.backward()
        assert x.grad is not None
