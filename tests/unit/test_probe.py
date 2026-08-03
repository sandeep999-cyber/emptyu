import yaml
from pathlib import Path

import pytest

from src.training.probe import _apply_recommendation


@pytest.fixture()
def src_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "trainer.yaml"
    p.write_text(
        "trainer:\n"
        "  batch_size: 64\n"
        "  micro_batch_size: 32\n"
        "  torch_compile: false\n"
        "  val_batch_size: 64\n",
        encoding="utf-8",
    )
    return p


def test_apply_writes_tuned_copy_and_leaves_source_untouched(src_yaml: Path, tmp_path: Path):
    out = tmp_path / "probe_config.yaml"
    _apply_recommendation(str(src_yaml), 64, True, str(out))

    assert out.exists()
    cfg = yaml.safe_load(out.read_text())["trainer"]
    assert cfg["micro_batch_size"] == 64
    assert cfg["torch_compile"] is True
    # CUDA-graph reduce-overhead is unsafe with this model's autograd on Colab;
    # the tuned config always uses the safe default inductor mode (null).
    assert cfg["compile_mode"] is None

    src = yaml.safe_load(src_yaml.read_text())["trainer"]
    assert src["micro_batch_size"] == 32
    assert src["torch_compile"] is False
    assert "compile_mode" not in src


def test_apply_compile_off_sets_mode_null(src_yaml: Path, tmp_path: Path):
    out = tmp_path / "probe_config.yaml"
    _apply_recommendation(str(src_yaml), 32, False, str(out))
    cfg = yaml.safe_load(out.read_text())["trainer"]
    assert cfg["micro_batch_size"] == 32
    assert cfg["torch_compile"] is False
    assert cfg["compile_mode"] is None


def test_apply_inserts_compile_mode_when_absent(src_yaml: Path, tmp_path: Path):
    out = tmp_path / "probe_config.yaml"
    _apply_recommendation(str(src_yaml), 64, True, str(out))
    lines = out.read_text().splitlines()
    assert "  compile_mode: null" in lines
    assert lines.index("  compile_mode: null") > lines.index("  torch_compile: true")


def test_apply_creates_parent_dirs(src_yaml: Path, tmp_path: Path):
    out = tmp_path / "nested" / "dir" / "probe_config.yaml"
    _apply_recommendation(str(src_yaml), 64, True, str(out))
    assert out.exists()


def test_apply_raises_on_missing_key(src_yaml: Path, tmp_path: Path):
    missing = tmp_path / "bad.yaml"
    missing.write_text("trainer:\n  batch_size: 64\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="micro_batch_size"):
        _apply_recommendation(str(missing), 64, True, str(tmp_path / "out.yaml"))
