"""Every ```python block in README.md must run, and every transform must be documented."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from datapalette.transforms import REGISTRY

README = (Path(__file__).parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, flags=re.S)


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_block_runs(code, tmp_path, monkeypatch):
    if "from sklearn" in code:
        pytest.importorskip("sklearn")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "README.md", "exec"), {"__name__": "__readme__"})


def test_every_transform_is_in_the_table():
    missing = [name for name in REGISTRY if f"| `{name}` |" not in README]
    assert not missing, f"README transform table is missing {missing}"
