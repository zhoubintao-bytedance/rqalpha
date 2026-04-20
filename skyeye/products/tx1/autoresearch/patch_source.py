"""TX1 autoresearch 的外部 patch 探测接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skyeye.products.tx1.autoresearch.git_ops import list_changed_paths


def detect_workspace_patch(workdir: str | Path) -> dict[str, Any] | None:
    """读取当前 worktree 的候选 patch；没有改动时返回空。"""
    changed_paths = list_changed_paths(workdir, include_untracked=True)
    if not changed_paths:
        return None
    return describe_workspace_patch(changed_paths)


def describe_workspace_patch(changed_paths: list[str]) -> dict[str, Any]:
    """把工作区改动裁剪成 loop 可消费的最小描述。"""
    normalized_paths = [str(path) for path in changed_paths]
    return {
        "changed_paths": normalized_paths,
        "changed_count": len(normalized_paths),
    }
