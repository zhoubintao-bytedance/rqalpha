"""TX1 autoresearch 的 git 操作封装。"""

from __future__ import annotations

from pathlib import Path
import subprocess


def ensure_read_only_paths_untouched(changed_paths: list[str], read_only_roots: list[str]) -> list[str]:
    """检查变更列表里是否命中了只读路径。"""
    hits = []
    for changed_path in changed_paths:
        for read_only_root in read_only_roots:
            if str(changed_path).startswith(str(read_only_root)):
                hits.append(str(changed_path))
                break
    return hits


def resolve_repo_root(cwd: str | Path) -> Path:
    """把传入目录标准化成 Path，供后续 git 命令使用。"""
    return Path(cwd).resolve()


def get_current_commit(workdir: str | Path) -> str:
    """读取当前工作区 HEAD 的短 commit。"""
    return _run_git_command(
        workdir=resolve_repo_root(workdir),
        args=["rev-parse", "--short", "HEAD"],
    ).strip()


def get_current_branch(workdir: str | Path) -> str:
    """读取当前工作区所在分支名。"""
    return _run_git_command(
        workdir=resolve_repo_root(workdir),
        args=["branch", "--show-current"],
    ).strip()


def list_changed_paths(workdir: str | Path) -> list[str]:
    """列出相对 HEAD 的已修改路径，供只读路径审计使用。"""
    output = _run_git_command(
        workdir=resolve_repo_root(workdir),
        args=["diff", "--name-only", "--relative", "HEAD"],
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def create_experiment_commit(workdir: str | Path, message: str) -> str:
    """提交当前实验改动，并返回提交后的短 commit。"""
    repo_root = resolve_repo_root(workdir)
    _run_git_command(workdir=repo_root, args=["add", "-A"])
    _run_git_command(workdir=repo_root, args=["commit", "-m", str(message)])
    return get_current_commit(repo_root)


def rollback_to_commit(workdir: str | Path, commit: str) -> None:
    """把工作区强制回滚到指定 commit。"""
    _run_git_command(
        workdir=resolve_repo_root(workdir),
        args=["reset", "--hard", str(commit)],
    )


def _run_git_command(*, workdir: Path, args: list[str]) -> str:
    """统一执行 git 子命令，失败时直接抛错。"""
    completed = subprocess.run(
        ["git", *args],
        cwd=str(workdir),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout
