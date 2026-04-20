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


def collect_workspace_safety_checks(workdir: str | Path) -> dict[str, object]:
    """收集 autoresearch 启动前要求的 git/worktree 安全检查结果。"""
    repo_root = resolve_repo_root(workdir)
    checks = {
        "is_git_repo": False,
        "is_worktree": False,
        "is_clean": False,
        "has_untracked_files": False,
        "reason_code": None,
    }
    try:
        _run_git_command(workdir=repo_root, args=["rev-parse", "--show-toplevel"])
        checks["is_git_repo"] = True
        common_dir = _run_git_command(workdir=repo_root, args=["rev-parse", "--git-common-dir"]).strip()
        git_dir = _run_git_command(workdir=repo_root, args=["rev-parse", "--git-dir"]).strip()
        common_dir_norm = common_dir.replace("\\", "/")
        git_dir_norm = git_dir.replace("\\", "/")
        checks["is_worktree"] = (git_dir_norm != common_dir_norm) or ("/worktrees/" in git_dir_norm)
        porcelain = _run_git_command(workdir=repo_root, args=["status", "--porcelain"])
        lines = [line for line in porcelain.splitlines() if line.strip()]
        checks["has_untracked_files"] = any(line.startswith("?? ") for line in lines)
        checks["is_clean"] = not lines
    except subprocess.CalledProcessError:
        checks["reason_code"] = "not_git_repo"
        return checks

    if not checks["is_worktree"]:
        checks["reason_code"] = "not_worktree"
    elif checks["has_untracked_files"]:
        checks["reason_code"] = "worktree_not_clean"
    elif not checks["is_clean"]:
        checks["reason_code"] = "worktree_not_clean"
    return checks


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


def get_head_commit(workdir: str | Path) -> str:
    """读取当前 HEAD 的短 commit，供 resume 对齐使用。"""
    return get_current_commit(workdir)


def checkout_commit(workdir: str | Path, commit: str) -> None:
    """把当前 worktree 对齐到指定 commit，同时保留所在分支。"""
    _run_git_command(
        workdir=resolve_repo_root(workdir),
        args=["reset", "--hard", str(commit)],
    )


def list_changed_paths(workdir: str | Path, include_untracked: bool = False) -> list[str]:
    """列出相对 HEAD 的改动路径，可选带上未跟踪文件。"""
    repo_root = resolve_repo_root(workdir)
    if not include_untracked:
        output = _run_git_command(
            workdir=repo_root,
            args=["diff", "--name-only", "--relative", "HEAD"],
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    output = _run_git_command(
        workdir=repo_root,
        args=["status", "--porcelain", "--untracked-files=all"],
    )
    changed_paths = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        path_text = raw_line[3:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1].strip()
        if path_text and path_text not in changed_paths:
            changed_paths.append(path_text)
    return changed_paths


def assert_only_allowed_paths_changed(
    changed_paths: list[str],
    allowed_write_roots: list[str],
    read_only_roots: list[str],
) -> list[str]:
    """检查改动是否越出允许写入的研究面白名单。"""
    invalid = []
    for changed_path in changed_paths:
        if _matches_any_root(changed_path, read_only_roots):
            continue
        if _matches_any_root(changed_path, allowed_write_roots):
            continue
        invalid.append(str(changed_path))
    return invalid


def create_experiment_commit(workdir: str | Path, message: str, allowed_paths: list[str] | None = None) -> str:
    """提交当前实验改动，并返回提交后的短 commit。"""
    repo_root = resolve_repo_root(workdir)
    if allowed_paths:
        normalized_paths = [str(path) for path in allowed_paths]
        _run_git_command(workdir=repo_root, args=["add", "--", *normalized_paths])
    else:
        _run_git_command(workdir=repo_root, args=["add", "-A"])
    _run_git_command(workdir=repo_root, args=["commit", "-m", str(message)])
    return get_current_commit(repo_root)


def rollback_candidate_commit(workdir: str | Path, commit: str) -> None:
    """回退本轮 candidate commit，对齐 autoresearch 的单轮撤销语义。"""
    _run_git_command(
        workdir=resolve_repo_root(workdir),
        args=["reset", "--hard", str(commit)],
    )


def rollback_to_commit(workdir: str | Path, commit: str) -> None:
    """把工作区强制回滚到指定 commit。"""
    rollback_candidate_commit(workdir, commit)


def rollback_to_parent_commit(workdir: str | Path, parent_commit: str) -> None:
    """把当前 worktree 回到本轮候选的父 commit。"""
    rollback_candidate_commit(workdir, parent_commit)


def _matches_any_root(path_text: str, roots: list[str]) -> bool:
    """判断路径是否命中任一文件或目录根。"""
    normalized_path = str(path_text).replace("\\", "/").lstrip("./")
    for root in roots:
        normalized_root = str(root).replace("\\", "/").lstrip("./")
        if not normalized_root:
            continue
        if normalized_root.endswith("/"):
            if normalized_path.startswith(normalized_root):
                return True
            continue
        if normalized_path == normalized_root or normalized_path.startswith(normalized_root + "/"):
            return True
    return False


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
