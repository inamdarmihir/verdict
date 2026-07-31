from __future__ import annotations

from pathlib import Path

from verdict.checkpoint import CheckpointCommit


def test_snapshot_checkpoint_stack() -> None:
    cp = CheckpointCommit(repo_path=".", use_git=False)
    sha1 = cp.commit("s1", "first")
    sha2 = cp.commit("s2", "second")
    assert sha1.startswith("snap-0-")
    assert sha2.startswith("snap-1-")
    assert cp.stack == [sha1, sha2]
    assert cp.rollback() == sha2


def test_git_checkpoint(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "README").write_text("hi\n", encoding="utf-8")
    cp = CheckpointCommit(repo_path=str(tmp_path), use_git=True)
    sha = cp.commit("s1", "initial checkpoint")
    assert len(sha) == 40
    (tmp_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    assert (tmp_path / "dirty.txt").exists()
    cp.rollback()
    assert not (tmp_path / "dirty.txt").exists()
