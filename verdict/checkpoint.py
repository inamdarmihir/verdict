"""Checkpoint commits as blast-radius boundaries for agentic loops."""

from __future__ import annotations

import subprocess


class CheckpointCommit:
    def __init__(self, repo_path: str, use_git: bool = True) -> None:
        self.repo_path = repo_path
        self.use_git = use_git
        self._stack: list[str] = []

    @property
    def stack(self) -> list[str]:
        return list(self._stack)

    def commit(self, step_id: str, description: str) -> str:
        if not self.use_git:
            sha = f"snap-{len(self._stack)}-{step_id}"
        else:
            subprocess.run(["git", "add", "-A"], cwd=self.repo_path, check=True)
            msg = f"verdict-checkpoint: {step_id} — {description[:72]}"
            subprocess.run(
                ["git", "commit", "-m", msg, "--allow-empty"],
                cwd=self.repo_path,
                check=True,
            )
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                text=True,
            ).strip()
        self._stack.append(sha)
        return sha

    def rollback(self) -> str:
        prev = self._stack[-1] if self._stack else "ROOT"
        if self.use_git and self._stack:
            # Hard reset restores tracked files; clean drops untracked agent edits.
            subprocess.run(
                ["git", "reset", "--hard", prev],
                cwd=self.repo_path,
                check=True,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=self.repo_path,
                check=True,
            )
        return prev
