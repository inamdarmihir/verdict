"""Component Four — CheckpointCommit: blast-radius boundaries via git.

Every merge point between the bounded and escalation paths is also a
checkpoint: a known-good state the system can roll back to if a downstream
verifier later fails.

In most agent frameworks, checkpoints mean *resumability*. In Verdict they
also mean *containment*: bound how much bad work can accumulate before anyone
notices. Blast radius after step ``k``::

    B(k) = |{ j > k : depends(j, k) }|

Checkpointing after every gated step keeps ``B(k)`` small by construction.

Pair this with the host loop's own checkpointer (e.g. LangGraph
``MemorySaver``) — that persists *graph* state for resume; this module
persists *repo* state for blast-radius control. Different axes.

Public surface owned by this module
-----------------------------------
* ``CheckpointCommit``
"""

from __future__ import annotations

import subprocess


class CheckpointCommit:
    """Record (and optionally roll back to) a known-good repository state.

    Parameters
    ----------
    repo_path:
        Git working tree root. Ignored for SHA generation when ``use_git=False``.
    use_git:
        When ``True``, create real commits (``git add -A`` + ``commit``).
        When ``False``, push synthetic ``snap-N-<step_id>`` tokens onto an
        in-memory stack — useful for demos, CI, and unit tests.

    Notes
    -----
    Rollback is boring and total: hard reset to the previous checkpoint SHA,
    discard uncommitted agent edits, preserve the escalation label for
    calibration.
    """

    def __init__(self, repo_path: str, use_git: bool = True) -> None:
        self.repo_path = repo_path
        self.use_git = use_git
        self._stack: list[str] = []

    @property
    def stack(self) -> list[str]:
        """Copy of checkpoint SHAs / synthetic snap ids, oldest first."""
        return list(self._stack)

    def commit(self, step_id: str, description: str) -> str:
        """Create a checkpoint after a gated step and return its identifier.

        Parameters
        ----------
        step_id:
            Stable step id embedded in the commit message / snap token.
        description:
            Step intent; truncated to 72 chars in the git commit subject.

        Returns
        -------
        str
            Git SHA (or synthetic snap id when ``use_git=False``).
        """
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
        """Hard-reset the working tree to the latest checkpoint.

        Returns
        -------
        str
            The SHA / snap id reset to, or ``\"ROOT\"`` when the stack is empty.
        """
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
