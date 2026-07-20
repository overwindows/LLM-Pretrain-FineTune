"""Shared utilities: subprocess runner, retry, structured logging."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_cmd(
    cmd: list[str],
    env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
    check: bool = True,
    log_prefix: str = "",
) -> subprocess.CompletedProcess:
    """Run a command, streaming stdout/stderr to the logger in real time.

    Parameters
    ----------
    cmd:
        Command as a list of strings.
    env:
        Full environment dict (use ``build_subprocess_env`` from env_setup).
        None = inherit os.environ.
    cwd:
        Working directory for the subprocess.
    check:
        If True, raise RuntimeError if the process exits non-zero.
    log_prefix:
        String prepended to each log line (useful to identify the stage).

    Returns
    -------
    subprocess.CompletedProcess with returncode.
    """
    prefix = f"[{log_prefix}] " if log_prefix else ""
    cmd_str = " ".join(cmd)
    logger.info("%sRunning: %s", prefix, cmd_str)
    if cwd:
        logger.info("%s  cwd=%s", prefix, cwd)

    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        logger.info("%s%s", prefix, line)
    proc.wait()

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{prefix}Command failed (rc={proc.returncode}): {cmd_str}"
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, "\n".join(lines), "")


def retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    delay_secs: float = 5.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    label: str = "",
) -> T:
    """Call fn() up to max_attempts times, with exponential backoff on failure.

    Parameters
    ----------
    fn:
        Callable to retry.
    max_attempts:
        Maximum number of attempts (including the first).
    delay_secs:
        Seconds to wait before the second attempt.
    backoff:
        Multiply delay_secs by this factor on each failure.
    exceptions:
        Tuple of exception types to catch and retry on.
    label:
        Human-readable name for logging.

    Returns
    -------
    The return value of fn() on success.

    Raises
    ------
    The last exception raised by fn() if all attempts fail.
    """
    delay = delay_secs
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    "[%s] attempt %d/%d failed: %s — retrying in %.1fs",
                    label, attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
                delay *= backoff
            else:
                logger.error(
                    "[%s] all %d attempts failed: %s", label, max_attempts, exc
                )
    assert last_exc is not None
    raise last_exc


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to stdout with timestamp + level."""
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
