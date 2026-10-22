"""Validate the shell files delivered in the bounded smoke source archive."""
import io
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SHELL_FILES = ("scripts/setup_env.sh",)


@unittest.skipUnless(shutil.which("git") and (REPO_ROOT / ".git").exists(), "Git checkout required")
class ShellArchiveTests(unittest.TestCase):
    def archive(self):
        # Worktree attributes also exercise the policy before its first commit.
        raw = subprocess.check_output(
            [
                "git", "-c", "core.autocrlf=true", "-c", "core.eol=crlf",
                "archive", "--worktree-attributes", "--format=tar", "HEAD",
                *SMOKE_SHELL_FILES,
            ],
            cwd=REPO_ROOT,
            timeout=30,
        )
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            return {
                name: archive.extractfile(name).read()
                for name in SMOKE_SHELL_FILES
            }

    def test_linux_bootstrap_archive_matches_lf_committed_blob(self):
        for name, data in self.archive().items():
            with self.subTest(path=name):
                blob = subprocess.check_output(
                    ["git", "show", f"HEAD:{name}"], cwd=REPO_ROOT, timeout=30
                )
                self.assertNotIn(b"\r", blob)
                self.assertNotIn(b"\r", data)
                self.assertEqual(data, blob)

    @unittest.skipUnless(shutil.which("bash"), "Bash syntax check requires Bash")
    def test_delivered_shell_archive_bash_syntax(self):
        for name, data in self.archive().items():
            with self.subTest(path=name):
                # A file argument also works with Windows Bash pipe emulation.
                with tempfile.TemporaryDirectory() as tmp:
                    (Path(tmp) / "bootstrap.sh").write_bytes(data)
                    result = subprocess.run(
                        [shutil.which("bash"), "-n", "bootstrap.sh"],
                        capture_output=True, cwd=tmp, timeout=30,
                    )
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
