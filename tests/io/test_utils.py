import subprocess

import pytest

import susi.io.utils as io_utils

# %% get_git_revision_short_hash
#
# The hash goes into every run's metadata.json as the record of which SUSI
# code produced that run. It must therefore describe the *checkout*, not
# wherever the user happened to launch python from -- projects live outside
# the repo (SUSI_PROJECTS_ROOT) and are often git repositories of their own.


@pytest.fixture
def a_git_repo_that_is_not_susi(tmp_path):
    """A committed git repository standing in for a user's project folder."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "susi_calls.py").write_text("# my project", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-qm",
            "my project",
        ],
        check=True,
    )
    return tmp_path


def test_git_revision_in_the_checkout_is_its_short_hash():
    # The other tests compare hashes with each other, which a function that
    # always fell back would pass too; this pins the checkout's own answer.
    checkout_head = (
        subprocess.check_output(
            ["git", "-C", str(io_utils.repo_root()), "rev-parse", "--short", "HEAD"]
        )
        .decode("ascii")
        .strip()
    )

    assert io_utils.get_git_revision_short_hash() == checkout_head


def test_git_revision_is_the_same_wherever_the_process_was_started(
    tmp_path, monkeypatch
):
    from_the_checkout = io_utils.get_git_revision_short_hash()

    # A folder in no repository at all: this used to raise CalledProcessError
    # and take the whole run down with it before the simulation started.
    monkeypatch.chdir(tmp_path)

    assert io_utils.get_git_revision_short_hash() == from_the_checkout


def test_git_revision_ignores_the_repository_the_run_was_started_in(
    a_git_repo_that_is_not_susi, monkeypatch
):
    # The damaging case: a project folder made into a repository of its own,
    # which is what a team sharing a project is told to do. The recorded
    # hash used to be that project's commit -- a hash that does not exist in
    # the SUSI repo, in a field whose whole purpose is to name SUSI's code.
    project_commit = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(a_git_repo_that_is_not_susi),
                "rev-parse",
                "--short",
                "HEAD",
            ]
        )
        .decode("ascii")
        .strip()
    )
    monkeypatch.chdir(a_git_repo_that_is_not_susi)

    assert io_utils.get_git_revision_short_hash() != project_commit


# An installed (non-editable) SUSI has no checkout: `repo_root()` is
# `.../lib/python3.11`, and git there either fails -- which used to crash
# every run before it started -- or finds whatever repository the venv sits
# in and reports *that* commit. Either way the hash cannot be had, so the
# run records a sentence that says so, and carries on.


def _is_the_fallback(git_commit_hash: str) -> bool:
    return git_commit_hash.startswith(
        "could not get git commit hash. The package version was: "
    )


def test_git_revision_falls_back_without_asking_git_outside_a_checkout(
    a_git_repo_that_is_not_susi, monkeypatch, capsys
):
    # No pyproject.toml, so not a checkout. The folder *is* a repository,
    # so had git been asked it would have answered with this repository's
    # commit: the silently-wrong case of a venv inside a user's project.
    monkeypatch.setattr(io_utils, "repo_root", lambda: a_git_repo_that_is_not_susi)

    assert _is_the_fallback(io_utils.get_git_revision_short_hash())
    assert "could not get git commit hash" in capsys.readouterr().err


def test_git_revision_falls_back_when_git_fails(tmp_path, monkeypatch, capsys):
    # Looks like a checkout, but is no repository: git exits non-zero.
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(io_utils, "repo_root", lambda: tmp_path)
    # Stop git searching upward, in case tmp_path sits inside a repository.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    assert _is_the_fallback(io_utils.get_git_revision_short_hash())
    assert "could not get git commit hash" in capsys.readouterr().err


def test_git_revision_falls_back_when_git_is_not_installed(monkeypatch, capsys):
    # The real checkout, but no git binary to ask.
    monkeypatch.setenv("PATH", "")

    assert _is_the_fallback(io_utils.get_git_revision_short_hash())
    assert "could not get git commit hash" in capsys.readouterr().err
