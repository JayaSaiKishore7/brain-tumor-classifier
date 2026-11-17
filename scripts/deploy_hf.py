"""
Deploy the serving app to the Hugging Face Space.

    python scripts/deploy_hf.py --dry-run   # show what would be uploaded/deleted
    python scripts/deploy_hf.py             # deploy (needs `hf auth login` first)

Safety rails:
  - Only deploys committed code: refuses if the git working tree has changes.
  - Runs `ruff check` and `pytest` first (the same gate as CI); refuses on failure.
  - Uploads exactly the serving files (Dockerfile, requirements.txt, model/, web/,
    plus deploy/huggingface/README.md as the Space's README.md) in one atomic commit,
    and deletes anything else in the Space, so the Space mirrors one git commit.
  - The Space commit message records the source git commit, for rollback.

Rollback: check out the last good commit and run this script again.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("HF_SPACE_ID", "JayaSaiKishore/brain-tumor-classifier")
KEEP_IN_SPACE = {".gitattributes"}  # Space-managed; never deleted


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def git(*args):
    r = run(["git", *args])
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def collect_files():
    """Map Space path -> local path, using only git-tracked files."""
    files = {"README.md": os.path.join(ROOT, "deploy", "huggingface", "README.md")}
    tracked = git("ls-files", "Dockerfile", "requirements.txt", "model", "web").splitlines()
    for rel in tracked:
        files[rel] = os.path.join(ROOT, rel)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list changes, upload nothing")
    parser.add_argument("--skip-checks", action="store_true", help="skip ruff + pytest")
    args = parser.parse_args()

    if git("status", "--porcelain") and not args.dry_run:
        sys.exit(
            "Working tree has uncommitted changes. Commit first; deploys must map to a commit."
        )
    sha = git("rev-parse", "--short", "HEAD")

    if not args.skip_checks and not args.dry_run:
        for cmd in (
            [sys.executable, "-m", "ruff", "check", "."],
            [sys.executable, "-m", "pytest", "-q"],
        ):
            print("$", " ".join(cmd[2:]))
            r = subprocess.run(cmd, cwd=ROOT, check=False)
            if r.returncode != 0:
                sys.exit("Checks failed; not deploying.")

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    files = collect_files()
    existing = set(api.list_repo_files(SPACE_ID, repo_type="space"))
    stale = sorted(existing - set(files) - KEEP_IN_SPACE)

    print(f"Space: {SPACE_ID}   source commit: {sha}")
    print(f"Upload {len(files)} files ({len(set(files) - existing)} new):")
    for path in sorted(files):
        print("  +" if path not in existing else "  ~", path)
    for path in stale:
        print("  -", path)

    if args.dry_run:
        print("Dry run: nothing uploaded.")
        return

    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=src) for p, src in files.items()]
    ops += [CommitOperationDelete(path_in_repo=p) for p in stale]
    info = api.create_commit(
        SPACE_ID,
        repo_type="space",
        operations=ops,
        commit_message=f"Deploy {sha} from GitHub main",
    )
    print("Deployed:", info.commit_url)
    print(f"Watch the build: https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
