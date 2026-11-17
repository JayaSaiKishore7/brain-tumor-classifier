"""
Deploy the static site to the Hugging Face Space (a free *static* Space: the model runs
in the visitor's browser, so there is no server to pay for).

    python scripts/deploy_hf.py --dry-run   # build + show what would change; upload nothing
    python scripts/deploy_hf.py             # deploy (needs `hf auth login` first)
    python scripts/deploy_hf.py --create    # first deploy: also create the Space if missing

Safety rails:
  - Only deploys committed code: refuses if the git working tree has changes.
  - Runs `ruff check` and `pytest` first (the same gate as CI); refuses on failure.
  - Builds build/static_site/ fresh from the commit, then uploads it in one atomic
    commit and deletes anything else in the Space, so the Space mirrors one git commit.
  - Refuses a Space that isn't static: Docker/Gradio Spaces need a paid plan on free
    accounts and would upload fine but never run.
  - Only creates a Space when --create is passed, and only as a free static Space.
  - The Space commit message records the source git commit, for rollback.
  - After uploading, runs smoke_check.py and browser_check.py (a real in-browser
    prediction) against the live site, and prints rollback steps if either fails.

Rollback: check out the last good commit, run this script again, check out main.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_ID = os.environ.get("HF_SPACE_ID", "JayaSaiKishore/brain-tumor-classifier")
KEEP_IN_SPACE = {".gitattributes"}  # Space-managed; never deleted

sys.path.insert(0, os.path.join(ROOT, "scripts"))


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def collect_files(site_dir):
    """Map Space path -> local path for everything in the build."""
    files = {}
    for dirpath, _, names in os.walk(site_dir):
        for name in names:
            local = os.path.join(dirpath, name)
            files[os.path.relpath(local, site_dir).replace(os.sep, "/")] = local
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list changes, upload nothing")
    parser.add_argument("--create", action="store_true", help="create the static Space if missing")
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
            if subprocess.run(cmd, cwd=ROOT, check=False).returncode != 0:
                sys.exit("Checks failed; not deploying.")

    import build_static
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    site_dir = build_static.main()
    files = collect_files(site_dir)
    api = HfApi()

    try:
        info = api.space_info(SPACE_ID)
    except RepositoryNotFoundError:
        info = None

    if info is None:
        if not args.create:
            sys.exit(f"Space {SPACE_ID} doesn't exist. Re-run with --create to create it.")
        existing = set()
        if args.dry_run:
            print(f"Would create free static Space {SPACE_ID}.")
        else:
            api.create_repo(SPACE_ID, repo_type="space", space_sdk="static", private=False)
            print(f"Created static Space {SPACE_ID}")
    else:
        if info.sdk != "static":
            sys.exit(
                f"Space {SPACE_ID} uses sdk={info.sdk!r}, not 'static'. Docker/Gradio Spaces "
                "need a paid plan on free accounts. Recreate it as a static Space."
            )
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
    commit = api.create_commit(
        SPACE_ID,
        repo_type="space",
        operations=ops,
        commit_message=f"Deploy {sha} from GitHub main",
    )
    print("Deployed:", commit.commit_url)
    print(f"Space page: https://huggingface.co/spaces/{SPACE_ID}")

    # Post-deploy verification: the live site, checked the way a visitor uses it.
    for script in ("smoke_check.py", "browser_check.py"):
        print(f"$ {script}")
        cmd = [sys.executable, os.path.join(ROOT, "scripts", script)]
        if subprocess.run(cmd, cwd=ROOT, check=False).returncode != 0:
            sys.exit(
                f"POST-DEPLOY CHECK FAILED ({script}). Roll back: git checkout <last good "
                "commit>; python scripts/deploy_hf.py --skip-checks; git checkout main"
            )
    print("Deploy verified.")


if __name__ == "__main__":
    main()
