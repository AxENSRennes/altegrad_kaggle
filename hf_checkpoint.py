"""Hugging Face Hub utilities for checkpoint management."""
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Default repo
DEFAULT_REPO = "Moinada/altegrad-mol-caption"


def get_api():
    """Get HfApi instance with token from environment."""
    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token)


def upload_checkpoint(
    local_path: str,
    repo_id: str = DEFAULT_REPO,
    path_in_repo: str = None,
    keep_path: bool = False,
):
    """Upload a checkpoint to HF Hub.

    Args:
        local_path: Path to local checkpoint file
        repo_id: HF Hub repository ID (e.g., "username/repo-name")
        path_in_repo: Filename in the repo (defaults to local path or filename)
        keep_path: If True, preserve directory structure (e.g., data/file.pkl)
    """
    api = get_api()
    local_path = Path(local_path)
    if path_in_repo is None:
        path_in_repo = str(local_path) if keep_path else local_path.name

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {local_path} -> {repo_id}/{path_in_repo}")


def download_checkpoint(
    filename: str,
    local_dir: str = ".",
    repo_id: str = DEFAULT_REPO,
) -> str:
    """Download a checkpoint from HF Hub.

    Args:
        filename: Name of the file in the HF repo
        local_dir: Local directory to download to
        repo_id: HF Hub repository ID

    Returns:
        Path to the downloaded file
    """
    token = os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        repo_type="model",
        token=token,
    )
    print(f"Downloaded {repo_id}/{filename} -> {path}")
    return path


def list_checkpoints(repo_id: str = DEFAULT_REPO) -> list:
    """List all checkpoint files in the HF repo.

    Args:
        repo_id: HF Hub repository ID

    Returns:
        List of .pt filenames in the repo
    """
    token = os.environ.get("HF_TOKEN")
    files = list_repo_files(repo_id, repo_type="model", token=token)
    return [f for f in files if f.endswith(".pt")]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HF Checkpoint Manager")
    parser.add_argument("action", choices=["upload", "download", "list"])
    parser.add_argument(
        "--file", "-f", help="Local file to upload or remote file to download"
    )
    parser.add_argument("--repo", "-r", default=DEFAULT_REPO, help="HF repo ID")
    parser.add_argument(
        "--local-dir", "-d", default=".", help="Local directory for downloads"
    )
    parser.add_argument(
        "--keep-path", "-k", action="store_true",
        help="Preserve directory structure in repo (e.g., data/file.pkl)"
    )
    args = parser.parse_args()

    if args.action == "upload":
        if not args.file:
            parser.error("upload requires --file")
        upload_checkpoint(args.file, args.repo, keep_path=args.keep_path)
    elif args.action == "download":
        if not args.file:
            parser.error("download requires --file")
        download_checkpoint(args.file, local_dir=args.local_dir, repo_id=args.repo)
    elif args.action == "list":
        checkpoints = list_checkpoints(args.repo)
        if checkpoints:
            for f in checkpoints:
                print(f)
        else:
            print("No checkpoints found")
