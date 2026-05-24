# download_backbone.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/C-RADIOv4-H",
    local_dir="./checkpoint_radiov4",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.pth.tar", "README.md", ".gitattributes"] # Skips unneeded bulk files!
)
print("Download complete!")