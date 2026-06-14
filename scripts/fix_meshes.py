"""
Fix mesh files for MuJoCo compatibility:
1. Convert ASCII STL files to binary STL
2. Decimate meshes with too many faces (MuJoCo limit: 200000)

Usage:
    python scripts/fix_meshes.py
"""

import os
import sys
import numpy as np

MESHES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "meshes"
)
MAX_FACES = 190000  # slightly below MuJoCo's 200000 limit


def is_ascii_stl(filepath: str) -> bool:
    """Check if an STL file is ASCII format."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(80)
            # ASCII STL starts with 'solid'
            if header[:5] == b"solid":
                # But binary STL can also start with 'solid' in header
                # Check if next non-whitespace after header is 'facet' or 'endsolid'
                f.seek(0)
                first_line = f.readline().decode("ascii", errors="ignore").strip()
                second_line = f.readline().decode("ascii", errors="ignore").strip()
                if second_line.startswith("facet") or second_line.startswith("endsolid"):
                    return True
        return False
    except Exception:
        return False


def get_binary_face_count(filepath: str) -> int:
    """Get face count from a binary STL file."""
    try:
        with open(filepath, "rb") as f:
            f.seek(80)  # skip header
            count_bytes = f.read(4)
            if len(count_bytes) == 4:
                return int.from_bytes(count_bytes, byteorder="little")
    except Exception:
        pass
    return -1


def convert_and_fix(filepath: str) -> bool:
    """Convert ASCII STL to binary and/or decimate if needed."""
    from stl import mesh as stl_mesh

    try:
        m = stl_mesh.Mesh.from_file(filepath)
        n_faces = len(m.vectors)
        changed = False
        was_ascii = is_ascii_stl(filepath)

        if was_ascii:
            print(f"  Converting ASCII -> binary: {os.path.basename(filepath)} ({n_faces} faces)")
            changed = True

        if n_faces > MAX_FACES:
            # Simple decimation: uniformly sample faces
            indices = np.linspace(0, n_faces - 1, MAX_FACES, dtype=int)
            new_data = np.zeros(MAX_FACES, dtype=m.data.dtype)
            for idx_new, idx_old in enumerate(indices):
                new_data[idx_new] = m.data[idx_old]
            new_mesh = stl_mesh.Mesh(new_data)
            new_mesh.save(filepath)
            print(f"  Decimated: {os.path.basename(filepath)} {n_faces} -> {MAX_FACES} faces")
            return True

        if changed:
            # Save as binary STL (overwrite)
            m.save(filepath)
            return True

        return False
    except Exception as e:
        print(f"  ERROR processing {os.path.basename(filepath)}: {e}")
        return False


def main():
    print("Fixing mesh files for MuJoCo compatibility...")
    print(f"Meshes directory: {MESHES_DIR}\n")

    if not os.path.isdir(MESHES_DIR):
        print(f"ERROR: Directory not found: {MESHES_DIR}")
        sys.exit(1)

    stl_files = [
        f for f in os.listdir(MESHES_DIR)
        if f.lower().endswith(".stl") and not f.endswith(".convex.stl")
    ]
    stl_files.sort()

    fixed = 0
    for fname in stl_files:
        fpath = os.path.join(MESHES_DIR, fname)

        needs_fix = False
        if is_ascii_stl(fpath):
            needs_fix = True
        else:
            fc = get_binary_face_count(fpath)
            if fc > MAX_FACES:
                needs_fix = True

        if needs_fix:
            if convert_and_fix(fpath):
                fixed += 1

    print(f"\nFixed {fixed} mesh files.")

    # Also report any remaining large meshes
    print("\nMesh face counts:")
    for fname in stl_files:
        fpath = os.path.join(MESHES_DIR, fname)
        fc = get_binary_face_count(fpath)
        flag = " *** TOO LARGE ***" if fc > 200000 else ""
        print(f"  {fname}: {fc} faces{flag}")


if __name__ == "__main__":
    main()
