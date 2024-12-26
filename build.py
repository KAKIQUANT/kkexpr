"""Build script for compiling Rust extensions."""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import stat
import tempfile
import time

def build_rust_extension():
    """Build the Rust extension module."""
    rust_dir = Path("src/rust_expr")
    
    # Set Python environment variables
    python_path = sys.executable
    os.environ["PYO3_PYTHON"] = python_path
    os.environ["PYTHON_SYS_EXECUTABLE"] = python_path
    
    # Check if cargo is available
    try:
        subprocess.run(["cargo", "--version"], check=True)
    except subprocess.CalledProcessError:
        print("Error: Cargo not found. Please install Rust and Cargo.")
        sys.exit(1)
    
    # Build the extension
    print(f"Building Rust extension using Python: {python_path}")
    try:
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_dir,
            check=True,
            env=os.environ
        )
    except subprocess.CalledProcessError as e:
        print(f"Error building Rust extension: {e}")
        sys.exit(1)
    
    # Copy the built library to the correct locations
    lib_name = "rust_expr"
    if sys.platform == "win32":
        lib_ext = ".pyd"  # Use .pyd for Windows Python extensions
        src = rust_dir / "target" / "release" / f"{lib_name}.dll"
    elif sys.platform == "darwin":
        lib_ext = ".so"  # Use .so for macOS
        src = rust_dir / "target" / "release" / f"lib{lib_name}.dylib"
    else:
        lib_ext = ".so"  # Use .so for Linux
        src = rust_dir / "target" / "release" / f"lib{lib_name}.so"
    
    # Create package directory
    pkg_dir = Path("src/datafeed/rust_expr")
    pkg_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("""
from .rust_expr import momentum_factor, mean_reversion_factor, relative_strength_factor, alpha101_factor_42

__all__ = [
    'momentum_factor',
    'mean_reversion_factor',
    'relative_strength_factor',
    'alpha101_factor_42'
]
""")
    
    # Copy library to package directory with correct extension
    dst = pkg_dir / f"rust_expr{lib_ext}"
    print(f"Copying {src} to {dst}")
    
    # Try multiple times with increasing delays
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if dst.exists():
                try:
                    # Try to rename the file first (this will fail if file is in use)
                    temp_name = dst.with_suffix('.old')
                    dst.rename(temp_name)
                    temp_name.unlink()  # Delete the old file
                except Exception as e:
                    print(f"Warning: Could not remove old file on attempt {attempt + 1}: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise
            
            # Copy the new file
            shutil.copy2(src, dst)
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Error copying library after {max_attempts} attempts: {e}")
                sys.exit(1)
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1 * (attempt + 1))  # Exponential backoff
    
    print("Build completed successfully!")

if __name__ == "__main__":
    build_rust_extension() 