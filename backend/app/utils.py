"""
Resource management utilities for safe file and directory operations.
"""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union


class ResourceManager:
    """Manager for temporary files and directories with automatic cleanup."""

    @staticmethod
    @contextmanager
    def temporary_directory(
        base_dir: Optional[Union[str, Path]] = None, prefix: str = "music_detector_"
    ) -> Generator[Path, None, None]:
        """
        Context manager for temporary directory with guaranteed cleanup.

        Args:
            base_dir: Base directory for temp folder (None = system default)
            prefix: Prefix for temp directory name

        Yields:
            Path to temporary directory

        Example:
            with ResourceManager.temporary_directory() as temp_dir:
                # use temp_dir
                pass
            # temp_dir is automatically cleaned up
        """
        temp_dir = None
        try:
            if base_dir:
                base_dir = Path(base_dir)
                base_dir.mkdir(parents=True, exist_ok=True)
                temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=base_dir))
            else:
                temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
            yield temp_dir
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    @contextmanager
    def temporary_file(
        suffix: str = "", base_dir: Optional[Union[str, Path]] = None
    ) -> Generator[Path, None, None]:
        """
        Context manager for temporary file with guaranteed cleanup.

        Args:
            suffix: File suffix/extension
            base_dir: Base directory for temp file (None = system default)

        Yields:
            Path to temporary file
        """
        temp_file = None
        try:
            if base_dir:
                base_dir = Path(base_dir)
                base_dir.mkdir(parents=True, exist_ok=True)
                fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=base_dir)
            else:
                fd, temp_path = tempfile.mkstemp(suffix=suffix)

            # Close file descriptor immediately
            import os

            os.close(fd)
            temp_file = Path(temp_path)
            yield temp_file
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)

    @staticmethod
    def safe_remove_directory(path: Union[str, Path], ignore_errors: bool = True) -> bool:
        """
        Safely remove a directory and all contents.

        Args:
            path: Directory path
            ignore_errors: Whether to ignore errors during removal

        Returns:
            True if successful, False otherwise
        """
        path = Path(path)
        if not path.exists():
            return True

        try:
            shutil.rmtree(path, ignore_errors=ignore_errors)
            return True
        except Exception:
            return False

    @staticmethod
    def safe_remove_file(path: Union[str, Path]) -> bool:
        """
        Safely remove a file.

        Args:
            path: File path

        Returns:
            True if successful, False otherwise
        """
        path = Path(path)
        try:
            path.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    @staticmethod
    def ensure_directory(path: Union[str, Path], parents: bool = True) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path
            parents: Create parent directories if needed

        Returns:
            Path object to directory
        """
        path = Path(path)
        path.mkdir(parents=parents, exist_ok=True)
        return path

    @staticmethod
    def get_file_size_mb(path: Union[str, Path]) -> float:
        """
        Get file size in megabytes.

        Args:
            path: File path

        Returns:
            File size in MB
        """
        path = Path(path)
        if not path.exists():
            return 0.0
        return path.stat().st_size / (1024 * 1024)

    @staticmethod
    def copy_file_safe(source: Union[str, Path], dest: Union[str, Path]) -> bool:
        """
        Safely copy a file with error handling.

        Args:
            source: Source file path
            dest: Destination file path

        Returns:
            True if successful, False otherwise
        """
        try:
            source = Path(source)
            dest = Path(dest)

            if not source.exists():
                return False

            # Ensure destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source, dest)
            return True
        except Exception:
            return False
