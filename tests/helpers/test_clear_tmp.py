import logging
import sys
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from helpers import PrjPaths, clear_tmp


def test_clear_tmp_normal(tmp_path):
    """
    Test 1: Normal clear_tmp operation.
    Populate a tmp directory with files and folders, then run clear_tmp.
    Assert folders remained intact while files were removed.
    """
    mock_tmp = tmp_path / "tmp"
    mock_tmp.mkdir(parents=True, exist_ok=True)

    with patch.object(PrjPaths, "TMP", mock_tmp):
        # Store before populating state
        initial_state = set(mock_tmp.iterdir())
        assert len(initial_state) == 0

        # Create folders and random files in them / in mock_tmp
        folder1 = mock_tmp / "folder1"
        folder1.mkdir()
        nested_file = folder1 / "nested_file.txt"
        nested_file.write_text("nested content")

        folder2 = mock_tmp / "folder2"
        folder2.mkdir()

        file1 = mock_tmp / "random_file_1.txt"
        file1.write_text("random content 1")

        file2 = mock_tmp / "random_file_2.tmp"
        file2.write_text("random content 2")

        # Store after populating state
        populated_state = set(mock_tmp.iterdir())

        # Note the diff between populating and initial state
        diff_state = populated_state - initial_state
        assert len(diff_state) == 4

        # Run clear_tmp
        clear_tmp(mock_tmp)

        # Assert folders remained
        assert folder1.exists() and folder1.is_dir()
        assert folder2.exists() and folder2.is_dir()
        assert nested_file.exists()

        # Assert files removed
        assert not file1.exists()
        assert not file2.exists()


def test_clear_tmp_wrong_path(tmp_path, caplog):
    """
    Test 2: Give the wrong path to clear_tmp.
    Run clear_tmp on a directory outside PrjPaths.TMP.
    Assert warning was logged and no changes occurred in wrong dir.
    """
    mock_tmp = tmp_path / "tmp"
    mock_tmp.mkdir(parents=True, exist_ok=True)

    wrong_dir = tmp_path / "wrong_dir"
    wrong_dir.mkdir(parents=True, exist_ok=True)

    # Create mock dir structure inside wrong_dir
    sub_folder = wrong_dir / "important_folder"
    sub_folder.mkdir()

    wrong_file = wrong_dir / "important_file.txt"
    wrong_file.write_text("Should not be removed")

    # Store state before running clear_tmp
    state_before = set(wrong_dir.iterdir())
    assert len(state_before) == 2

    with patch.object(PrjPaths, "TMP", mock_tmp), caplog.at_level(logging.WARNING):
        clear_tmp(wrong_dir)

    # Assert warn successful
    assert "Unable to clear tmp folder. Invalid path passed:" in caplog.text
    assert "Operation Skipped" in caplog.text

    # Assert no changes in wrong dir
    state_after = set(wrong_dir.iterdir())
    assert state_after == state_before
    assert wrong_file.exists()
    assert sub_folder.exists()
