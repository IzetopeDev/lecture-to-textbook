import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from helpers import write_to_file


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for each test"""
    return tmp_path / "test_output"

def test_write_basic_file(temp_dir):
    """Test basic file writing"""
    result = write_to_file("Hello World", temp_dir)
    
    assert result.exists()
    assert result.name == "output.txt"
    assert result.read_text() == "Hello World"

def test_write_with_qualifier(temp_dir):
    """Test file writing with qualifier"""
    result = write_to_file("Test content", temp_dir, qualifier="test123")
    
    assert result.exists()
    assert result.name == "output-test123.txt"
    assert result.read_text() == "Test content"

def test_write_without_qualifier(temp_dir):
    """Test file writing without qualifier (None)"""
    result = write_to_file("Test content", temp_dir, qualifier=None)
    
    assert result.name == "output.txt"

def test_append_mode_default(temp_dir):
    """Test that default mode is append"""
    write_to_file("First line\n", temp_dir, qualifier="append")
    result = write_to_file("Second line\n", temp_dir, qualifier="append")
    
    content = result.read_text()
    assert content == "First line\nSecond line\n"

def test_overwrite_mode(temp_dir):
    """Test overwrite mode"""
    write_to_file("Original content", temp_dir, qualifier="overwrite")
    result = write_to_file(
        "New content", 
        temp_dir, 
        qualifier="overwrite",
        overwrite=True
    )
    
    content = result.read_text()
    assert content == "New content"
    assert "Original" not in content

def test_creates_directory_if_not_exists(tmp_path):
    """Test that directory is created if it doesn't exist"""
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    
    result = write_to_file("Test", nested_dir)
    
    assert nested_dir.exists()
    assert result.exists()

def test_returns_correct_path(temp_dir):
    """Test that function returns the correct Path object"""
    result = write_to_file("Test", temp_dir, qualifier="xyz")
    
    assert isinstance(result, Path)
    assert result == temp_dir / "output-xyz.txt"

def test_unicode_content(temp_dir):
    """Test writing unicode content"""
    unicode_text = "Hello 世界 🌍"
    result = write_to_file(unicode_text, temp_dir)
    
    assert result.read_text(encoding="utf-8") == unicode_text

def test_oserror_handling(tmp_path):
    """Test OSError handling"""
    # Create a file instead of a directory to force an error
    bad_path = tmp_path / "file.txt"
    bad_path.write_text("I'm a file, not a directory")
    
    # Try to use it as a directory
    with pytest.raises(OSError):
        write_to_file("Test", bad_path / "subdir")

def test_oserror_logging(tmp_path, caplog):
    """Test that OSError is logged"""
    bad_path = tmp_path / "file.txt"
    bad_path.write_text("blocking")
    
    with caplog.at_level(logging.ERROR), pytest.raises(OSError):
        write_to_file("Test", bad_path / "subdir", qualifier="error_test")
    
    assert "Failed writing output file" in caplog.text
    assert "qualifier=error_test" in caplog.text

def test_empty_string(temp_dir):
    """Test writing empty string"""
    result = write_to_file("", temp_dir)
    
    assert result.exists()
    assert result.read_text() == ""

def test_multiline_content(temp_dir):
    """Test writing multiline content"""
    content = "Line 1\nLine 2\nLine 3"
    result = write_to_file(content, temp_dir)
    
    assert result.read_text() == content
