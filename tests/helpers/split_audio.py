

import sys
from pathlib import Path

import ffmpeg
import pytest

# Add project root to Python path (2 levels up)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from helpers import split_audio


def test_split_audio_default():
    audioChunks:list[Path] | Exception = split_audio(Path("./tests/sources/test_audio_split.mp3"))

    assert isinstance(audioChunks, list)
    assert all(chunk.exists() for chunk in audioChunks)
    assert len(audioChunks) == 2

    totalDuration = 0
    for chunk in audioChunks:
        probe = ffmpeg.probe(chunk)
        totalDuration += float(probe['format']['duration'])

    assert totalDuration >= 1580

def test_split_audio_5_min_chunks():
    audioChunks:list[Path] | Exception = split_audio(Path("./tests/sources/test_audio_split.mp3"), 5)

    assert isinstance(audioChunks, list)
    assert all(chunk.exists() for chunk in audioChunks)
    assert len(audioChunks) == 6

    totalDuration = 0
    for chunk in audioChunks:
        probe = ffmpeg.probe(chunk)
        totalDuration += float(probe['format']['duration'])

    assert totalDuration >= 1580

def test_split_audio_file_not_found():
    with pytest.raises(FileNotFoundError):
        split_audio(Path("./invalid_file.mp3"))
