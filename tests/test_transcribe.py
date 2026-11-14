import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import transcribe
from settings import PrjPaths

# =====================================================================
# Tests: Model Loading & CPU Fallback
# =====================================================================

def test_load_whisper_cuda_fallback():
    """Verify that if CUDA loading fails with a RuntimeError, it falls back to CPU."""
    with patch("whisper.load_model") as mock_load, patch("torch.cuda.is_available", return_value=True):
        cpu_model = MagicMock()
        # First call on CUDA fails; second call on CPU succeeds
        mock_load.side_effect = [RuntimeError("CUDA initialization error"), cpu_model]

        model = transcribe._load_whisper(model_name="tiny", device="cuda")

        assert model == cpu_model
        assert mock_load.call_count == 2
        mock_load.assert_called_with("tiny", device="cpu")


def test_load_whisper_invalid_model():
    """Verify that attempting to load an unsupported model name raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not found"):
        transcribe._load_whisper(model_name="nonexistent_model_xyz")


# =====================================================================
# Tests: Single Audio Transcription
# =====================================================================

def test_transcribe_missing_audio_file():
    """Verify that passing a non-existent audio file raises FileNotFoundError."""
    mock_model = MagicMock()
    missing_file = Path("non_existent_audio.mp3")

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcribe.transcribe(mock_model, missing_file)


def test_transcribe_success(tmp_path):
    """Verify that transcribe creates the output file and returns the expected dictionary."""
    mock_audio = tmp_path / "sample.mp3"
    mock_audio.write_bytes(b"dummy audio content")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Hello world from Whisper"}

    mock_transcripts_raw = tmp_path / "transcripts"
    mock_transcripts_raw.mkdir(parents=True, exist_ok=True)

    with patch.object(PrjPaths, "TRANSCRIPTS_RAW", mock_transcripts_raw):
        result_dict = transcribe.transcribe(mock_model, mock_audio)

        assert "result" in result_dict
        assert "output_file" in result_dict
        output_file = result_dict["output_file"]
        assert isinstance(output_file, Path)
        assert output_file.is_file()

        saved_text = output_file.read_text(encoding="utf-8")
        assert saved_text == "Hello world from Whisper"


def test_transcribe_cuda_oom_fallback(tmp_path):
    """Verify that if CUDA runs out of memory during transcribe(), it falls back to CPU."""
    mock_audio = tmp_path / "speech.mp3"
    mock_audio.write_bytes(b"dummy audio content")

    mock_model = MagicMock()
    mock_model.device = torch.device("cuda:0")

    # First call triggers CUDA OOM; retry on CPU succeeds
    mock_model.transcribe.side_effect = [
        torch.cuda.OutOfMemoryError("CUDA out of memory"),
        {"text": "Transcription after CPU fallback"},
    ]

    mock_transcripts_raw = tmp_path / "transcripts"
    mock_transcripts_raw.mkdir(parents=True, exist_ok=True)

    with patch.object(PrjPaths, "TRANSCRIPTS_RAW", mock_transcripts_raw):
        result_dict = transcribe.transcribe(mock_model, mock_audio)

        assert mock_model.to.called
        mock_model.to.assert_called_with("cpu")
        output_file = result_dict["output_file"]
        assert isinstance(output_file, Path)
        assert output_file.read_text() == "Transcription after CPU fallback"


# =====================================================================
# Tests: Batch Transcription
# =====================================================================

def test_batch_transcribe_folder_not_found(tmp_path):
    """Verify that batch_transcribe raises FileNotFoundError if input folder does not exist."""
    non_existent = tmp_path / "missing_folder"
    with pytest.raises(FileNotFoundError):
        transcribe.batch_transcribe(non_existent)


def test_batch_transcribe_empty_folder(tmp_path):
    """Verify that batch_transcribe raises FileNotFoundError if input folder has no recordings."""
    empty_folder = tmp_path / "empty_audio"
    empty_folder.mkdir()

    with pytest.raises(FileNotFoundError):
        transcribe.batch_transcribe(empty_folder)


def test_batch_transcribe_skips_failed_file_and_continues(tmp_path):
    """
    Verify that if one file encounters a processing error,
    it is logged/skipped and remaining recordings are still processed.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    file_1 = audio_dir / "bad_recording.mp3"
    file_1.write_bytes(b"bad")

    file_2 = audio_dir / "good_recording.mp3"
    file_2.write_bytes(b"good")

    with patch("transcribe._load_whisper") as mock_load, patch("transcribe.transcribe") as mock_transcribe:
        mock_load.return_value = MagicMock()

        # file_1 fails with corrupt audio; file_2 succeeds
        mock_transcribe.side_effect = [
            RuntimeError("Corrupt audio header"),
            {"result": {"text": "ok"}, "output_file": tmp_path / "good.txt"},
        ]

        # Should complete without crashing
        transcribe.batch_transcribe(audio_dir)

        assert mock_transcribe.call_count == 2


def test_batch_transcribe_ffmpeg_missing_aborts(tmp_path):
    """Verify that if ffmpeg is missing from PATH, batch_transcribe immediately aborts."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "audio.mp3").write_bytes(b"data")

    with patch("transcribe._load_whisper") as mock_load, patch("transcribe.transcribe") as mock_transcribe:
        mock_load.return_value = MagicMock()
        mock_transcribe.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'ffmpeg'")

        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            transcribe.batch_transcribe(audio_dir)


def test_batch_transcribe_integration(tmp_path):
    """End-to-end integration test verifying file outputs match recording stem names."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "lecture1.mp3").write_bytes(b"audio1")
    (audio_dir / "lecture2.wav").write_bytes(b"audio2")

    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Integrated transcript text"}

    with (
        patch("transcribe._load_whisper", return_value=mock_model),
        patch.object(PrjPaths, "TRANSCRIPTS_RAW", transcripts_dir),
    ):
        transcribe.batch_transcribe(audio_dir)

        expected_1 = transcripts_dir / "lecture1.txt"
        expected_2 = transcripts_dir / "lecture2.txt"

        assert expected_1.exists()
        assert expected_2.exists()
        assert expected_1.read_text() == "Integrated transcript text"
        assert expected_2.read_text() == "Integrated transcript text"
