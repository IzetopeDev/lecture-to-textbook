import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import transcribe
from settings import PrjPaths


def test_batch_transcribe_2_files():
    """
    Runs the entire transcribe.py workflow without mocking any components.
    Verifies that audio files in PrjPaths.audio_files produce transcript files
    in PrjPaths.transcripts_raw.
    """
    # 1. Run the main function in transcribe.py
    transcribe.main()

    # 2. Get the list of input audio files
    audio_files = [
        f for f in os.listdir(PrjPaths.audio_files) if not f.startswith(".")
    ]
    assert len(audio_files) > 0, f"No audio files found in {PrjPaths.audio_files}"

    # 3. Verify that a transcript output file exists and is non-empty for each audio file
    for audio_file in audio_files:
        qualifier = audio_file.split(".")[0]
        expected_transcript = PrjPaths.transcripts_raw / f"output-{qualifier}.txt"

        assert expected_transcript.exists(), (
            f"Expected transcript file '{expected_transcript.name}' "
            f"was not created in {PrjPaths.transcripts_raw}"
        )
        assert expected_transcript.stat().st_size > 0, (
            f"Transcript file '{expected_transcript.name}' is empty"
        )