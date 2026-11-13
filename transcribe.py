import os
import time
from pathlib import Path

import torch
import whisper

from helpers import DLOG, split_audio, write_to_file
from settings import PrjModels, PrjPaths


def load_model() -> whisper.model.Whisper:
    startTime = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DLOG.info(
        f"Loading Whisper model '{PrjModels.TRANSCRIPTION}' on device '{device}'..."
    )
    if device == "cpu":
        DLOG.warning("Loading on CPU!")

    try:
        model = whisper.load_model(PrjModels.TRANSCRIPTION, device=device)
    except Exception as e:
        if device == "cuda":
            DLOG.warning(
                f"Failed to load model on CUDA ({e}). Falling back to CPU..."
            )
            device = "cpu"
            model = whisper.load_model(PrjModels.TRANSCRIPTION, device=device)
        else:
            raise

    elapsed = time.perf_counter() - startTime
    DLOG.info(f"Model loaded in {elapsed:.2f}s on {device}")
    return model


def transcribe(
    model: whisper.model.Whisper,
    pathRecording: Path,
) -> dict:
    startTime = time.perf_counter()

    if pathRecording.is_file():
        audio_path = pathRecording
    elif (PrjPaths.AUDIO_FILES / pathRecording).is_file():
        audio_path = PrjPaths.AUDIO_FILES / pathRecording
    else:
        audio_path = pathRecording

    DLOG.info(f"Transcribing {audio_path}...")

    try:
        result = model.transcribe(str(audio_path))
        elapsed = time.perf_counter() - startTime

        DLOG.info(f"Transcription done in {elapsed:.2f}s for {audio_path.name}")

        return {
            "pathRecording": pathRecording,
            "transcript": result["text"],
        }

    except Exception as e:
        DLOG.error(f"Failed to transcribe {audio_path}: {e}", exc_info=True)
        raise


def main() -> None:
    DLOG.info("=" * 60)
    DLOG.info("Starting audio transcription workflow")

    if not PrjPaths.AUDIO_FILES.exists():
        DLOG.warning(f"Audio files directory does not exist: {PrjPaths.AUDIO_FILES}")
        return

    recordings: list[Path] = [
        PrjPaths.AUDIO_FILES / pathAudioFile
        for pathAudioFile in os.listdir(PrjPaths.AUDIO_FILES)
        if not pathAudioFile.startswith(".")
    ]

    if not recordings:
        DLOG.warning(f"No audio files found in directory: {PrjPaths.AUDIO_FILES}")
        return

    DLOG.info(
        f"Found {len(recordings)} recording(s) to transcribe in {PrjPaths.AUDIO_FILES}"
    )
    for r in recordings:
        DLOG.info(f"  - {r.name}")

    # Load model once
    model = load_model()

    # Process each recording sequentially
    for idx, recording in enumerate(recordings, 1):
        DLOG.info(
            f"Processing recording [{idx}/{len(recordings)}]: {recording.name}"
        )

        try:
            recChunks = split_audio(recording)
            DLOG.info(f"[WORKFLOW] {recording.name} split into {len(recChunks)} chunk(s).")
        except FileNotFoundError:
            DLOG.warning(
                f"Skipped splitting {recording.name} as path could not be found. Transcribing without chunking."
            )
            recChunks = [recording]

        for chunk_idx, chunk in enumerate(recChunks, 1):
            DLOG.info(f"[WORKFLOW] Transcribing chunk [{chunk_idx}/{len(recChunks)}]: {chunk.name}")
            try:
                chunk_result = transcribe(model, chunk)
                qualifier = recording.name.split(".")[0]
                write_to_file(
                    chunk_result["transcript"],
                    PrjPaths.TRANSCRIPTS_RAW,
                    qualifier=qualifier,
                    overwrite=False
                )
            except OSError as e:
                DLOG.error(f"Failed to write transcript for {chunk} due to error {e}")
                raise

            except Exception as e:
                DLOG.error(
                    f"Skipped processing {recording.name} due to error {e}",
                    exc_info=True
                )
                raise

    DLOG.info("All transcriptions completed successfully!")
    DLOG.info("=" * 60)


if __name__ == "__main__":
    main()
