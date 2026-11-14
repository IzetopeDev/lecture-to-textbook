import os
import time
from pathlib import Path

import torch
import whisper

from helpers import DLOG, split_audio, write_to_file
from settings import PrjModels, PrjPaths


def _load_whisper(
     model_name: str = PrjModels.TRANSCRIPTION,
     device: str = "cuda"
) -> whisper.model.Whisper:
    startTime = time.perf_counter()

    valid_models = whisper.available_models()
    if model_name not in valid_models:
        msg = f"Invalid model '{model_name}'. Available models are: {valid_models}"
        DLOG.error(msg)
        raise ValueError(msg)

    valid_devices = ["cuda", "cpu"]
    if device not in valid_devices:
        msg = f"Invalid device '{device}'. Valid devices are: {valid_devices}"
        DLOG.error(msg)
        raise ValueError(msg)

    if not torch.cuda.is_available():
       device = "cpu"
       DLOG.warning("GPU not found! Loading on CPU!")

    DLOG.info(
        f"Loading Whisper model '{model_name}' on device '{device}'..."
    )

    try:
        model = whisper.load_model(model_name, device=device)
    except Exception:
        if device == "cuda":
            DLOG.warning(
                "Failed to load model on CUDA. Falling back to CPU...",
                exc_info=True
            )
            device = "cpu"
            model = whisper.load_model(model_name, device=device)
            
        else:
            DLOG.exception("Failed to load model!")
            raise 

    elapsed = time.perf_counter() - startTime
    DLOG.info(f"Model loaded in {elapsed:.2f}s on {device}")

    return model


def _transcribe(
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
        DLOG.exception(f"Failed to transcribe {audio_path}")
        raise


def main() -> None:
    DLOG.info("Starting audio transcription workflow...")

    if not PrjPaths.AUDIO_FILES.exists():
        DLOG.error(f"Audio files directory does not exist: {PrjPaths.AUDIO_FILES}")
        raise FileNotFoundError()

    recordings: list[Path] = [
        PrjPaths.AUDIO_FILES / audioFileName
        for audioFileName in os.listdir(PrjPaths.AUDIO_FILES)
        if not audioFileName.startswith(".")
    ]

    if not recordings:
        DLOG.error(f"No audio files found in directory: {PrjPaths.AUDIO_FILES}")
        raise FileNotFoundError()

    DLOG.info(
        f"Found {len(recordings)} recording(s) to transcribe in {PrjPaths.AUDIO_FILES}:"
    )
    for r in recordings:
        DLOG.info(f"  - {r.name}")

    # Load model once
    model = _load_model()

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
                chunk_result = _transcribe(model, chunk)
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
