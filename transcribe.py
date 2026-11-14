import time
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch
import whisper

from helpers import DLOG
from settings import PrjModels, PrjPaths, Prompts

T = TypeVar("T")


def _fallback_to_cpu(
    action: Callable[[], T],
    context: str,
    model: whisper.model.Whisper | None = None,
) -> T:
    DLOG.warning(f"CUDA failed while {context}. Falling back to CPU...", exc_info=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if model is not None:
        model.to("cpu")
    try:
        return action()
    except Exception:
        DLOG.exception(f"Failed while {context} on CPU fallback!")
        raise


def _load_whisper(
     model_name: str = PrjModels.TRANSCRIPTION,
     device: str = "cuda"
) -> whisper.model.Whisper:
    startTime = time.perf_counter()
    
    if not torch.cuda.is_available():
       device = "cpu"
       DLOG.warning("GPU not found! Loading on CPU!")

    DLOG.info(
        f"Loading Whisper model '{model_name}' on device '{device}'..."
    )

    try:
        model = whisper.load_model(model_name, device=device)
    except urllib.error.URLError as e:
        DLOG.error(f"Network error while downloading Whisper model '{model_name}': {e}")
        raise
    except FileNotFoundError as e:
        DLOG.error(f"Model checkpoint file not found for '{model_name}': {e}")
        raise
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if device == "cuda":
            device = "cpu"
            model = _fallback_to_cpu(
                lambda: whisper.load_model(model_name, device="cpu"),
                context=f"loading model '{model_name}'",
            )
        else:
            DLOG.error(f"Runtime error while loading Whisper model '{model_name}': {e}")
            raise
    except Exception:
        DLOG.exception(f"Unexpected error loading Whisper model '{model_name}'!")
        raise 

    elapsed = time.perf_counter() - startTime
    DLOG.info(f"Model loaded in {elapsed:.2f}s on {device}")

    return model


def transcribe(
    model: whisper.model.Whisper | str,
    path_audio_file: Path,
    path_ouput: Path | None = PrjPaths.TMP_TRANSCRIPT_CHUNKS,
) -> dict[str, dict | Path]:

    startTime = time.perf_counter()

    if not path_audio_file.is_file():
        DLOG.error(f"Audio file does not exist: {path_audio_file}")
        raise FileNotFoundError(f"Audio file not found: {path_audio_file}")

    _model = model if not isinstance(model, str) else _load_whisper(model)

    DLOG.info(f"Transcribing {path_audio_file}...")

    try:
        try:
            result = _model.transcribe(
                str(path_audio_file), 
                verbose=False,
                initial_prompt=Prompts.WHISPER_INITIAL_PROMPT,
            )
        except torch.cuda.OutOfMemoryError:
            result = _fallback_to_cpu(
                lambda: _model.transcribe(
                    str(path_audio_file),
                    verbose=False,
                    initial_prompt=Prompts.WHISPER_INITIAL_PROMPT,
                ),
                context=f"transcribing {path_audio_file.name}",
                model=_model,
            )

        elapsed = time.perf_counter() - startTime
        DLOG.info(f"Transcription done in {elapsed:.2f}s for {path_audio_file.name}")

        if path_ouput is None:
            try:
                print(result)
            except BrokenPipeError:
                DLOG.warning("Output pipe closed while printing transcript.")

    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower():
            DLOG.error(f"FFmpeg is not installed or not found in system PATH: {e}")
        else:
            DLOG.error(f"File not found: {e}")
        raise

    except MemoryError:
        DLOG.error(f"System memory exhausted while processing {path_audio_file.name}")
        raise

    except RuntimeError as e:
        DLOG.error(f"Runtime error while transcribing {path_audio_file.name}: {e}")
        raise

    except Exception:
        DLOG.exception(f"Unexpected error while transcribing {path_audio_file}")
        raise

    PrjPaths.TRANSCRIPTS_RAW.mkdir(parents=True, exist_ok=True)
    output_file = PrjPaths.TRANSCRIPTS_RAW / f"{path_audio_file.stem}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(result["text"]))
    DLOG.info(f"Transcript successfully saved to {output_file}")

    return {
        "result": result,
        "output_file": output_file 
    }
    
        


def batch_transcribe(
    input_folder: Path = PrjPaths.AUDIO_FILES,
    output_folder: Path | None = None,  
) -> None:
    DLOG.info("Starting audio transcription workflow...")

    if not input_folder.exists():
        DLOG.error(f"Audio files directory does not exist: {input_folder}")
        raise FileNotFoundError()

    recordings: list[Path] = [
        item for item in input_folder.iterdir()
        if item.is_file() and not item.name.startswith(".")
    ]

    if not recordings:
        DLOG.error(f"No audio files found in directory: {input_folder}")
        raise FileNotFoundError()

    DLOG.info(
        f"Found {len(recordings)} recording(s) to transcribe in {input_folder}:"
    )
    for r in recordings:
        DLOG.info(f"  - {r.name}")

    # Load model once
    model = _load_whisper()

    failed_recordings: list[Path] = []

    # Process each recording sequentially
    for idx, recording in enumerate(recordings, 1):
        DLOG.info(f"Processing recording [{idx}/{len(recordings)}]: {recording.name}")
        try:
            transcribe(model, (input_folder / recording), output_folder)
            
        except FileNotFoundError as e:
            if "ffmpeg" in str(e).lower():
                DLOG.critical(f"FFmpeg not found in system PATH. Aborting batch: {e}")
                raise
            DLOG.error(f"Skipping {recording.name} (file not found): {e}")
            failed_recordings.append(recording)

        except (RuntimeError, OSError, MemoryError, ValueError) as e:
            DLOG.error(f"Skipping {recording.name} due to error: {e}")
            failed_recordings.append(recording)

    if failed_recordings:
        DLOG.warning(
            f"Batch completed with {len(failed_recordings)}/{len(recordings)} failure(s): "
            f"{[r.name for r in failed_recordings]}"
        )
    else:
        DLOG.info("All transcriptions completed successfully!")
    DLOG.info("=" * 60)


if __name__ == "__main__":
    batch_transcribe()
