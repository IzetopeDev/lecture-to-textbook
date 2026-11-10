import logging
import os
import time

import whisper

from helpers import split_audio, write_to_file
from settings import *


logging.info(f"To transcribe recordings :>> {RECORDINGS}")

def load_model() -> whisper.model.Whisper:
    startTime = time.perf_counter()
    logging.info("Loading model...")

    model = whisper.load_model(PrjModels.transcription, device="cuda")
    model = whisper.load_model(PrjModels.transcription, device="cuda")

    logging.info(f"Model loaded in {time.perf_counter() - startTime:.2f}s")
    return model

def transcribe(
    model: whisper.model.Whisper,
    pathRecording: Path,
) -> dict:
    startTime = time.perf_counter()
    logging.info(f"Transcribing {pathRecording}...")

    try:
        result = model.transcribe(str(PrjPaths.audio_files/ pathRecording))

        logging.info(f"Transcription done in {time.perf_counter() - startTime:.2f}s")

        return {
            "pathRecording": pathRecording,
            "transcript": result["text"],
        }

    except Exception as e:
        logging.error(f"Failed to transcribe {pathRecording}: {e}", exc_info=True)
        raise

def main() -> None:

    RECORDINGS: list[Path] = [Path(pathAudioFile) for pathAudioFile in os.listdir(PrjPaths.audio_files)]
    
    logging.basicConfig(
        filename = os.path.join(PrjPaths.logs, "app.log"),
        level = logging.INFO,
        format = "%(asctime)s | %(levelname)s | %(message)s",
        encoding = "utf-8"
    )
    
    # Load model once
    model = load_model()

    # Process each recording sequentially
    for recording in RECORDINGS:

        try:
            recChunks = split_audio(recording)
        except FileNotFoundError:
            logging.warning(f"Skipped splitting {recording} as path could not be found. Transcribing without chunking.")
            recChunks = [recording]

        for chunk in recChunks:
            try:
                chunk_result = transcribe(model, chunk)
                write_to_file(
                    chunk_result["transcript"],
                    PrjPaths.transcripts_raw,
                    qualifier=str(recording).split(".")[0],
                    overwrite=False
                )
            except OSError as e:
                logging.error(f"Failed to write transcript for {chunk} due to error {e}")
                raise

            except Exception:
                logging.error(f"Skipped processing {recording}")
                raise

    logging.info("All transcriptions complete")

if __name__ == "__main__":
    main()
