import logging
import math
import time
from pathlib import Path
from typing import Any

import ffmpeg

from settings import PrjPaths

logging.basicConfig(
    filename="./logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

def split_audio(
    pathAudioFile: Path,
    chunkLength: int = 20
) -> list[Path]:
    """
    Split an audio file into chunks of a specified length.
    Args:
        pathAudioFile (Path): path to audio file
        chunkLength (int): chunkLength in minutes.

    Returns:
        list[Path]: list of paths to audio chunks

    Raises:
        FileNotFoundError: if audio file not found
        Exception: if any other exception occurs
    """

    # start logging and create tmp path
    logging.info(f"splitting audio: {pathAudioFile}")
    PrjPaths.tmp_audio_chunks.mkdir(parents=True, exist_ok=True)

    if not pathAudioFile.exists():
        raise FileNotFoundError(f"Audio file not found: {pathAudioFile}")

    # get duration and ext of audio file
    try:
        probe = ffmpeg.probe(pathAudioFile)
    except ffmpeg.Error as e:
        print(f"Exception raised: {e}")
        raise
    except Exception as e:
        print(f"Uncaught Exception raised: {e}")
        raise

    duration = float(probe['format']['duration'])
    ext = str(probe['streams'][0]['codec_name'])

    u = 60

    try:
        chunks = []
        for i, start in enumerate(range(0, math.ceil(duration), int(chunkLength*u))):
            outputFile = PrjPaths.tmp_audio_chunks / f"chunk{i}.{ext}"
            (
                ffmpeg
                .input(str(pathAudioFile), ss=start, t=chunkLength*u)
                .output(str(outputFile))
                .overwrite_output()
                .run(quiet=True)
            )

            chunks.append(outputFile)
    except Exception as e:
        logging.error("exception raised: ", e)
        raise

    return chunks



def write_to_file(
    text: str,
    destintation: Path,
    qualifier: Any | None = None,
    filetype: str = ".txt",
    overwrite: bool = False,
) -> Path:
    """
    Writes a string to a file.

    Args:
        text (str): The text to write.
        destintation (Path): The directory where the file will be saved.
        qualifier (Any | None): An optional qualifier to append to the filename.
        filetype (str): The file extension.
        overwrite (bool): Whether to overwrite the file

    Returns:
        Path: The path to the written file.

    Raises:
        OSError: If an error occurs while writing the file.
    """
    
    StartTime: float = time.perf_counter()

    try:
        destintation.mkdir(parents=True, exist_ok=True)

        Suffix = f"-{qualifier}" if qualifier is not None else ""
        FinalFilePath = destintation / f"output{Suffix}{filetype}"

        logging.info("Writing file: %s", FinalFilePath)

        with open(
            FinalFilePath,
            "w" if overwrite else "a",
            encoding="utf-8",
        ) as file:
            file.write(text)

        logging.info(
            "Done in %.2fs: %s",
            time.perf_counter() - StartTime,
            FinalFilePath,
        )
        return FinalFilePath

    except OSError:
        logging.exception("Failed writing output file for qualifier=%s", qualifier)
        raise
