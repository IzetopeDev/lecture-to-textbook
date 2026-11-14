import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import ffmpeg

from settings import PrjPaths

# SUGGEST:
# implement multiple loggers for granularity with DLOG as parent
# so that everything sends to DLOG and there will not be duplicate
# handlers watching and recording the same thing.

# REMINDER:
# only create new loggers in the same file as DLOG for easy traceability.

DLOG = logging.getLogger("LTNApp") # DLOG: Debug Logger, LTNApp: lecture to notes app.

_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_log_level = logging.DEBUG

class _ConsoleStreamHandler(logging.StreamHandler):
    """StreamHandler that dynamically resolves sys.stderr to support redirections and test capture."""

    def __init__(self) -> None:
        super().__init__(None)

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


PrjPaths.LOGS.mkdir(parents=True, exist_ok=True)

if not DLOG.handlers:
    # TODO(once stable): change "a" to "w" to prevent over-accumulation of logs.
    _fileHandler = logging.FileHandler((PrjPaths.LOGS / Path("debug.log")), "a", "utf-8", errors="backslashreplace")
    _fileHandler.setFormatter(_formatter)
    _fileHandler.setLevel(_log_level)

    _streamHandler = _ConsoleStreamHandler()
    _streamHandler.setFormatter(_formatter)
    _streamHandler.setLevel(_log_level)

    DLOG.addHandler(_fileHandler)
    DLOG.addHandler(_streamHandler)

DLOG.setLevel(_log_level)


def clear_tmp(
    dir: Path = PrjPaths.TMP
) -> None:
    if not dir.is_relative_to(PrjPaths.TMP):
        DLOG.warning(f"Unable to clear tmp folder. Invalid path passed: {dir!s}. Operation Skipped")
        return

    DLOG.info(f"clearing {dir}")

    count = 0
    for child in dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
            DLOG.debug(f"Cleared {child}")
            count += 1

    DLOG.info(f"Total files cleared: {count}")


def split_audio(
    path_audio_file: Path,
    chunk_length: int = 20,
    path_output: Path = PrjPaths.TMP_AUDIO_CHUNKS,
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

    DLOG.info(f"Splitting audio file: {path_audio_file} (chunk length: {chunk_length} min)")
    path_output.mkdir(parents=True, exist_ok=True)

    if not path_audio_file.exists():
        DLOG.error(f"Audio file not found: {path_audio_file}")
        raise FileNotFoundError()

    try:
        probe = ffmpeg.probe(path_audio_file)
    except ffmpeg.Error:
        DLOG.exception(f"FFmpeg probe passed an error for {path_audio_file}")
        raise
    except Exception:
        DLOG.exception(f"Uncaught exception during FFmpeg probe for {path_audio_file}")
        raise

    duration = float(probe['format']['duration'])
    ext = str(probe['streams'][0]['codec_name'])

    DLOG.info(f"Audio file probed: duration={duration:.2f}s, format/codec={ext}")

    try:
        chunks = []
        ranges = list(range(0, math.ceil(duration), int(chunk_length * 60))) #NOTE: hardcoded conversion to minutes!
        total_chunks = len(ranges)
        DLOG.info(f"Splitting into {total_chunks} chunk(s)...")

        clear_tmp(PrjPaths.TMP_AUDIO_CHUNKS)

        for i, start in enumerate(ranges):
            outputFile = PrjPaths.TMP_AUDIO_CHUNKS / f"chunk{i}.{ext}"
            (
                ffmpeg
                .input(str(path_audio_file), ss=start, t=chunk_length * 60) #NOTE: hardcoded conversion to minutes!
                .output(str(outputFile))
                .overwrite_output()
                .run(quiet=True)
            )
            DLOG.info(f"Created audio chunk [{i+1}/{total_chunks}]: {outputFile}")
            chunks.append(outputFile)

        DLOG.info(f"Successfully split audio file '{path_audio_file.name}' into {len(chunks)} chunk(s)")
    except Exception:
        DLOG.exception("Exception raised while splitting audio")
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

        DLOG.info(f"Writing file: {FinalFilePath}")

        with open(
            FinalFilePath,
            "w" if overwrite else "a",
            encoding="utf-8",
        ) as file:
            file.write(text)

        elapsed = time.perf_counter() - StartTime
        DLOG.info(f"Done in {elapsed:.2f}s: {FinalFilePath}")
        return FinalFilePath

    except OSError:
        DLOG.exception(f"Failed writing output file for qualifier={qualifier}")
        raise
