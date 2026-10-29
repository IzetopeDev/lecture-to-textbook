from pathlib import Path
import logging
import time
from typing import Any


def write_to_file(
    text: str,
    destintation: Path,
    qualifier: Any | None = None,
    filetype: str = ".txt",
    overwrite: bool = False,
) -> Path:
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