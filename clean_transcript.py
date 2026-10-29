from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai._gaos.types.interactions import Interaction
from google.genai._gaos.utils.eventstreaming import AsyncStream
import google.genai._gaos

from settings import *
from helpers import write_to_file


logging.basicConfig(
    filename="./logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

load_dotenv()

API_KEY: str | None = os.getenv("API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

CLIENT: genai.Client | None = None

def setup_client() -> genai.Client:
    if not API_KEY:
        raise ValueError("Missing API_KEY in environment.")

    return genai.Client(api_key=API_KEY)

def get_chunk_files(pathChunks: Path) -> list[Path]:
    if not pathChunks.exists():
        raise FileNotFoundError(f"Chunks directory does not exist: {pathChunks}")

    if not pathChunks.is_dir():
        raise NotADirectoryError(f"Chunks path is not a directory: {pathChunks}")

    ChunkFiles = sorted(
        filePath for filePath in pathChunks.iterdir() if filePath.is_file()
    )

    if not ChunkFiles:
        raise FileNotFoundError(f"No chunk files found in directory: {pathChunks}")

    return ChunkFiles

def read_chunk_file(filePath: Path) -> str:
    try:
        return filePath.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.exception("Chunk file not found: %s", filePath)
        raise
    except OSError:
        logging.exception("Failed to read chunk file: %s", filePath)
        raise

async def prompt_model(
    input: list[dict[str, str]] | str,
    model: str,
    count: int,
) -> dict[str, Any]:
    if CLIENT is None:
        raise RuntimeError("Client has not been initialized.")

    logging.info("Processing chunk %d...", count)
    StartTime: float = time.perf_counter()

    try:
        InteractionResult: (
            AsyncStream[google.genai._gaos.interactions.InteractionSSEEvent]
            | Interaction
        ) = await CLIENT.aio.interactions.create(
            model=model,
            input=input,
        )
    except Exception:
        logging.exception("Model request failed for chunk %d", count)
        raise

    TimeElapsed: float = round(time.perf_counter() - StartTime, 2)
    OutputText: str = getattr(InteractionResult, "output_text", "") or ""

    if not OutputText:
        logging.warning("Chunk %d returned empty output_text.", count)

    logging.info("Returned chunk %d in %ss", count, TimeElapsed)

    return {
        "interaction": InteractionResult,
        "timeElapsed": TimeElapsed,
        "index": count,
        "counts": {
            "characters": len(OutputText),
            "words": len(OutputText.split()),
        },
    }

async def main() -> None:
    global CLIENT

    try:
        CLIENT = setup_client()
        ChunkFiles = get_chunk_files(PrjPaths.chunks)

        logging.info(
            "\nCLIENT: %s\nCHUNKS(%d): %s\nPROMPT:\n===start===\n%s\n===end===\n",
            CLIENT,
            len(ChunkFiles),
            [str(filePath) for filePath in ChunkFiles],
            Prompts.clean_transcript,
        )
    except Exception:
        logging.exception("Startup validation failed.")
        print("Startup failed. Check app.log.")
        return

    print("prompting model")

    Tasks = []
    for Count, ChunkPath in enumerate(ChunkFiles):
        try:
            ChunkText = read_chunk_file(ChunkPath)
            Tasks.append(
                prompt_model(
                    Prompts.clean_transcript + ChunkText,
                    MODEL_NAME,
                    Count,
                )
            )
        except Exception:
            logging.exception("Skipping chunk %d due to read failure: %s", Count, ChunkPath)
            print(f"Chunk {Count} skipped. Check app.log.")

    if not Tasks:
        logging.warning("No valid chunk tasks to process.")
        print("No valid chunk files to process.")
        return

    Results = await asyncio.gather(*Tasks, return_exceptions=True)

    FilesToWrite = []
    for Count, Result in enumerate(Results):
        if isinstance(Result, Exception):
            logging.error(
                "Chunk %d failed: %s",
                Count,
                Result,
                exc_info=(type(Result), Result, Result.__traceback__),
            )
            print(f"Chunk {Count} failed. Check app.log.")
            continue

        OutputText = getattr(Result["interaction"], "output_text", "") or ""
        FilesToWrite.append(
            asyncio.to_thread(
                write_to_file,
                OutputText,
                PrjPaths.transcripts_cleaned,
                Result["index"],
            )
        )

    if not FilesToWrite:
        logging.warning("No successful model outputs to write.")
        print("No outputs to write.")
        return

    WriteResults = await asyncio.gather(*FilesToWrite, return_exceptions=True)

    for Count, Result in enumerate(WriteResults):
        if isinstance(Result, Exception):
            logging.error(
                "Write task %d failed: %s",
                Count,
                Result,
                exc_info=(type(Result), Result, Result.__traceback__),
            )
            print(f"Write task {Count} failed. Check app.log.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
        print("Interrupted.")
    except Exception:
        logging.exception("Unhandled fatal error.")
        print("Fatal error. Check app.log.")



