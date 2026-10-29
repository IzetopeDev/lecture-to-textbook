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

    chunkFiles = sorted(
        filePath for filePath in pathChunks.iterdir() if filePath.is_file()
    )

    if not chunkFiles:
        raise FileNotFoundError(f"No chunk files found in directory: {pathChunks}")

    return chunkFiles

def read_chunk_file(filePath: Path) -> str:
    try:
        return filePath.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.exception(f"Chunk file not found: {filePath}")
        raise
    except OSError:
        logging.exception(f"Failed to read chunk file: {filePath}")
        raise

async def prompt_model(
    input: list[dict[str, str]] | str,
    model: str,
    count: int,
) -> dict[str, Any]:
    if CLIENT is None:
        raise RuntimeError("Client has not been initialized.")

    logging.info("Processing chunk %d...", count)
    startTime: float = time.perf_counter()

    try:
        interactionResult: (
            AsyncStream[google.genai._gaos.interactions.InteractionSSEEvent]
            | Interaction
        ) = await CLIENT.aio.interactions.create(
            model=model,
            input=input,
        )
    except Exception:
        logging.exception("Model request failed for chunk %d", count)
        raise

    timeElapsed: float = round(time.perf_counter() - startTime, 2)
    outputText: str = getattr(interactionResult, "output_text", "") or ""

    if not outputText:
        logging.warning("Chunk %d returned empty output_text.", count)

    logging.info("Returned chunk %d in %ss", count, timeElapsed)

    return {
        "interaction": interactionResult,
        "output_text": outputText,
        "time_elapsed": timeElapsed,
        "index": count,
        "counts": {
            "characters": len(outputText),
            "words": len(outputText.split()),
        },
    }

async def main() -> None:
    global CLIENT

    try:
        CLIENT = setup_client()
        chunkFiles = get_chunk_files(PrjPaths.chunks)

        logging.info(
            f"\nCLIENT: {CLIENT}\n"
            f"CHUNKS({len(chunkFiles)}): {[(str(filePath) + "\n") for filePath in chunkFiles]}\n"
            f"PROMPT:\n===start===\n{Prompts.clean_transcript}\n===end===\n"
        )
    except Exception:
        logging.exception("Startup validation failed.")
        print("Startup failed. Check app.log.")
        return

    print("prompting model")

    tasks = []
    for count, chunkPath in enumerate(chunkFiles):
        try:
            chunkText = read_chunk_file(chunkPath)
            tasks.append(
                prompt_model(
                    Prompts.clean_transcript + chunkText,
                    PrjModels.clean_transcript,
                    count,
                )
            )
        except Exception:
            logging.exception(f"Skipping chunk {count} due to read failure: {chunkPath}")
            print(f"Chunk {count} skipped. Check app.log.")

    if not tasks:
        logging.warning("No valid chunk tasks to process.")
        print("No valid chunk files to process.")
        return

    results = await asyncio.gather(*tasks, return_exceptions=True)

    filesToWrite = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(
                f"Chunk {result["index"]} failed: {result}",
                exc_info=(type(result), result, result.__traceback__),
            )
            print(f"Chunk {result["index"]} failed. Check app.log.")
            continue

        filesToWrite.append(
            asyncio.to_thread(
                write_to_file,
                result["output_text"],
                PrjPaths.transcripts_cleaned,
                result["index"],
            )
        )

    if not filesToWrite:
        logging.warning("No successful model outputs to write.")
        print("No outputs to write.")
        return

    writeResults = await asyncio.gather(*filesToWrite, return_exceptions=True)

    for count, result in enumerate(writeResults):
        if isinstance(result, Exception):
            logging.error(
                f"Write task {count} failed: {result}",
                exc_info=(type(result), result, result.__traceback__),
            )
            print(f"Write task {count} failed. Check app.log.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
        print("Interrupted.")
    except Exception:
        logging.exception("Unhandled fatal error.")
        print("Fatal error. Check app.log.")



