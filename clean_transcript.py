from __future__ import annotations

import asyncio
import time
from typing import Any

from google import genai
from google.genai._gaos.types.interactions import Interaction
from google.genai._gaos.utils.eventstreaming import AsyncStream

from helpers import DLOG, write_to_file
from settings import *

CLIENT: genai.Client | None = None


def setup_client() -> genai.Client:
    DLOG.info("Setting up Gemini API client...")
    if not PrjEnv.GEMINI_API_KEY:
        DLOG.error("Missing API_KEY in environment variables.")
        raise ValueError("Missing API_KEY in environment.")

    return genai.Client(api_key=PrjEnv.GEMINI_API_KEY)


def get_chunk_files(pathChunks: Path) -> list[Path]:
    DLOG.info(f"Retrieving chunk files from: {pathChunks}")

    if not pathChunks.exists():
        DLOG.error(f"Chunks directory does not exist: {pathChunks}")
        raise FileNotFoundError(f"Chunks directory does not exist: {pathChunks}")

    if not pathChunks.is_dir():
        DLOG.error(f"Chunks path is not a directory: {pathChunks}")
        raise NotADirectoryError(f"Chunks path is not a directory: {pathChunks}")

    chunkFiles = sorted(
        filePath for filePath in pathChunks.iterdir() if filePath.is_file()
    )

    if not chunkFiles:
        DLOG.error(f"No chunk files found in directory: {pathChunks}")
        raise FileNotFoundError(f"No chunk files found in directory: {pathChunks}")

    DLOG.info(f"Found {len(chunkFiles)} chunk file(s)")
    for f in chunkFiles:
        DLOG.debug(f"  - {f.name}")

    return chunkFiles


def read_chunk_file(filePath: Path) -> str:
    DLOG.info(f"Reading chunk file: {filePath}")
    try:
        content = filePath.read_text(encoding="utf-8")
        DLOG.info(f"Successfully read {filePath.name} ({len(content)} characters)")
        return content
    except FileNotFoundError:
        DLOG.exception(f"Chunk file not found: {filePath}")
        raise
    except OSError:
        DLOG.exception(f"Failed to read chunk file: {filePath}")
        raise


async def prompt_model(
    input: list[dict[str, str]] | str,
    model: str,
    count: int,
) -> dict[str, Any]:
    if CLIENT is None:
        DLOG.error("Client has not been initialized.")
        raise RuntimeError("Client has not been initialized.")

    DLOG.info(f"Processing chunk {count} with model '{model}'...")
    startTime: float = time.perf_counter()

    try:
        interactionResult: (
            AsyncStream[Any]
            | Interaction
        ) = await CLIENT.aio.interactions.create(
            model=model,
            input=input,
        )
    except Exception as e:
        DLOG.exception(f"Model request failed for chunk {count}: {e}")
        raise

    timeElapsed: float = round(time.perf_counter() - startTime, 2)
    outputText: str = getattr(interactionResult, "output_text", "") or ""

    if not outputText:
        DLOG.warning(f"Chunk {count} returned empty output_text.")

    char_count = len(outputText)
    word_count = len(outputText.split())

    DLOG.info(
        f"Returned chunk {count} in {timeElapsed}s ({char_count} chars, {word_count} words)"
    )

    return {
        "interaction": interactionResult,
        "output_text": outputText,
        "time_elapsed": timeElapsed,
        "index": count,
        "counts": {
            "characters": char_count,
            "words": word_count,
        },
    }


async def main() -> None:
    global CLIENT

    DLOG.info("=" * 60)
    DLOG.info("Starting Transcript Cleaning Workflow (Gemini AI)")

    try:
        CLIENT = setup_client()
        chunkFiles = get_chunk_files(PrjPaths.CHUNKS)

        chunk_names = [filePath.name for filePath in chunkFiles]
        DLOG.info(
            f"Client initialized successfully. "
            f"Chunks to process ({len(chunkFiles)}): {chunk_names}\n"
            f"PROMPT:\n===start===\n{Prompts.CLEAN_TRANSCRIPT}\n===end===\n"
        )
    except Exception as e:
        DLOG.exception(f"Startup validation failed: {e}")
        return

    DLOG.info(f"Prompting model '{PrjModels.CLEAN_TRANSCRIPT}' for {len(chunkFiles)} chunk(s)...")

    tasks = []
    for count, chunkPath in enumerate(chunkFiles):
        try:
            chunkText = read_chunk_file(chunkPath)
            tasks.append(
                prompt_model(
                    Prompts.CLEAN_TRANSCRIPT + chunkText,
                    PrjModels.CLEAN_TRANSCRIPT,
                    count,
                )
            )
        except Exception as e:
            DLOG.exception(f"Skipping chunk {count} ({chunkPath.name}) due to read failure: {e}")

    if not tasks:
        DLOG.warning("No valid chunk tasks to process.")
        return

    DLOG.info(f"Executing {len(tasks)} model prompt task(s) concurrently...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    filesToWrite = []
    for result in results:
        if isinstance(result, Exception):
            chunk_idx = getattr(result, "index", "unknown")
            DLOG.error(f"Chunk {chunk_idx} failed: {result}", exc_info=True)
            continue

        if isinstance(result, dict):
            idx = result["index"]
            output_text = result["output_text"]
            DLOG.info(f"Scheduling write for cleaned transcript chunk {idx}...")
            filesToWrite.append(
                asyncio.to_thread(
                    write_to_file,
                    output_text,
                    PrjPaths.TRANSCRIPTS_CLEANED,
                    idx,
                )
            )

    if not filesToWrite:
        DLOG.warning("No successful model outputs to write.")
        return

    DLOG.info(f"Writing {len(filesToWrite)} output file(s)...")
    writeResults = await asyncio.gather(*filesToWrite, return_exceptions=True)

    success_count = 0
    for count, result in enumerate(writeResults):
        if isinstance(result, Exception):
            DLOG.error(
                f"Write task {count} failed: {result}",
                exc_info=(type(result), result, result.__traceback__),
            )
        else:
            success_count += 1

    DLOG.info(f"Transcript cleaning complete. Successfully wrote {success_count}/{len(writeResults)} file(s).")
    DLOG.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        DLOG.info("Program interrupted by user.")
    except Exception as e:
        DLOG.exception(f"Unhandled fatal error: {e}")



