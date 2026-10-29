from google.genai._gaos.types.interactions import Interaction
from google.genai._gaos.utils.eventstreaming import AsyncStream
import google.genai._gaos
from typing import Any

import time
import logging

import os
from dotenv import load_dotenv
from google import genai
import asyncio

logging.basicConfig(
    filename="./logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)

load_dotenv()

API_KEY: str = os.getenv("API_KEY")
PATH_CHUNKS = "./chunks/"
CHUNKS: list[str] = os.listdir(PATH_CHUNKS)
CLIENT = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.1-flash-lite"
PROMPT = """
You are editing ONE CHUNK of the transcript.

## Goal

Make the transcript more readable in english while preserving ALL meaningful content.

## Rules
- Summary is forbidden
- Use only vocabulary and information found in the source.
- Every substantive word in the edited transcript must come from the source transcript.
- Words may be removed, reordered, or grammatically adjusted only when supported by words already in the source.
- Remove filler only when it does not remove meaning.
- Do not introduce synonyms, replacement vocabulary, new concepts, facts, explanations, examples, opinions, or conclusions.
- Preserve the original flow and order.
- Preserve every meaningful idea, qualification, example, explanation, distinction, and detail. Omission is forbidden
- Preserve repetitions when they carry meaning.
- Separate distinct sub-ideas into readable paragraphs.
- Do not cite or mention the source file.

## Formatting
- in plaintext

Source Chunk:

"""

logging.info(f"""
CLIENT: {CLIENT}
CHUNKS({CHUNKS.__len__}): {CHUNKS}
PROMPT: 
===start===
{PROMPT}
===end===
""")
    

def writeToFile(text:str, qualifier: None | Any = None, filetype = '.txt', overwrite = False) -> None:
    startTime: float = time.perf_counter()
    
    finalFilePath =  f"./outputs/output{f"-{str(qualifier)}" if qualifier != None else ""}{filetype}"


    logging.info(f"writing file: {finalFilePath}")

    with open(
        finalFilePath,
        "w" if overwrite else "a", 
        encoding="utf-8"
    ) as file:
        file.write(text)

    logging.info(f"Done in {time.perf_counter() - startTime}: {finalFilePath}s")
    

async def promptModel(input:list[dict[str,str]] | str, model:str, count:int) -> dict[str,Any]:
    logging.info(f"Processing Chunk {count}...")
    startTime: float = time.perf_counter()
    
    interaction: AsyncStream[google.genai._gaos.interactions.InteractionSSEEvent] | Interaction = await CLIENT.aio.interactions.create(
        model = model,
        input = input
    )
    timeElapsed: float = round(time.perf_counter() - startTime, 2)
    
    logging.info(f"Returned Chunk {count} in {timeElapsed}s")
    return {
        "interaction" : interaction,
        "timeElapsed" : timeElapsed,
        "index" : count,
        "counts" : {
            "characters" : len(interaction.output_text),
            "words" : len(interaction.output_text.split()),
        },
    }


def writeToFile(text:str, qualifier: None | Any = None, filetype = '.txt', overwrite = False) -> None:
    startTime: float = time.perf_counter()
    
    finalFilePath =  f"./outputs/output{f"-{str(qualifier)}" if qualifier != None else ""}{filetype}"


    logging.info(f"writing file: {finalFilePath}")

    with open(
        finalFilePath,
        "w" if overwrite else "a", 
        encoding="utf-8"
    ) as file:
        file.write(text)

    logging.info(f"Done in {time.perf_counter() - startTime}: {finalFilePath}s")

async def main() -> None:
    print("prompting model")
        
    tasks = [
        promptModel(
            PROMPT + open(PATH_CHUNKS + chunk).read(), 
            MODEL_NAME, 
            count
        )
        for count, chunk in enumerate(CHUNKS)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    filesToWrite = []
    for count, result in enumerate(results):

        if isinstance(result, Exception):
            logging.error(
                "Chunk %d failed: %s",
                count,
                result,
                exc_info=(
                    type(result),
                    result,
                    result.__traceback__
                )
            )
            print(f"Chunk {count} receive. Check app.log.")

        filesToWrite.append(
            asyncio.to_thread(
                writeToFile,
                result["interaction"].output_text,
                result["index"],
            )
        )


    write_results = await asyncio.gather(*filesToWrite, return_exceptions=True)

    for result in write_results:
        if isinstance(result, Exception):
            logging.error(
                "Chunk %d failed: %s",
                count,
                result,
                exc_info=(
                    type(result),
                    result,
                    result.__traceback__
                )
            )
            print(f"Chunk {count} failed to write. Check app.log.")




asyncio.run(main())