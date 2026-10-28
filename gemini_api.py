import time
from importlib.resources import read_text
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


#chunks:list = []
#shouldContinue = True
#while shouldContinue:
#
#    usrInput = input("paste chunk\n")
#
#    logging.info(f"prompted for chunk \nusrInput :>> {usrInput}")
#
#    if (usrInput.strip() == "done"):
#        shouldContinue = False
#        break
#
#    chunks.append(usrInput)
#
#logging.info(f"chunks :>> {chunks}")

async def main() -> None:
    print("prompting model")
        
    tasks = [
        promptModel(
            [{
                "type": "text",
                "text": PROMPT + open(PATH_CHUNKS + chunk).read()
            }], 
            MODEL_NAME, 
            count
        )
        for count, chunk in enumerate(CHUNKS)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)


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
            print(f"Chunk {count} failed. Check app.log.")


async def promptModel(input:list[dict[str,str]], model:str, count:int) -> None:
    logging.info(f"Processing Chunk {count}...")
    startTime: float = time.perf_counter()
    
    interaction = await CLIENT.aio.interactions.create(
        model = model,
        input = input
    )

    with open(f"./outputs/output-{count}.md", "w", encoding="utf-8") as file:
        file.write(interaction.output_text)

    print(f"Response saved to output-{count}.txt")  
    logging.info(f"""
    Response saved to output-{count}.txt\
    Time elapsed: {round(time.perf_counter() - startTime,)}s
    """)


asyncio.run(main())