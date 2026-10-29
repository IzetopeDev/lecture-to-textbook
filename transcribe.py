import whisper.model
import os
import asyncio
import logging
import time
import whisper
from dotenv import load_dotenv

load_dotenv()

PATH_LOGS = os.getenv("PATH_LOGS")
PATH_RECORDINGS = os.getenv("PATH_RECORDINGS")
RECORDINGS = os.listdir(PATH_RECORDINGS)

logging.basicConfig(
    filename = os.path.join(PATH_LOGS, "app.log"),
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    encoding = "utf-8"
)


logging.info(f"To transcribe recordings :>> {RECORDINGS}")

async def loadModel() -> whisper.model.Whisper:
    startTime = time.perf_counter()
    logging.info(f"Loading model...")

    model: whisper.model.Whisper = await asyncio.to_thread(
        lambda: whisper.load_model("small.en", device="cuda")
    )

    logging.info(f"Model loaded in {time.perf_counter() - startTime}s")

    return model


def transcribe(model:whisper.model.Whisper, recording:str) -> dict[str, str]:
    startTime = time.perf_counter()
    logging.info(f"Transcribing {recording}...")

    result = model.transcribe(os.path.join(PATH_RECORDINGS, recording))

    logging.info(f"Transcription done in {time.perf_counter() - startTime}")

    return {
        "recording" : recording,
        "transcript" : result["text"],
    }

def writeToFile(text:str, qualifier: str, filetype = '.txt', overwrite = True) -> None:
    startTime: float = time.perf_counter()
    
    finalFilePath =  f"./input-output/transcripts/transcript{f"-{str(qualifier)}" if qualifier != None else ""}{filetype}"


    logging.info(f"writing file: {finalFilePath}")

    with open(
        finalFilePath,
        "w" if overwrite else "a", 
        encoding="utf-8"
    ) as file:
        file.write(text)

    logging.info(f"Done in {time.perf_counter() - startTime}s: {finalFilePath}")

async def main() -> list[str]:

    model: whisper.model.Whisper = await loadModel()

    tasks = [ 
        asyncio.to_thread(transcribe, model, recording) 
        for recording in RECORDINGS 
    ]        
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Task failed: {result}", exc_info=(type(result), result, result.__traceback__))
            continue


        writeToFile(result["transcript"], result["recording"].split(".")[0])
    

asyncio.run(main())
    