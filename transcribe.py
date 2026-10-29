import whisper
import os
import logging
import time
from dotenv import load_dotenv

load_dotenv()

PATH_LOGS = os.getenv("PATH_LOGS")
PATH_RECORDINGS = os.getenv("PATH_RECORDINGS")
RECORDINGS = os.listdir(PATH_RECORDINGS)
MODEL_TYPE = os.getenv("TRANSCRIPTION_MODEL_TYPE")

logging.basicConfig(
    filename = os.path.join(PATH_LOGS, "app.log"),
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    encoding = "utf-8"
)

logging.info(f"To transcribe recordings :>> {RECORDINGS}")

def load_model() -> whisper.model.Whisper:
    startTime = time.perf_counter()
    logging.info("Loading model...")
    
    model = whisper.load_model(MODEL_TYPE, device="cuda")
    
    logging.info(f"Model loaded in {time.perf_counter() - startTime:.2f}s")
    return model

def transcribe(model: whisper.model.Whisper, recording: str) -> dict:
    startTime = time.perf_counter()
    logging.info(f"Transcribing {recording}...")
    
    try:
        result = model.transcribe(os.path.join(PATH_RECORDINGS, recording))
        
        logging.info(f"Transcription done in {time.perf_counter() - startTime:.2f}s")
        
        return {
            "recording": recording,
            "transcript": result["text"],
            "success": True
        }
    
    except Exception as e:
        logging.error(f"Failed to transcribe {recording}: {e}", exc_info=True)
        return {
            "recording": recording,
            "success": False,
            "error": str(e)
        }

def write_to_file(text: str, qualifier: str, filetype = '.txt', overwrite=True) -> None:
    startTime = time.perf_counter()
    
    finalFilePath = f"./input-output/transcripts/transcript-{qualifier}{filetype}"
    
    logging.info(f"Writing file: {finalFilePath}")
    
    with open(finalFilePath, "w" if overwrite else "a", encoding="utf-8") as file:
        file.write(text)
    
    logging.info(f"Done in {time.perf_counter() - startTime:.2f}s: {finalFilePath}")

def main() -> None:
    # Load model once
    model = load_model()
    
    # Process each recording sequentially
    for recording in RECORDINGS:
        result = transcribe(model, recording)
        
        if result["success"]:
            write_to_file(result["transcript"], result["recording"].split(".")[0])
        else:
            logging.error(f"Skipped writing transcript for {recording} due to error")
    
    logging.info("All transcriptions complete")

if __name__ == "__main__":
    main()
