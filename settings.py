import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class PrjEnv:
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

@dataclass(frozen=True)
class PrjPaths:
    CHUNKS: Path = Path("./input-output/chunks")
    TRANSCRIPTS_RAW: Path = Path("./input-output/transcripts")
    TRANSCRIPTS_CLEANED: Path = Path("./input-output/outputs")
    AUDIO_FILES: Path = Path("./input-output/audio files")
    LOGS: Path = Path('./logs')
    TMP: Path = Path('./tmp')
    TMP_AUDIO_CHUNKS = Path('./tmp/audio_chunks')

class PrjModels:
    TRANSCRIPTION: str = "medium.en"
    CLEAN_TRANSCRIPT: str = "gemini-3.1-flash-lite"

@dataclass(frozen=True)
class Prompts:
    CLEAN_TRANSCRIPT: str = """
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
