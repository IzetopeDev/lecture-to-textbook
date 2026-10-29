from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class PrjPaths:
    chunks: Path = Path("./input-output/chunks")
    transcripts_raw: Path = Path("./input-output/transcripts")
    transcripts_cleaned: Path = Path("./input-output/ouputs")
    audio_files: Path = Path("./input-output/audio files")
    logs: Path = Path('./logs')

class PrjModels:
    transcription: str = "medium.en"

@dataclass(frozen=True)
class Prompts:
    clean_transcript: str = """
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
