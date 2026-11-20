# Lecture to Textbook

**Lecture to Textbook** is a tool built to solve a simple problem: making sure information from lectures doesn't get lost, and turning raw spoken recordings into reliable notes formatted for easy reading and quick reference.

Most AI note-taking tools try to summarize lectures into bullet points, which strips out the nuance, examples, and exact explanations that actually matter when you're studying. This project does the opposite. Summarization and omission are strictly forbidden—it cleans up spoken filler and rough grammar while keeping every substantive word, concept, and detail intact.

## Why It Exists

- **Less Rewatching, Faster Lookups:** Scrubbing through a 1- or 2-hour lecture recording just to find one explanation is a massive time sink. Having a clean, readable text version cuts that down to a **10-minute lookup and 10 minutes of actual understanding**.
- **Better Retention & Fewer Repeat Questions:** When you can easily read through everything that was said in class, you retain more of the material and avoid asking questions that were already answered during the lecture.
- **Automating the Tedious Parts:** Doing this workflow manually—splitting files, transferring transcript chunks, and prompting an LLM to clean them up—easily takes about 30 minutes per round. This app is built to collapse that manual back-and-forth into a single run.
- **Built for Local AI, Privacy, and Quality:** Lecture materials often contain sensitive or proprietary coursework. The goal of this project is to run on local AI (currently using local Whisper models for transcription, with a focus on local processing and strict prompt rules for cleaning) so your materials stay private without sacrificing the quality of the final notes.
