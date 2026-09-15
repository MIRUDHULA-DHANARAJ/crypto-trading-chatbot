import asyncio
import io

import edge_tts

from src.generator import client  


# Speech-to-text

def transcribe_audio(audio_bytes: bytes) -> str:
    transcription = client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes),
        model="whisper-large-v3",
        response_format="text",
    )
    return transcription.strip()


# Text-to-speech 

VOICE = "en-US-AriaNeural"  

async def _synthesize(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, VOICE)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


def synthesize_speech(text: str) -> bytes:
    return asyncio.run(_synthesize(text))