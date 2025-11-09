import os
import re
import sys
import json
import time
import queue
import torch
import asyncio
import logging
import uvicorn
import threading
import numpy as np
import stream2sentence as s2s
from loguru import logger
from datetime import datetime
from pydantic import BaseModel
from queue import Queue, Empty
from openai import AsyncOpenAI
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from supabase import create_client, Client
from concurrent.futures import ThreadPoolExecutor
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator
from backend.RealtimeSTT.audio_recorder import AudioToTextRecorder
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from backend.RealtimeTTS.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator

SUPABASE_URL = "https://jslevsbvapopncjehhva.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TextChunk:
    message_id: str
    chunk_id: int
    text: str
    timestamp: float

@dataclass
class AudioChunk:
    message_id: str
    chunk_id: int
    audio_data: bytes
    is_first: bool
    is_last: bool
    timestamp: float

@dataclass
class AudioTokens:
    message_id: str
    chunk_id: int
    audio_tokens: torch.Tensor

class RequestPacket(BaseModel):
    temperature: float = 0.95
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 25
    frequency_penalty: float = 0.03
    presence_penalty: float = 0.03
    repetition_penalty: float = 1.03

##########################################
##--          Chat Manager            --##
##########################################

class ChatManager:
    """Conversation Management"""


    async def get_available_models():
        """list of available models from OpenRouter"""


    async def get_active_characters():
        """get characters in current chat"""


    async def active_character_names_to_match():
        """array of all possible name variations to match"""


    async def get_character_system_prompt():
        """get character prompts from Supabase"""


    async def get_conversation_history():
         """get current conversation history"""


    async def determine_who_responds():
        """parse previous message for mentioned characters"""

        # parse the last message/response in chat (user or character).
        # look for active characters that were mentioned (partial matches, first or last name, nicknames).
        # matches respond in the order they were mentioned.
        # we must receive all response text from matched character 1 before sending request to OpenRouter for matched character 2 ->
        # -> we do this so that full/most recent conversation history/context is included in message request ->
        # -> we can send request for character 2 while TTS audio is still playing for character 1 ->
        # -> we can also begin processing text response (and generating audio) for character 2 -> 
        # -> we just buffer for playback as soon as character 1's audio has finished playing.
        # characters can respond to other characters using same method of parsing last message for matches of other active characters ->
        # -> we obviously exclude the character that sent the message when parsing for matches ->
        # -> also, there are some slightly modified rules for character to character interacting ->
        # -> simply put, each active character is allowed one response (per conversational turn), at which point a human/user must respond ->
        # -> this way the conversation doesn't spiral out of control and into into an infinite loop of character replies ->
        # -> once a human responds, the turn resets and all active characters may respond once (if they are mentioned).


    async def injection_message():
        """(system) message that can dynamically or manually inserted at any time"""

        # will be triggered by frontend control - nth message, time based, or instant.



    def strip_character_tags(self, text: str) -> str:
        """Strip character tags from text for display/TTS purposes"""
        return re.sub(r'<[^>]+>', '', text).strip()



        
    async def send_character_message(self, messages: List[Dict], request_packet: RequestPacket) -> AsyncGenerator[str, None]:  # yields text strings

        request_packet = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": request_packet.temperature,
            "top_p": request_packet.top_p,
            "min_p": request_packet.min_p,
            "top_k": request_packet.top_k,
            "frequency_penalty": request_packet.frequency_penalty,
            "presence_penalty": request_packet.presence_penalty,
            "repetition_penalty": request_packet.repetition_penalty
        }

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key = "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",
        )
            
        stream = await client.chat.completions.create(request_packet)
            
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

#############################################
##--    Text to Audio Stream Pipeline    --##
#############################################

class TextToAudioStream:
    """"""

    async def prepare_voice_clone(self, voice: str):
        """Load reference audio and text for voice cloning"""

        audio_path = os.path.join(self.voice_dir, f"{voice}.wav")
        text_path = os.path.join(self.voice_dir, f"{voice}.txt")

        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        messages = [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]

        return messages

    async def prepare_voice_profile(self, scene_prompt: str, speaker_desc: str):
        """"""

        # write simple query to get scene_prompt and speaker_desc from Supabase.

        system_message = Message(role="system", 
                                content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>")


    async def get_voice_profile_audio_tokens():
        """get previously generated audio tokens/ids for voice consistency"""

        # audio tokens stored in Supabase "voices" table in column "audio_tokens" as serialized json.


    async def process_text_to_audio_streams(self, text_stream: AsyncGenerator[str, None], message_id)-> AsyncGenerator[AudioChunk, None]:
        """
        Main pipeline: text stream → audio chunks
        Manages three concurrent tasks:
        1. Feed text stream into sentence accumulator
        2. Generate audio from buffered sentences
        3. Stream PCM chunks to browser
        """

        text_queue = asyncio.Queue()
        audio_queue = asyncio.Queue()

        tasks = []
        try:
            # Task 1: Feed text stream into sentence chunks
            task1 = asyncio.create_task(self.feed_text_to_sentences(text_stream, text_queue, message_id))
            tasks.append(task1)
                
            # Task 2: Generate audio from text chunks (concurrent)
            task2 = asyncio.create_task(self.generate_audio(text_queue,audio_queue))
            tasks.append(task2)
                
            # Task 3: Stream PCM chunks (this is the generator that yields)
            async for audio_chunk in self.stream_pcm_chunks(audio_queue=audio_queue):
                yield audio_chunk
            
        finally:
            # Cleanup
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def feed_text_to_sentences(self, text_stream: AsyncGenerator[str, None], text_queue: asyncio.Queue, message_id: str,):
        """
        Consumes streaming text, buffers into complete sentences,
        puts TextChunk objects into queue
        """
        try:
            char_iter = CharIterator()
            thread_safe_char_iter = AccumulatingThreadSafeGenerator(char_iter)
            sentence_generator = s2s.generate_sentences(thread_safe_char_iter)

            # Background task to feed text into char iterator
            async def text_feeder():
                try:
                    async for text_chunk in text_stream:
                        char_iter.add(text_chunk)
                finally:
                    char_iter.stop()
            
            feed_task = asyncio.create_task(text_feeder())

            try:
                # Pull complete sentences and queue them
                for sentence in sentence_generator:
                    if sentence.strip():
                        chunk_id = self._next_chunk_id
                        self._next_chunk_id += 1
                        
                        text_chunk = TextChunk(
                            message_id=message_id,
                            chunk_id=chunk_id,
                            text=sentence,
                            timestamp=datetime.now().timestamp(),
                        )
                        
                        await text_queue.put(text_chunk)

                        logger.info(f"[TextChunk {chunk_id}] Queued: '{sentence[:50]}...'")
            finally:
                await feed_task
            
            # Signal end of text chunks
            await text_queue.put(None)

        except Exception as e:
            print(e)

    async def generate_audio(self, text_queue: asyncio.Queue, audio_queue: asyncio.Queue, voice=None):
        """
        Processes text chunks(sentences)
        Each chunk generates audio tokens which are queued with metadata.
        """
        tasks = set()
        try:
            while True:
                text_chunk = await text_queue.get()

                if text_chunk is None:
                    break

                #create task per sentence
                task = asyncio.creat_task(self.generate_audio_tokens(text_chunk, audio_queue, voice))
                tasks.add(task)

                await asyncio.gather(*tasks, return_exceptions=True)
                await audio_queue.put(None)
        
        except Exception as e:
            await audio_queue.put(None)
            print(e)
    
    async def generate_audio_tokens(self, text_chunk: TextChunk, audio_queue: asyncio.Queue, voice: str):
        """Generate audio tokens for a single text chunk"""
        chunk_id = text_chunk.chunk_id
        
        logger.info(f"[Generation {chunk_id}] Starting: '{text_chunk.text[:50]}...' ({text_chunk.character_name})")

        try:
            messages = await self.prepare_voice_clone(voice)

            messages.append(Message(role="user", content=text_chunk.text))

            chat_ml_sample = ChatMLSample(messages=messages)
            
            # Generate audio tokens
            async for delta in self.serve_engine.generate_delta_stream(
                chat_ml_sample=chat_ml_sample,
                temperature=0.7,
                top_p=0.95,
                force_audio_gen=True,
            ):
                if delta.audio_tokens is None:
                    continue

                await audio_queue.put(
                    AudioTokens(
                        message_id=text_chunk.message_id,
                        chunk_id=text_chunk.chunk_id,
                        audio_tokens=delta.audio_tokens,
                    )
                )

            await audio_queue.put(
                AudioTokens(
                    message_id=text_chunk.message_id,
                    chunk_id=text_chunk.chunk_id,
                    audio_tokens=None,  # sentinel for end-of-sentence
                )
            )
                
        except Exception as e:
            print(e)


    async def stream_pcm_chunks(self, audio_queue: asyncio.Queue) -> AsyncIterator[AudioChunk]:
       """decodes sentences and yields pcm audio chunks"""

 
##########################################
##--           FastAPI App            --##
##########################################

app = FastAPI(title="Real-time TTS Streaming Server")

# Global instances (initialize on startup)
client: AsyncOpenAI = None
serve_engine: HiggsAudioServeEngine = None


@app.on_event("startup")
async def startup_event():
    """Initialize clients and model on server startup"""
    global openai_client, serve_engine, tts_pipeline
    
    logger.info("Starting up TTS server...")
    
    # Initialize OpenAI client
    client = AsyncOpenAI()

    # Initialize TTS model
    serve_engine = HiggsAudioServeEngine(
        model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
        device="cuda",
    )

async def connect(self, websocket):
    self.connections.add(websocket)
    await self._send_json(websocket, {"type": "connected"})

def disconnect(self, websocket):
    self.connections.discard(websocket)

#########################################
##--            WebSocket            --##
#########################################

@app.websocket("/ws")
async def websocket_endpoint( self, websocket: WebSocket, data):
    await websocket.accept()
    await connect(websocket)

    try:
        while True:
            message = await websocket.receive()

            if 'bytes' in message:
                self.stt_manager.feed_audio(message['bytes'])

            elif 'text' in message:
                try:
                    data = json.loads(message['text'])
                    message_type = data.get("type")
                    if message_type == "start_listening":
                        self.stt_manager.start_listening()
                    elif message_type == "stop_listening":
                        self.stt_manager.stop_listening()
                    elif message_type == "chat_message":
                        await self.handle_chat_message(data)

                except Exception as e:
                    logger.error(f"Error handling message: {e}")


            # need to add:
            #   text stream from OpenRouter
            #   streaming audio chunks (pcm) send to client


    except WebSocketDisconnect:
        pass

    finally:
        disconnect(websocket)

##########################################
##--          Static Files            --##
##########################################

# Serve frontend files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)