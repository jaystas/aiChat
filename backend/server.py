import os
import re
import sys
import json
import time
import uuid
import queue
import torch
import uvicorn
import asyncio
import aiohttp
import logging
import requests
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
from threading import Thread, Event, Lock
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

from backend.character_manager import (CharacterManager, Character, CharacterCreate, CharacterUpdate)
from backend.voice_manager import (VoiceManager, Voice, VoiceCreate, VoiceUpdate)

SUPABASE_URL = "https://jslevsbvapopncjehhva.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelParameters(BaseModel):
    temperature: float = 1.01
    top_p: float = 0.95
    min_p: float = 0.03
    top_k: int = 25
    frequency_penalty: float = 0.03
    presence_penalty: float = 0.03
    repetition_penalty: float = 1.03

@dataclass
class AudioChunk:
    """Represents a single audio chunk for streaming playback"""
    chunk_id: str              # Unique chunk identifier (e.g., "msg-001-chunk-0")
    message_id: str            # Parent message ID
    character_id: str          # Which character is speaking
    character_name: str        # Character name for display
    audio_data: bytes          # PCM16 @ 24kHz audio data
    chunk_index: int           # Position in message (0, 1, 2...)
    is_final: bool             # Last chunk in this message?
    timestamp: float = field(default_factory=time.time)


#############################################
##--   Speech to Text Pipeline Manager   --##
#############################################

class STTManager:
    """"""

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

#############################################
##--            Chat Management          --##
#############################################

class ChatManager:
    """Manages chat conversations with OpenRouter LLM streaming"""

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.model: str = "anthropic/claude-3.5-sonnet"
        self.api_key: str = "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"

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        self.callbacks = {
            'text_stream_start': None,   # Called when character starts responding
            'text_chunk': None,           # Called for each LLM text chunk
            'text_stream_stop': None,     # Called when character finishes
        }

        # Model cache for fetch_available_models
        self._models_cache = {"timestamp": 0, "data": []}
        self._models_cache_ttl = 3600

    def set_callback(self, event_name, callback):
        """Set callback for specific event"""
        if event_name in self.callbacks:
            self.callbacks[event_name] = callback

    def set_model(self, model: str):
        """Set the OpenRouter model"""
        self.model = model

    def get_model(self) -> str:
        """Get current model"""
        return self.model

    async def fetch_available_models(self, force_refresh: bool = False) -> List[str]:
        """Fetch and cache available OpenRouter model identifiers."""
        now = time.time()
        cache = self._models_cache
        if not force_refresh and cache["data"] and now - cache["timestamp"] < self._models_cache_ttl:
            return cache["data"]

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get("https://openrouter.ai/api/v1/models", headers=headers) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise Exception(f"OpenRouter models API error: {response.status} - {detail}")
                payload = await response.json()

        data = payload.get("data") if isinstance(payload, dict) else []
        model_ids = {
            item.get("id")
            for item in data
            if isinstance(item, dict) and item.get("id")
        }
        models = sorted(model_ids)
        self._models_cache = {"timestamp": now, "data": models}
        return models
    
    async def character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create character instruction message with character tags"""
        return {
            "role": "system",
            "content": f"Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}'s reply. Do not respond for/as anyone else. Wrap your entire response in <{character.name}></{character.name}> tags."
        }
    
    async def conversation_history(self):
        """conversation history with ongoing messages added"""


    async def parse_character_mentions(self, message: str, active_characters: List[Character]) -> List[Character]:
        """Parse a message for character mentions in order of appearance"""
        mentioned_characters = []
        processed_characters = set()

        # Create an array of all possible name mentions with their positions
        name_mentions = []

        for character in active_characters:
            name_parts = character.name.lower().split()

            for name_part in name_parts:
                # Find all occurrences of this name part in the message
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, message, re.IGNORECASE):
                    name_mentions.append({
                        'character': character,
                        'position': match.start(),
                        'name_part': name_part
                    })

        # Sort by position in the message
        name_mentions.sort(key=lambda x: x['position'])

        # Add characters in order of first mention, avoiding duplicates
        for mention in name_mentions:
            if mention['character'].id not in processed_characters:
                mentioned_characters.append(mention['character'])
                processed_characters.add(mention['character'].id)

        # If no one was mentioned, all active characters respond (in order)
        if not mentioned_characters:
            mentioned_characters = sorted(active_characters, key=lambda character: character.name)

        return mentioned_characters
    
    async def build_chat_message(self):
        """add user message and build body of chat message for OpenRouter API"""


    async def send_chat_message(self, message: str, character: Character, user_name: str , active_characters: List[Character], model_params: ModelParameters = None, 
                               system_prompt: str = None):
        """send message to OpenRouter get streaming response"""

        # 
        # 1. get active characters.
        # 2. get the last message that was sent (user or character) - parse for mentions/matches of active characters.
        # 3. for mentioned characters (in order of mention) we send a chat message and get streaming response
        #  
        #    we need to build the chat message
        #
        #   add user message to conversation history.
        #   character's system_prompt
        #   conversation_history
        #   include character_instruction_message
        #   model parameters (get via client message)
        #   use client = AsyncOpenAI and get streaming response
        #
        #   on_text_stream_start callback

        return await self.character_response_stream

    async def character_response_stream(self, message: str, active_characters: List[Character], user_name: str = "Jay",
                              model_params: ModelParameters = None, system_prompt: str = None):
        """stream response from OpenRouter API"""

        text_queue = asyncio.Queue()

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )

        try:
            text_stream = await client.chat.completions.create(
            
            )

            async for text_chunk in text_stream:
                if text_chunk.choices[0].delta.content:
                    yield text_chunk.choices[0].delta.content


            # Create character iterator for sentence detection
            char_iter = CharIterator()
            thread_safe_char_iter = AccumulatingThreadSafeGenerator(char_iter)

            # Set up sentence generator
            sentence_generator = s2s.generate_sentences(
                thread_safe_char_iter,
            )

        except Exception as e:
            logger.error(f"Error streaming from OpenRouter: {e}")
    
##############################################
##--    Text to Speech Pipeline Manager   --##
##############################################

class TTSManager:
    """text to speech pipeline management"""


    def __init__(self):
        self.serve_engine = None
        self.voice_dir = "/workspace/tts/Code/backend/voices"
        self.sample_rate = 24000
        self._initialized = False
        self._engine_lock = Lock()
        self._chunk_overlap_duration = 0.04  # 40ms crossfade
        self._chunk_size = 16  # Tokens per streaming chunk
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize(self):
        if self._initialized:
            return
        logger.info("Initializing Higgs Audio TTS service...")

        try:

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.serve_engine = HiggsAudioServeEngine(
                model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
                audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
                device=device
            )

            self._initialized = True
            logger.info("Higgs Audio TTS service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Higgs Audio TTS: {e}")
            raise

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

        scene_prompt = f"a"
        speaker_desc = f""

        system_message = (f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>")

        messages = Message(role="system", 
                           content=system_message)

        return messages

    async def get_voice_profile_audio_tokens():
        """get previously generated audio tokens/ids for voice consistency"""

        # audio tokens stored in Supabase "voices" table in column "audio_tokens" as serialized json.



    async def stream_pcm_chunks(self, text: str, voice: str, scene_prompt: str, speaker_desc: str) -> AsyncIterator[bytes]:
        """Stream PCM16 audio chunks using delta token streaming"""

        if not self._initialized:
            raise RuntimeError("TTS Manager not initialized")

        # Need to add logic -> if method = clone, if method = profile
        messages = self.prepare_voice_clone(self, voice)

        messages = self.prepare_voice_profile(self, scene_prompt, speaker_desc)

        # Add user message with text to generate
        messages.append(Message(role="user", content=text))

        # Create ChatML sample
        chat_sample = ChatMLSample(messages=messages)

        # Stream generation with delta tokens
        try:
            streamer = self.serve_engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                stop_strings=['<|end_of_text|>', '<|eot_id|>'],
                ras_win_len=7,
                ras_win_max_num_repeat=2,
                force_audio_gen=True,
            )

            # Initialize streaming state
            audio_tokens: List[torch.Tensor] = []
            audio_tensor: Optional[torch.Tensor] = None
            seq_len = 0

            # Crossfade setup
            cross_fade_samples = int(self._chunk_overlap_duration * self.sample_rate)
            fade_out = np.linspace(1, 0, cross_fade_samples) if cross_fade_samples > 0 else None
            fade_in = np.linspace(0, 1, cross_fade_samples) if cross_fade_samples > 0 else None
            prev_tail: Optional[np.ndarray] = None

            with torch.inference_mode():
                async for delta in streamer:
                    # Skip if no audio tokens
                    if delta.audio_tokens is None:
                        continue

                    # Check for end token (1025)
                    if torch.all(delta.audio_tokens == 1025):
                        break

                    # Accumulate audio tokens
                    audio_tokens.append(delta.audio_tokens[:, None])
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    # Count sequence length (skip padding token 1024)
                    if torch.all(delta.audio_tokens != 1024):
                        seq_len += 1

                    # Decode and yield when chunk size reached
                    if seq_len > 0 and seq_len % self._chunk_size == 0:
                        try:
                            # Revert delay pattern and decode
                            vq_code = (
                                revert_delay_pattern(audio_tensor, start_idx=seq_len - self._chunk_size + 1)
                                .clip(0, 1023)
                                .to(self._device)
                            )
                            waveform_tensor = self.serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                            # Convert to numpy
                            if isinstance(waveform_tensor, torch.Tensor):
                                waveform_np = waveform_tensor.detach().cpu().numpy()
                            else:
                                waveform_np = np.asarray(waveform_tensor, dtype=np.float32)

                            # Apply crossfade
                            if prev_tail is None:
                                # First chunk
                                if cross_fade_samples > 0 and waveform_np.size > cross_fade_samples:
                                    chunk_head = waveform_np[:-cross_fade_samples]
                                    prev_tail = waveform_np[-cross_fade_samples:]
                                else:
                                    chunk_head = waveform_np
                                    prev_tail = None

                                if chunk_head.size > 0:
                                    # Convert to PCM16 and yield
                                    pcm = np.clip(chunk_head, -1.0, 1.0)
                                    pcm16 = (pcm * 32767.0).astype(np.int16)
                                    yield pcm16.tobytes()
                            else:
                                # Subsequent chunks with crossfade
                                if cross_fade_samples > 0 and waveform_np.size >= cross_fade_samples:
                                    overlap = prev_tail * fade_out + waveform_np[:cross_fade_samples] * fade_in
                                    middle = (
                                        waveform_np[cross_fade_samples:-cross_fade_samples]
                                        if waveform_np.size > 2 * cross_fade_samples
                                        else np.array([], dtype=waveform_np.dtype)
                                    )
                                    to_send = overlap if middle.size == 0 else np.concatenate([overlap, middle])

                                    if to_send.size > 0:
                                        # Convert to PCM16 and yield
                                        pcm = np.clip(to_send, -1.0, 1.0)
                                        pcm16 = (pcm * 32767.0).astype(np.int16)
                                        yield pcm16.tobytes()

                                    prev_tail = waveform_np[-cross_fade_samples:]
                                else:
                                    # Convert to PCM16 and yield
                                    pcm = np.clip(waveform_np, -1.0, 1.0)
                                    pcm16 = (pcm * 32767.0).astype(np.int16)
                                    yield pcm16.tobytes()

                        except Exception as e:
                            # Skip errors and continue
                            logger.warning(f"Error decoding audio chunk: {e}")
                            continue

            # Flush remaining tokens
            if seq_len > 0 and seq_len % self._chunk_size != 0 and audio_tensor is not None:
                try:
                    vq_code = (
                        revert_delay_pattern(audio_tensor, start_idx=seq_len - seq_len % self._chunk_size + 1)
                        .clip(0, 1023)
                        .to(self._device)
                    )
                    waveform_tensor = self.serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                    if isinstance(waveform_tensor, torch.Tensor):
                        waveform_np = waveform_tensor.detach().cpu().numpy()
                    else:
                        waveform_np = np.asarray(waveform_tensor, dtype=np.float32)

                    if prev_tail is None:
                        pcm = np.clip(waveform_np, -1.0, 1.0)
                        pcm16 = (pcm * 32767.0).astype(np.int16)
                        yield pcm16.tobytes()
                    else:
                        if cross_fade_samples > 0 and waveform_np.size >= cross_fade_samples:
                            overlap = prev_tail * fade_out + waveform_np[:cross_fade_samples] * fade_in
                            rest = waveform_np[cross_fade_samples:]
                            to_send = overlap if rest.size == 0 else np.concatenate([overlap, rest])

                            pcm = np.clip(to_send, -1.0, 1.0)
                            pcm16 = (pcm * 32767.0).astype(np.int16)
                            yield pcm16.tobytes()
                        else:
                            pcm = np.clip(waveform_np, -1.0, 1.0)
                            pcm16 = (pcm * 32767.0).astype(np.int16)
                            yield pcm16.tobytes()
                except Exception as e:
                    logger.warning(f"Error flushing remaining audio: {e}")

            # Yield final tail if exists
            if prev_tail is not None and prev_tail.size > 0:
                pcm = np.clip(prev_tail, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                yield pcm16.tobytes()

        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}")
            raise

    def get_available_voices(self):
        """Get list of available voices in format expected by frontend"""
        if not os.path.exists(self.voice_dir):
            return []

        voices = []
        for file in os.listdir(self.voice_dir):
            if file.endswith('.wav'):
                voice_name = file[:-4]  # Remove .wav extension
                # Only include if matching .txt file exists
                if os.path.exists(os.path.join(self.voice_dir, f"{voice_name}.txt")):
                    # Format: {id: "voice_name", name: "Voice Name"}
                    display_name = voice_name.replace('_', ' ').title()
                    voices.append({
                        "id": voice_name,
                        "name": display_name
                    })

        # Sort by display name
        voices.sort(key=lambda v: v['name'])
        return voices

    def shutdown(self):
        """Cleanup resources"""
        logger.info('Shutting down TTS manager')
        self.serve_engine = None
        self._initialized = False
    
    async def text_to_audio_stream(self):
        """complete pipeline"""


##############################################
##--           WebSocket Manager          --##
##############################################

class WebSocketManager:
    """"""
    def __init__(self, character_manager: CharacterManager, voice_manager: VoiceManager, stt_manager: STTManager, chat_manager: ChatManager, tts_manager: TTSManager ):

        self.character_magager=character_manager,
        self.voice_manager=voice_manager,
        self.stt_manager=stt_manager,
        self.chat_manager=chat_manager,
        self.tts_manager=tts_manager

        # Single WebSocket connection
        self.websocket = None

        # Connect STT callbacks to WebSocket broadcasting
        self.stt_manager.set_callback('stt_realtime_update', self.on_stt_realtime_update)
        self.stt_manager.set_callback('stt_realtime_stabilized', self.on_stt_realtime_stabilized)
        self.stt_manager.set_callback('stt_transcription_final', self.on_stt_transcription_final)

        # Connect LLM callbacks to WebSocket broadcasting
        self.chat_manager.set_callback('text_stream_start', self.on_text_stream_start)
        self.chat_manager.set_callback('text_chunk', self.on_text_chunk)
        self.chat_manager.set_callback('text_stream_stop', self.on_text_stream_stop)

        self.tts_manager.set_callback('audio_stream_start', self.on_audio_stream_start)
        self.tts_manager.set_callback('audio_chunk', self.on_audio_chunk)
        self.tts_manager.set_callback('audio_stream_stop', self.on_audio_stream_stop)

    async def connect(self, websocket: WebSocket):
        """Store the WebSocket connection"""
        self.websocket = websocket
        logger.info("WebSocket connected")

    def disconnect(self):
        """Clear the WebSocket connection"""
        self.websocket = None
        logger.info("WebSocket disconnected")

    async def send_json(self, event_type: str, data: Dict):
        """Send JSON message to client"""
        if self.websocket:
            try:
                message = {"type": event_type, "data": data}
                await self.websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending JSON: {e}")

    async def send_audio(self, audio_bytes: bytes):
        """Send audio bytes to client"""
        if self.websocket:
            try:
                await self.websocket.send_bytes(audio_bytes)
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    # Callback handlers
    async def on_stt_realtime_update(self, data):
        await self.send_json("stt_realtime_update", data)

    async def on_stt_realtime_stabilized(self, data):
        await self.send_json("stt_realtime_stabilized", data)

    async def on_stt_transcription_final(self, data):
        await self.send_json("stt_transcription_final", data)

    async def on_text_stream_start(self, data):
        await self.send_json("text_stream_start", data)

    async def on_text_chunk(self, text_chunk):
        await self.send_json("text_chunk", {"text": text_chunk})

    async def on_text_stream_stop(self, data):
        await self.send_json("text_stream_stop", data)

    async def on_audio_stream_start(self, data):
        await self.send_json("audio_stream_start", data)

    async def on_audio_chunk(self, audio_bytes):
        await self.send_audio(audio_bytes)

    async def on_audio_stream_stop(self, data):
        await self.send_json("audio_stream_stop", data)

    async def handle_client_message(self, message):
        """Route incoming JSON messages"""

        message_type = message.get("type")
            
        if message_type == "start_recording":
            self.stt_manager.start_recording()

        elif message_type == "stop_recording":
            self.stt_manager.stop_recording()

        elif message_type == "set_model":
            model = message.get("model")
            self.chat_manager.set_model(model)
            await self.send_json("model_set", {"model": model})

    async def handle_user_message(self, data):
        """Handle user message from client"""
        try:
            user_message = data.get("message", "")
            user_name = data.get("user_name", "Jay")
            
            await self.chat_manager.send_chat_message(user_message, user_name)

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def handle_character_message(self, data):
        """Handle character message to parse for mentions"""

            # get last (character) message this allows us to parse for mentions so that characters may respond to other characters.
            
#############################################
##--              FastAPI App            --##
#############################################

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    global websocket_manager, character_manager, voice_manager, stt_manager, chat_manager, tts_manager

    # Initialize all managers
    character_manager = CharacterManager(supabase)
    voice_manager = VoiceManager(supabase)
    stt_manager = STTManager()
    chat_manager = ChatManager()
    tts_manager = TTSManager(voice_manager)

    websocket_manager = WebSocketManager(
        character_manager=character_manager,
        voice_manager=voice_manager,
        stt_manager=stt_manager,
        chat_manager=chat_manager,
        tts_manager=tts_manager
    )

    yield

    if websocket_manager.websocket:
        websocket_manager.disconnect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#############################################
##--         Character API Routes        --##
#############################################

@app.get("/api/characters", response_model=List[Character])
async def get_all_characters():
    """Get all characters"""
    return await character_manager.get_all_characters()

@app.get("/api/characters/active", response_model=List[Character])
async def get_active_characters():
    """Get all active characters"""
    return await character_manager.get_active_characters()

@app.get("/api/characters/search", response_model=List[Character])
async def search_characters(q: str = Query(..., description="Search query")):
    """Search characters by name"""
    return await character_manager.search_characters(q)

@app.get("/api/characters/{character_id}", response_model=Character)
async def get_character(character_id: str):
    """Get a specific character by ID"""
    return await character_manager.get_character(character_id)

@app.post("/api/characters", response_model=Character)
async def create_character(character: CharacterCreate):
    """Create a new character"""
    return await character_manager.create_character(character)

@app.put("/api/characters/{character_id}", response_model=Character)
async def update_character(character_id: str, character: CharacterUpdate):
    """Update an existing character"""
    return await character_manager.update_character(character_id, character)

@app.put("/api/characters/{character_id}/active")
async def set_character_active(character_id: str, is_active: bool = Query(...)):
    """Set character active status"""
    character = await character_manager.set_character_active(character_id, is_active)
    return {"success": True, "character": character}

@app.delete("/api/characters/{character_id}")
async def delete_character(character_id: str):
    """Delete a character"""
    await character_manager.delete_character(character_id)
    return {"message": "Character deleted successfully"}

#############################################
##--            Chat API Routes          --##
#############################################

@app.post("/api/chat/send")
async def send_chat_message(
    message: str = Query(...),
    user_name: str = Query(default="Jay"),
    model_params: Optional[ModelParameters] = None,
    global_prompt: Optional[str] = None
):
    """Send a chat message to active characters"""
    try:
        result = await websocket_manager.chat_manager.send_chat_message(
            message, user_name, model_params, global_prompt
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/clear")
async def clear_conversation():
    """Clear conversation history"""
    websocket_manager.chat_manager.clear_conversation_history()
    return {"success": True, "message": "Conversation history cleared"}

@app.get("/api/chat/model")
async def get_current_model():
    """Get current chat model"""
    return {"model": websocket_manager.chat_manager.get_model()}

@app.put("/api/chat/model")
async def set_chat_model(model: str = Query(...)):
    """Set chat model"""
    websocket_manager.chat_manager.set_model(model)
    return {"success": True, "model": model}

#############################################
##--          Utility API Routes         --##
#############################################

@app.get("/api/openrouter/models")
async def list_openrouter_models(force_refresh: bool = Query(default=False, description="Force refresh cached model list")):
    try:
        models = await websocket_manager.chat_manager.fetch_available_models(force_refresh=force_refresh)
        return {"models": models}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/voices")
async def list_available_voices():
    return {"voices": websocket_manager.tts_manager.get_available_voices()}

#############################################
##--          WebSocket Endpoint         --##
#############################################

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive()
            
            if message.get('bytes'):
                websocket_manager.stt_manager.feed_audio(message['bytes'])
            
            elif message.get('text'):

                data = json.loads(message['text'])
                message_type = data.get('type')
                
                if message_type == 'chat_message':
                    await websocket_manager.handle_chat_message(data)
                    
                else:
                    await websocket_manager.handle_client_message(data)

    except WebSocketDisconnect:
        pass
    
    finally:
        websocket_manager.disconnect(websocket)

#########################################
##--          Static Files           --##
#########################################

# Serve frontend files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)