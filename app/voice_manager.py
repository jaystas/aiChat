"""
Voice Manager Module
Handles all voice-related database operations with Supabase
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from supabase import Client
from fastapi import HTTPException
import logging
import json
import re
import threading

logger = logging.getLogger(__name__)


class Voice(BaseModel):
    voice_id: str
    voice: str
    method: str
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""
    audio_tokens: Optional[str] = None  # Serialized audio tokens (JSON)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VoiceCreate(BaseModel):
    voice: str
    method: str
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""


class VoiceUpdate(BaseModel):
    voice: Optional[str] = None
    method: Optional[str] = None
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    speaker_desc: Optional[str] = None
    scene_prompt: Optional[str] = None
    audio_tokens: Optional[str] = None


class VoiceManager:
    """Voice management service using Supabase with caching"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.table_name = "voices"
        self.voice_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
    
    def _generate_voice_id(self, voice_name: str) -> str:
        """
        Generate a sequential voice_id from the voice name.
        Format: lowercase-name-with-dashes-###
        Example: "Morgan Voice" -> "morgan-voice-001", "morgan-voice-002", etc.
        """
        # Convert to lowercase and replace spaces with dashes
        base_id = voice_name.lower().strip()
        base_id = re.sub(r'[^a-z0-9\s-]', '', base_id)  # Remove special chars
        base_id = re.sub(r'\s+', '-', base_id)  # Replace spaces with dashes
        base_id = re.sub(r'-+', '-', base_id)  # Remove multiple consecutive dashes
        base_id = base_id.strip('-')  # Remove leading/trailing dashes
        
        try:
            # Query existing IDs with the same base pattern
            response = self.supabase.table(self.table_name)\
                .select("voice_id")\
                .like("voice_id", f"{base_id}-%")\
                .execute()
            
            # Find the highest number in existing IDs
            highest_num = 0
            pattern = re.compile(rf"^{re.escape(base_id)}-(\d{{3}})$")
            
            for row in response.data:
                match = pattern.match(row["voice_id"])
                if match:
                    num = int(match.group(1))
                    highest_num = max(highest_num, num)
            
            # Generate next sequential number
            next_num = highest_num + 1
            voice_id = f"{base_id}-{next_num:03d}"
            
            logger.info(f"Generated voice_id: {voice_id}")
            return voice_id
            
        except Exception as e:
            logger.error(f"Error generating voice_id: {e}")
            # Fallback to 001 if there's an error
            return f"{base_id}-001"
    
    def _serialize_audio_tokens(self, tokens: Optional[list]) -> str:
        """Serialize audio tokens to JSON string"""
        if tokens is None:
            return "[]"
        return json.dumps(tokens)
    
    def _deserialize_audio_tokens(self, tokens_str: Optional[str]) -> list:
        """Deserialize audio tokens from JSON string"""
        if not tokens_str:
            return []
        try:
            return json.loads(tokens_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to deserialize audio tokens: {tokens_str}")
            return []
    
    async def get_all_voices(self) -> List[Voice]:
        """Get all voices from database"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .execute()
            
            voices = []
            for row in response.data:
                voice_data = {
                    "voice_id": row["voice_id"],
                    "voice": row["voice"],
                    "method": row["method"],
                    "audio_path": row.get("audio_path", ""),
                    "text_path": row.get("text_path", ""),
                    "speaker_desc": row.get("speaker_desc", ""),
                    "scene_prompt": row.get("scene_prompt", ""),
                    "audio_tokens": row.get("audio_tokens"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at")
                }
                voices.append(Voice(**voice_data))
            
            logger.info(f"Retrieved {len(voices)} voices from database")
            return voices
            
        except Exception as e:
            logger.error(f"Error getting all voices: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def get_voice(self, voice_id: str) -> Voice:
        """Get a specific voice by ID"""
        # Check cache first
        with self.cache_lock:
            if voice_id in self.voice_cache:
                logger.debug(f"Retrieved voice {voice_id} from cache")
                return self.voice_cache[voice_id]["config"]
        
        # Not in cache, fetch from database
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("voice_id", voice_id)\
                .execute()
            
            if not response.data:
                raise HTTPException(status_code=404, detail="Voice not found")
            
            row = response.data[0]
            voice_data = {
                "voice_id": row["voice_id"],
                "voice": row["voice"],
                "method": row["method"],
                "audio_path": row.get("audio_path", ""),
                "text_path": row.get("text_path", ""),
                "speaker_desc": row.get("speaker_desc", ""),
                "scene_prompt": row.get("scene_prompt", ""),
                "audio_tokens": row.get("audio_tokens"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            }
            
            voice = Voice(**voice_data)
            
            # Add to cache
            with self.cache_lock:
                self.voice_cache[voice_id] = {
                    "config": voice,
                    "audio_tokens": self._deserialize_audio_tokens(voice.audio_tokens)
                }
            
            logger.info(f"Retrieved voice {voice_id} from database and cached")
            return voice
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def create_voice(self, voice_data: VoiceCreate) -> Voice:
        """Create a new voice"""
        try:
            voice_id = self._generate_voice_id(voice_data.voice)
            
            db_data = {
                "voice_id": voice_id,
                "voice": voice_data.voice,
                "method": voice_data.method,
                "audio_path": voice_data.audio_path,
                "text_path": voice_data.text_path,
                "speaker_desc": voice_data.speaker_desc,
                "scene_prompt": voice_data.scene_prompt,
                "audio_tokens": "[]"  # Initialize with empty tokens
            }
            
            response = self.supabase.table(self.table_name)\
                .insert(db_data)\
                .execute()
            
            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create voice")
            
            voice = await self.get_voice(voice_id)
            
            # Add to cache
            with self.cache_lock:
                self.voice_cache[voice_id] = {
                    "config": voice,
                    "audio_tokens": []
                }
            
            logger.info(f"Created voice: {voice.voice} ({voice_id})")
            return voice
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating voice: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def update_voice(self, voice_id: str, voice_data: VoiceUpdate) -> Voice:
        """Update an existing voice"""
        try:
            update_data = {}
            if voice_data.voice is not None:
                update_data["voice"] = voice_data.voice
            if voice_data.method is not None:
                update_data["method"] = voice_data.method
            if voice_data.audio_path is not None:
                update_data["audio_path"] = voice_data.audio_path
            if voice_data.text_path is not None:
                update_data["text_path"] = voice_data.text_path
            if voice_data.speaker_desc is not None:
                update_data["speaker_desc"] = voice_data.speaker_desc
            if voice_data.scene_prompt is not None:
                update_data["scene_prompt"] = voice_data.scene_prompt
            if voice_data.audio_tokens is not None:
                update_data["audio_tokens"] = voice_data.audio_tokens
            
            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            response = self.supabase.table(self.table_name)\
                .update(update_data)\
                .eq("voice_id", voice_id)\
                .execute()
            
            if not response.data:
                raise HTTPException(status_code=404, detail="Voice not found")
            
            voice = await self.get_voice(voice_id)
            
            # Update cache
            with self.cache_lock:
                if voice_id in self.voice_cache:
                    self.voice_cache[voice_id]["config"] = voice
                    if voice_data.audio_tokens is not None:
                        self.voice_cache[voice_id]["audio_tokens"] = self._deserialize_audio_tokens(voice.audio_tokens)
            
            logger.info(f"Updated voice: {voice_id}")
            return voice
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice"""
        try:
            await self.get_voice(voice_id)
            
            response = self.supabase.table(self.table_name)\
                .delete()\
                .eq("voice_id", voice_id)\
                .execute()
            
            # Remove from cache
            with self.cache_lock:
                if voice_id in self.voice_cache:
                    del self.voice_cache[voice_id]
            
            logger.info(f"Deleted voice: {voice_id}")
            return True
            
        except HTTPException as e:
            if e.status_code == 404:
                raise
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting voice {voice_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    def get_cached_audio_tokens(self, voice_id: str) -> Optional[list]:
        """Get audio tokens from cache if available"""
        with self.cache_lock:
            if voice_id in self.voice_cache:
                return self.voice_cache[voice_id]["audio_tokens"]
        return None
    
    def update_cached_audio_tokens(self, voice_id: str, audio_tokens: list):
        """Update audio tokens in cache"""
        with self.cache_lock:
            if voice_id in self.voice_cache:
                self.voice_cache[voice_id]["audio_tokens"] = audio_tokens
    
    def clear_cache(self):
        """Clear the entire voice cache"""
        with self.cache_lock:
            self.voice_cache.clear()
        logger.info("Voice cache cleared")