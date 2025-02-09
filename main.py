from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import re
import os
import asyncio
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Playlist Sharing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
API_URL = os.getenv("API_URL", "https://ptrmoy.onrender.com") 
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "300"))  

client = AsyncIOMotorClient(MONGODB_URL)
db = client.playlist_db

class Song(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    artist: str = Field(..., min_length=1, max_length=200)
    cover_url: str = Field(..., min_length=1, max_length=500)
    youtube_url: str = Field(..., min_length=1, max_length=500)
    
    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        if not re.match(r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+$|^https?://youtu\.be/[\w-]+$', v):
            raise ValueError('Invalid YouTube URL')
        return v
    
    @validator('cover_url')
    def validate_cover_url(cls, v):
        if not re.match(r'^https?://.+$', v):
            raise ValueError('Invalid cover URL')
        return v

class PlaylistCreate(BaseModel):
    custom_url: str = Field(..., min_length=3, max_length=50)
    sender_name: str = Field(..., min_length=1, max_length=100)
    welcome_message: str = Field(..., min_length=1, max_length=200)
    songs: List[Song] = Field(..., max_items=100)
    
    @validator('custom_url')
    def validate_custom_url(cls, v):
        if not re.match(r'^[a-zA-Z0-9]+$', v):
            raise ValueError('Custom URL must contain only alphanumeric characters')
        return v.lower()

async def ping_self():
    """Periodically ping the health check endpoint to keep the service active."""
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"{API_URL}/health") as response:
                    if response.status == 200:
                        print(f"Self-ping successful at {datetime.utcnow()}")
                    else:
                        print(f"Self-ping failed with status {response.status}")
            except Exception as e:
                print(f"Self-ping error: {str(e)}")
            
            await asyncio.sleep(PING_INTERVAL)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/api/playlists")
async def create_playlist(playlist: PlaylistCreate):
    existing = await db.playlists.find_one({"custom_url": playlist.custom_url})
    if existing:
        raise HTTPException(status_code=400, detail="Custom URL already taken")
    
    playlist_dict = playlist.dict()
    playlist_dict["created_at"] = datetime.utcnow()
    
    result = await db.playlists.insert_one(playlist_dict)
    
    return {
        "message": "Playlist created successfully",
        "playlist_id": str(result.inserted_id),
        "custom_url": playlist.custom_url
    }

@app.get("/api/playlists/{custom_url}")
async def get_playlist(custom_url: str):
    playlist = await db.playlists.find_one({"custom_url": custom_url.lower()})
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    
    playlist["_id"] = str(playlist["_id"])
    return playlist

@app.get("/api/url-available/{custom_url}")
async def check_url_availability(custom_url: str):
    if not re.match(r'^[a-zA-Z0-9]{3,50}$', custom_url):
        return {"available": False, "reason": "Invalid URL format"}
    
    existing = await db.playlists.find_one({"custom_url": custom_url.lower()})
    return {"available": existing is None}

async def create_indexes():
    await db.playlists.create_index("custom_url", unique=True)
    await db.playlists.create_index("created_at")

@app.on_event("startup")
async def startup_event():
    await create_indexes()
    asyncio.create_task(ping_self()) 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
