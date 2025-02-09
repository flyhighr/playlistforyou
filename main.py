from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import random
import string
from datetime import datetime
import re
import os
import asyncio
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv
import base64

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
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

client = AsyncIOMotorClient(MONGODB_URL)
db = client.playlist_db

spotify_token = None
token_expiry = None

async def get_spotify_token():
    global spotify_token, token_expiry
    current_time = datetime.utcnow()
    
    if spotify_token and token_expiry and token_expiry > current_time:
        return spotify_token
        
    try:
        auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')
        
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    spotify_token = result["access_token"]
                    token_expiry = current_time + datetime.timedelta(seconds=result["expires_in"] - 60)
                    return spotify_token
                else:
                    raise HTTPException(status_code=500, detail="Failed to get Spotify token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spotify authentication error: {str(e)}")

class SpotifyTrack(BaseModel):
    title: str
    artist: str
    cover_url: str
    spotify_id: str

@app.get("/api/search/songs/{query}")
async def search_songs(query: str):
    token = await get_spotify_token()
    
    url = f"https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "type": "track",
        "limit": 5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tracks = data["tracks"]["items"]
                    
                    results = []
                    for track in tracks:
                        results.append(SpotifyTrack(
                            title=track["name"],
                            artist=track["artists"][0]["name"],
                            cover_url=track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                            spotify_id=track["id"]
                        ))
                    return results
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to search Spotify")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Spotify: {str(e)}")

# Keep all existing code below this line
def generate_random_url(length=8):
    """Generate a random alphanumeric URL."""
    characters = string.ascii_lowercase + string.digits
    while True:
        random_url = ''.join(random.choices(characters, k=length))
        return random_url

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
    custom_url: Optional[str] = Field(None, min_length=3, max_length=50)
    sender_name: str = Field(..., min_length=1, max_length=100)
    welcome_message: str = Field(..., min_length=1, max_length=200)
    songs: List[Song] = Field(..., max_items=100)
    
    @validator('custom_url')
    def validate_custom_url(cls, v):
        if v is not None and not re.match(r'^[a-zA-Z0-9]+$', v):
            raise ValueError('Custom URL must contain only alphanumeric characters')
        return v.lower() if v else None

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

async def generate_unique_url():
    """Generate a unique random URL that doesn't exist in the database."""
    max_attempts = 5
    for _ in range(max_attempts):
        random_url = generate_random_url()
        existing = await db.playlists.find_one({"custom_url": random_url})
        if not existing:
            return random_url
    raise HTTPException(status_code=500, detail="Failed to generate unique URL")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/api/playlists")
async def create_playlist(playlist: PlaylistCreate):
    if playlist.custom_url:
        existing = await db.playlists.find_one({"custom_url": playlist.custom_url})
        if existing:
            raise HTTPException(status_code=400, detail="Custom URL already taken")
        final_url = playlist.custom_url
    else:
        final_url = await generate_unique_url()
    
    playlist_dict = playlist.dict()
    playlist_dict["custom_url"] = final_url
    playlist_dict["created_at"] = datetime.utcnow()
    
    result = await db.playlists.insert_one(playlist_dict)
    
    return {
        "message": "Playlist created successfully",
        "playlist_id": str(result.inserted_id),
        "custom_url": final_url
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
