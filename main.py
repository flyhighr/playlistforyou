from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, SecretStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import re
import os
import asyncio
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv
import base64
import logging
import logging.handlers
from urllib.parse import urlparse, parse_qs
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from cachetools import TTLCache
import certifi
from functools import wraps
from typing import Callable
import time
import random
import string
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

required_env_vars = [
    "MONGODB_URL",
    "API_URL",
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "SENTRY_DSN",
    "ENVIRONMENT"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

MONGODB_URL = os.getenv("MONGODB_URL")
API_URL = os.getenv("API_URL")
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "300"))
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
ALLOWED_ORIGINS = ["https://stmy.me"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'app.log',
            maxBytes=10485760,  
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=ENVIRONMENT,
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

# Database Manager
class DatabaseManager:
    def __init__(self, mongodb_url: str):
        self.client = AsyncIOMotorClient(
            mongodb_url,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=60000,
            connectTimeoutMS=30000,
            serverSelectionTimeoutMS=30000,
            socketTimeoutMS=45000,
            waitQueueTimeoutMS=30000,
            heartbeatFrequencyMS=20000,
            retryWrites=True,
            retryReads=True,
            tlsCAFile=certifi.where()
        )
        self.db = self.client.playlist_db
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        
    async def get_connection(self):
        while self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                await self.client.admin.command('ping')
                self._reconnect_attempts = 0
                return self.db
            except Exception as e:
                self._reconnect_attempts += 1
                logger.error(f"Database connection attempt {self._reconnect_attempts} failed: {str(e)}")
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    raise
                await asyncio.sleep(1 * self._reconnect_attempts)
                
    async def close(self):
        self.client.close()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize caches
spotify_token_cache = TTLCache(maxsize=1, ttl=3500)
playlist_cache = TTLCache(maxsize=1000, ttl=300)

# Initialize FastAPI app
app = FastAPI(
    title="Playlist Sharing API",
    description="Professional API for sharing music playlists",
    version="1.0.0"
)
app.state.limiter = limiter

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize database manager
db_manager = DatabaseManager(MONGODB_URL)

# Initialize Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Pydantic models
class SpotifyTrack(BaseModel):
    title: str
    artist: str
    cover_url: str
    spotify_id: str
    
class TimedMessage(BaseModel):
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: Optional[str] = Field(None, description="Optional end time in MM:SS format")
    message: str = Field(..., min_length=1, max_length=500)
    
    @validator('start_time', 'end_time')
    def validate_time_format(cls, v):
        if v is None:
            return v
        if not re.match(r'^[0-5][0-9]:[0-5][0-9]$', v):
            raise ValueError('Time must be in MM:SS format')
        return v
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if v is None:
            return v
        if 'start_time' in values:
            start_mins, start_secs = map(int, values['start_time'].split(':'))
            end_mins, end_secs = map(int, v.split(':'))
            if (end_mins * 60 + end_secs) <= (start_mins * 60 + start_secs):
                raise ValueError('End time must be after start time')
        return v

class Song(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    artist: str = Field(..., min_length=1, max_length=200)
    cover_url: str = Field(..., min_length=1, max_length=500)
    youtube_url: str = Field(..., min_length=1, max_length=500)
    timed_messages: Optional[List[TimedMessage]] = Field(default=[], max_items=50)
    
    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        clean_url = v.split('?')[0]
        if not re.match(r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+$|^https?://youtu\.be/[\w-]+$', clean_url):
            raise ValueError('Invalid YouTube URL')
        return clean_url
    
    @validator('cover_url')
    def validate_cover_url(cls, v):
        if not re.match(r'^https?://.+$', v):
            raise ValueError('Invalid cover URL')
        return v
    
    @validator('timed_messages')
    def validate_timed_messages(cls, v):
        if len(v) > 50:
            raise ValueError('Maximum 50 timed messages allowed per song')
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

# Utility functions and decorators
def cache_response(ttl_seconds: int = 300):
    def decorator(func: Callable):
        cache = TTLCache(maxsize=100, ttl=ttl_seconds)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            if cache_key in cache:
                return cache[cache_key]
            
            result = await func(*args, **kwargs)
            cache[cache_key] = result
            return result
        
        return wrapper
    return decorator

async def get_spotify_token():
    if 'token' in spotify_token_cache:
        return spotify_token_cache['token']
        
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
                    token = result["access_token"]
                    spotify_token_cache['token'] = token
                    return token
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to get Spotify token: {response.status}"
                    )
    except Exception as e:
        logger.error(f"Spotify authentication error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during Spotify authentication"
        )

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        f"Method: {request.method} Path: {request.url.path} "
        f"Status: {response.status_code} Duration: {duration:.2f}s"
    )
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

# Routes
@app.get("/health")
async def health_check():
    try:
        db = await db_manager.get_connection()
        await db.command("ping")
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

async def get_youtube_url(song_title: str, artist: str) -> Optional[str]:
    search_query = f"{song_title} {artist} official audio"
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'default_search': 'ytsearch5:',
    }
    
    try:
        def _search():
            with YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(search_query, download=False)
                if 'entries' in result and result['entries']:
                    best_match = None
                    highest_score = 0
                    
                    for video in result['entries']:
                        if not video:
                            continue
                            
                        title = video.get('title', '').lower()
                        score = 0
                        
                        if song_title.lower() in title:
                            score += 3
                        
                        if artist.lower() in title:
                            score += 2
                            
                        if any(term in title for term in ['official', 'audio', 'topic']):
                            score += 1
                            
                        if any(term in title for term in ['cover', 'live', 'karaoke', 'remix']):
                            score -= 2
                            
                        if score > highest_score:
                            highest_score = score
                            best_match = video
                    
                    if best_match and highest_score >= 3:
                        video_id = best_match['id']
                        return f"https://youtu.be/{video_id}"
            return None
            
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            url = await loop.run_in_executor(pool, _search)
            if url:
                logger.info(f"Found YouTube URL for {song_title} by {artist}: {url}")
            else:
                logger.warning(f"No YouTube URL found for {song_title} by {artist}")
            return url
    except Exception as e:
        logger.error(f"YouTube search error for {song_title} by {artist}: {str(e)}")
        return None

@app.get("/api/search/songs/{query}")
@limiter.limit("20/minute")
@cache_response(ttl_seconds=60)
async def search_songs(query: str, request: Request):
    token = await get_spotify_token()
    
    url = "https://api.spotify.com/v1/search"
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
                    
                    return [
                        {
                            "title": track["name"],
                            "artist": track["artists"][0]["name"],
                            "cover_url": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                            "spotify_id": track["id"]
                        }
                        for track in tracks
                    ]
                else:
                    logger.error(f"Spotify search failed: {response.status}")
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to search Spotify"
                    )
    except Exception as e:
        logger.error(f"Error searching Spotify: {str(e)}")
        raise HTTPException(
                        status_code=500,
                        detail="Internal server error during Spotify search"
                    )

@app.get("/api/youtube-url")
@limiter.limit("20/minute")
async def get_song_youtube_url(
    title: str,
    artist: str,
    request: Request
):
    try:
        url = await get_youtube_url(title, artist)
        if url:
            return {"youtube_url": url}
        else:
            return {"youtube_url": None}
    except Exception as e:
        logger.error(f"Error getting YouTube URL: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get YouTube URL"
        )

@app.post("/api/playlists")
@limiter.limit("10/minute")
async def create_playlist(playlist: PlaylistCreate, request: Request):
    try:
        db = await db_manager.get_connection()
        
        if playlist.custom_url:
            custom_url = playlist.custom_url.lower()
            existing = await db.playlists.find_one({"custom_url": custom_url})
            if existing:
                raise HTTPException(status_code=400, detail="Custom URL already taken")
            final_url = custom_url
        else:
            url_length = 5
            max_attempts = 10
            attempt = 0
            
            while attempt < max_attempts:
                chars = string.ascii_lowercase + string.digits
                chars = chars.replace('1', '').replace('l', '').replace('0', '').replace('o', '')
                
                final_url = ''.join(random.choices(chars, k=url_length))
                existing = await db.playlists.find_one({"custom_url": final_url})
                
                if not existing:
                    break
                    
                attempt += 1
                if attempt % 5 == 0:
                    url_length += 1
                    
            if attempt >= max_attempts:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to generate unique URL"
                )

        playlist_dict = playlist.dict()
        playlist_dict["custom_url"] = final_url
        playlist_dict["created_at"] = datetime.utcnow()
        
        result = await db.playlists.insert_one(playlist_dict)
        
        return {
            "message": "Playlist created successfully",
            "playlist_id": str(result.inserted_id),
            "custom_url": final_url
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating playlist: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while creating playlist"
        )


@app.get("/api/spotify-playlist/{playlist_url:path}")
@limiter.limit("10/minute")
@cache_response(ttl_seconds=300)
async def get_playlist_tracks(playlist_url: str, request: Request):
    try:
        playlist_id = playlist_url
        if "spotify.com" in playlist_url:
            parsed_url = urlparse(playlist_url)
            path_parts = parsed_url.path.split('/')
            playlist_id = path_parts[-1]
            
            if '?' in playlist_id:
                playlist_id = playlist_id.split('?')[0]
        
        logger.info(f"Processing Spotify playlist ID: {playlist_id}")
        
        all_tracks = []
        offset = 0
        
        while len(all_tracks) < 100:
            result = await get_spotify_playlist_tracks(playlist_id, offset)
            all_tracks.extend(result["tracks"])
            
            if offset + 50 >= result["total"] or offset + 50 >= 100:
                break
                
            offset += 50
            
            await asyncio.sleep(1)
        
        return {"tracks": all_tracks[:100]}  
        
    except Exception as e:
        logger.error(f"Error processing Spotify playlist: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process Spotify playlist"
        )

async def get_spotify_playlist_tracks(playlist_id: str, offset: int = 0):
    token = await get_spotify_token()
    
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "fields": "items(track(name,artists,album(images),id)),total",
        "limit": 50,
        "offset": offset
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tracks = []
                    
                    for item in data["items"]:
                        if item["track"] and item["track"]["name"]:
                            track = item["track"]
                            tracks.append({
                                "title": track["name"],
                                "artist": track["artists"][0]["name"] if track["artists"] else "Unknown Artist",
                                "cover_url": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                                "spotify_id": track["id"]
                            })
                    
                    return {
                        "tracks": tracks,
                        "total": data["total"]
                    }
                else:
                    logger.error(f"Spotify playlist fetch failed: {response.status} - {await response.text()}")
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to fetch Spotify playlist"
                    )
    except Exception as e:
        logger.error(f"Error fetching Spotify playlist: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while fetching Spotify playlist"
        )
        
@app.get("/api/playlists/{custom_url}")
@limiter.limit("60/minute")
@cache_response(ttl_seconds=300)
async def get_playlist(custom_url: str, request: Request):
    try:
        db = await db_manager.get_connection()
        playlist = await db.playlists.find_one({"custom_url": custom_url.lower()})
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")
        
        playlist["_id"] = str(playlist["_id"])
        for song in playlist["songs"]:
            if "timed_messages" not in song:
                song["timed_messages"] = []
            for message in song["timed_messages"]:
                start_mins, start_secs = map(int, message["start_time"].split(":"))
                message["start_seconds"] = start_mins * 60 + start_secs
                if message.get("end_time"):
                    end_mins, end_secs = map(int, message["end_time"].split(":"))
                    message["end_seconds"] = end_mins * 60 + end_secs
                else:
                    message["end_seconds"] = None
        
        return playlist
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving playlist: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving playlist"
        )


@app.get("/api/url-available/{custom_url}")
@limiter.limit("30/minute")
async def check_url_availability(custom_url: str, request: Request):
    try:
        if not re.match(r'^[a-zA-Z0-9]+$', custom_url):
            raise HTTPException(
                status_code=400,
                detail="Custom URL must contain only alphanumeric characters"
            )
        if len(custom_url) < 3:
            raise HTTPException(
                status_code=400,
                detail="Custom URL must be longer than 3 characters"
            )
            
            
        db = await db_manager.get_connection()
        existing = await db.playlists.find_one({"custom_url": custom_url.lower()})
        
        return {
            "available": not existing,
            "custom_url": custom_url.lower()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking URL availability: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while checking URL availability"
        )


async def create_indexes():
    try:
        db = await db_manager.get_connection()
        await db.playlists.create_index("custom_url", unique=True)
        await db.playlists.create_index("created_at")
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
        raise

async def ping_self():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"{API_URL}/health") as response:
                    if response.status == 200:
                        logger.info(f"Self-ping successful at {datetime.utcnow()}")
                    else:
                        logger.warning(f"Self-ping failed with status {response.status}")
            except Exception as e:
                logger.error(f"Self-ping error: {str(e)}")
            await asyncio.sleep(PING_INTERVAL)

@app.on_event("startup")
async def startup_event():
    try:
        await create_indexes()
        asyncio.create_task(ping_self())
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await db_manager.close()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": '%(levelprefix)s %(asctime)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            },
        }
    )
