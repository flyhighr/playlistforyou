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

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=ENVIRONMENT,
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

limiter = Limiter(key_func=get_remote_address)

spotify_token_cache = TTLCache(maxsize=1, ttl=3500) 
playlist_cache = TTLCache(maxsize=1000, ttl=300) 

app = FastAPI(
    title="Playlist Sharing API",
    description="Professional API for sharing music playlists",
    version="1.0.0"
)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

client = AsyncIOMotorClient(
    MONGODB_URL,
    maxPoolSize=50,
    minPoolSize=10,
    maxIdleTimeMS=45000,
    connectTimeoutMS=20000,
    serverSelectionTimeoutMS=30000,
    tlsCAFile=certifi.where()
)
db = client.playlist_db

# Metrics
Instrumentator().instrument(app).expose(app)

# Models
class SpotifyTrack(BaseModel):
    title: str
    artist: str
    cover_url: str
    spotify_id: str

class Song(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    artist: str = Field(..., min_length=1, max_length=200)
    cover_url: str = Field(..., min_length=1, max_length=500)
    youtube_url: str = Field(..., min_length=1, max_length=500)
    
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

@app.get("/health")
async def health_check():
    try:
        await db.command("ping")
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

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
                        SpotifyTrack(
                            title=track["name"],
                            artist=track["artists"][0]["name"],
                            cover_url=track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                            spotify_id=track["id"]
                        )
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


@app.post("/api/playlists")
@limiter.limit("10/minute")
async def create_playlist(playlist: PlaylistCreate, request: Request):
    try:
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

@app.get("/api/playlists/{custom_url}")
@limiter.limit("60/minute")
@cache_response(ttl_seconds=300)
async def get_playlist(custom_url: str, request: Request):
    try:
        playlist = await db.playlists.find_one({"custom_url": custom_url.lower()})
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist not found")
        
        playlist["_id"] = str(playlist["_id"])
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
    if not re.match(r'^[a-zA-Z0-9]{3,50}$', custom_url):
        return {"available": False, "reason": "Invalid URL format"}
    
    try:
        existing = await db.playlists.find_one({"custom_url": custom_url.lower()})
        return {"available": existing is None}
    except Exception as e:
        logger.error(f"Error checking URL availability: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while checking URL availability"
        )

async def create_indexes():
    try:
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
        client.close()
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
