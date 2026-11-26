from pymongo import AsyncMongoClient, ASCENDING, DESCENDING
from .config import get_settings

_client: AsyncMongoClient | None = None


def database():
    if _client is None:
        raise RuntimeError('MongoDB is not initialized')
    return _client[get_settings().mongodb_db]


async def connect_db():
    global _client
    settings = get_settings()
    _client = AsyncMongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=6500, tz_aware=True)
    try:
        await _client.admin.command('ping')
        db = database()
        await db.users.create_index('email', unique=True)
        await db.revoked_sessions.create_index('expires_at', expireAfterSeconds=0)
        await db.analyses.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
        await db.analyses.create_index([('user_id', ASCENDING), ('prediction', ASCENDING)])
    except Exception:
        await _client.close()
        _client = None
        raise


async def close_db():
    global _client
    if _client is not None:
        await _client.close()
        _client = None
