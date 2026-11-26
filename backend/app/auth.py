from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .config import get_settings
from .db import database

_password = PasswordHash.recommended()
_bearer = HTTPBearer(auto_error=False)
LEGACY_SEED_EMAIL = 'demo@deepshield.ai'


def hash_password(password: str) -> str:
    return _password.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _password.verify(password, hashed)


def issue_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({'sub': user_id, 'jti': str(uuid4()), 'iat': now, 'exp': now + timedelta(days=7)}, get_settings().jwt_secret, algorithm='HS256')


def session_claims(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail='Please sign in')
    try:
        payload = jwt.decode(creds.credentials, get_settings().jwt_secret, algorithms=['HS256'],
                             options={'require': ['sub', 'jti', 'exp']})
        subject = payload.get('sub')
        if not isinstance(subject, str):
            raise ValueError('Missing subject')
    except (InvalidTokenError, ValueError):
        raise HTTPException(status_code=401, detail='Session expired or invalid')
    return payload


async def current_user(session: dict = Depends(session_claims)) -> dict:
    db = database()
    if await db.revoked_sessions.find_one({'_id': session['jti']}, {'_id': 1}):
        raise HTTPException(status_code=401, detail='Session ended')
    user = await db.users.find_one({'_id': session['sub']})
    if not user or user['email'] == LEGACY_SEED_EMAIL:
        raise HTTPException(status_code=401, detail='User no longer exists')
    return user


def public_user(user: dict) -> dict:
    return {'id': user['_id'], 'name': user['name'], 'email': user['email'], 'role': user.get('role', 'analyst'), 'created_at': user['created_at']}
