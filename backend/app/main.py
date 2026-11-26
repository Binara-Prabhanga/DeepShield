import asyncio
import csv
import io
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, EmailStr, Field, field_validator
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from PIL import Image, ImageOps, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape

from .auth import hash_password, verify_password, issue_token, current_user, public_user, session_claims, LEGACY_SEED_EMAIL
from .config import get_settings
from .db import database, connect_db, close_db
from .services.article import extract_article, valid_public_url
from .services.assessment import assess
from .services.evidence import gather_evidence
from .services.llm import explain
from .services.model_adapter import run_models
from .services.preprocess import clean_text, extract_claim

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else dt


def present(doc):
    if not doc:
        return None
    result = {k: iso(v) for k, v in doc.items() if k not in ('user_id', '_id', 'is_demo', 'is_seed')}
    if 'prediction' in result:
        result['prediction'] = result['prediction'].removeprefix('DEMO · ')
    if doc.get('is_demo'):
        result['model_simulated'] = True
        if 'model' in result:
            result['model'] = dict(result['model'], mode='simulated', explanation_tokens=[])
        if 'signals' in result:
            result['signals'] = [
                dict(signal, kind='simulated') if signal['kind'] == 'demo' else signal
                for signal in result['signals']
            ]
    return result | {'id': doc['_id']}


def warm_up_models():
    """Load every configured model once, so the first analysis is not slow (the web app gives up after 100 s).
    Each part is optional and a failure here only means that part loads on first use instead."""
    import logging
    log = logging.getLogger('deepshield.warmup')
    settings = get_settings()
    steps = [('text model', lambda: run_models('Warm up sentence for the text model.', False))]
    if settings.stance_model:
        from .services.stance import classify_stance
        steps.append(('stance model', lambda: classify_stance('warm up', [{'title': 't', 'snippet': 'Warm up sentence.', 'source_type': 'web_search'}])))
    if settings.trust_model_path:
        from .services.trust import load_trust_model
        steps.append(('trust model', load_trust_model))
    if settings.domain_model_path:
        from .services.domain import out_of_domain
        steps.append(('domain check', lambda: out_of_domain('Warm up sentence.')))
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            log.warning('warm-up of the %s failed (%s); it will load on first use', name, type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    warm = asyncio.create_task(run_in_threadpool(warm_up_models))   # in the background: the API answers meanwhile
    try:
        yield
    finally:
        warm.cancel()
        await close_db()


app = FastAPI(title='DeepShield API', version='1.0.0', description='Evidence-assisted news investigation. Classifier outputs are simulated pending model training.', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().origins, allow_credentials=True,
                   allow_methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'], allow_headers=['Authorization', 'Content-Type'])


@app.exception_handler(ConnectionFailure)
async def database_unavailable(request, error):
    return JSONResponse({'detail': 'Database temporarily unavailable. Please try again.'}, status_code=503)


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class ProfileUpdate(BaseModel):
    name: str = Field(min_length=2, max_length=70)

    @field_validator('name', mode='before')
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class Register(Credentials, ProfileUpdate):
    pass


class NoteUpdate(BaseModel):
    notes: str = Field(max_length=3000)


@app.get('/api/health')
async def health():
    try:
        await database().command('ping')
        db_status = 'connected'
    except Exception:
        db_status = 'unavailable'
    healthy = db_status == 'connected'
    return JSONResponse(
        {'status': 'ok' if healthy else 'degraded', 'database': db_status,
         'model_mode': get_settings().model_mode, 'version': '1.0.0'},
        status_code=200 if healthy else 503,
    )


@app.post('/api/auth/register', status_code=201)
async def register(body: Register):
    if str(body.email).lower() == LEGACY_SEED_EMAIL:
        raise HTTPException(422, 'Use your own email address')
    password = body.password
    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        raise HTTPException(422, 'Password must contain letters and numbers')
    record = {'_id': str(uuid4()), 'email': str(body.email).lower(), 'name': body.name.strip(),
              'password': await run_in_threadpool(hash_password, password), 'role': 'analyst', 'created_at': utcnow()}
    try:
        await database().users.insert_one(record)
    except DuplicateKeyError:
        raise HTTPException(409, 'Email address already registered')
    return {'access_token': issue_token(record['_id']), 'token_type': 'bearer', 'user': public_user(record)}


@app.post('/api/auth/login')
async def login(body: Credentials):
    user = await database().users.find_one({'email': str(body.email).lower()})
    if not user or user['email'] == LEGACY_SEED_EMAIL or not await run_in_threadpool(verify_password, body.password, user['password']):
        raise HTTPException(401, 'Invalid email or password')
    return {'access_token': issue_token(user['_id']), 'token_type': 'bearer', 'user': public_user(user)}


@app.post('/api/auth/logout', status_code=204)
async def logout(session: dict = Depends(session_claims)):
    await database().revoked_sessions.update_one(
        {'_id': session['jti']},
        {'$setOnInsert': {'expires_at': datetime.fromtimestamp(session['exp'], tz=timezone.utc)}},
        upsert=True,
    )


@app.get('/api/auth/me')
async def me(user: dict = Depends(current_user)):
    return public_user(user)


@app.patch('/api/auth/profile')
async def profile(body: ProfileUpdate, user: dict = Depends(current_user)):
    await database().users.update_one({'_id': user['_id']}, {'$set': {'name': body.name.strip()}})
    new_user = await database().users.find_one({'_id': user['_id']})
    return public_user(new_user)


def image_info(image_bytes: bytes) -> tuple[str, dict]:
    import base64
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(413, 'Image must be smaller than 5 MB')
    try:
        source = Image.open(io.BytesIO(image_bytes))
        source.verify()
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ('JPEG', 'PNG', 'WEBP'):
            raise HTTPException(415, 'Upload a JPEG, PNG or WebP image')
        # Reject decompression bombs; safe web preview without GPS EXIF.
        if img.width * img.height > 24_000_000:
            raise HTTPException(413, 'Image dimensions too large')
        image_format, has_exif = img.format, bool(img.getexif())
        img = ImageOps.exif_transpose(img).convert('RGB')
        meta = {'width': img.width, 'height': img.height, 'format': image_format, 'size_bytes': len(image_bytes),
                'has_exif': has_exif, 'aspect_ratio': round(img.width / img.height, 3)}
        img.thumbnail((880, 560))
        img.info.clear()
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=78)
        return 'data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode(), meta
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, 'Image could not be decoded')


def categorize(score: int) -> str:
    if score >= 67:
        return 'Likely authentic'
    if score < 40:
        return 'Potentially misleading'
    return 'Needs review'


def decide(score: int, assessment: dict) -> tuple[int, str]:
    """Final score and label: the raw model score, then (in priority order) the calibrated Trust Score, a fact-checker's
    verdict on this very claim, the Gemini evidence judge (conservative), and the out-of-domain "needs review"."""
    prediction = categorize(score)
    if assessment['trust']:   # calibrated Trust Score (C2) replaces the raw score once TRUST_MODEL_PATH is set
        score, prediction = assessment['trust']['score'], assessment['trust']['decision']
    verdict = (assessment['judge'] or {}).get('verdict')
    if assessment['fact_check']:   # a fact-checker's verdict on this very claim beats any model score
        score = assessment['fact_check']['score']
        prediction = categorize(score)
    elif verdict == 'refuted':   # on LIAR2 every "refuted" was right, so it may only lower the score
        score = min(score, 20)
        prediction = categorize(score)
    elif verdict == 'supported' and assessment['out_of_domain']:   # plain facts: the text model is blind, the judge read the evidence
        score = max(score, 85)
        prediction = categorize(score)
    elif verdict == 'supported' and score < 40:   # in the model's domain, judge and model disagree: a person should look
        score, prediction = 50, 'Needs review'
    elif assessment['out_of_domain']:   # unlike the training data: say so rather than show a confident number
        score, prediction = 50, 'Needs review'
    return score, prediction


@app.post('/api/analyses', status_code=201)
async def create_analysis(
    text: Annotated[str, Form()] = '',
    url: Annotated[str, Form()] = '',
    image: UploadFile | None = File(None),
    user: dict = Depends(current_user),
):
    text = text.strip()
    url = url.strip()
    if url and not valid_public_url(url):
        raise HTTPException(422, 'Enter a valid public http(s) URL')
    extraction_error = None
    if not text and url:
        text, extraction_error = await extract_article(url)
    text = clean_text(text)
    if not text:
        raise HTTPException(422, extraction_error or 'Paste article text or provide an extractable article URL')
    if len(text) < 25 or len(text) > 16000:
        raise HTTPException(422, 'News text must be between 25 and 16,000 characters')
    img_data, img_meta, contents = None, {}, None
    if image and image.filename:
        contents = await image.read(5 * 1024 * 1024 + 1)
        img_data, img_meta = image_info(contents)
    model = await run_in_threadpool(run_models, text, img_data is not None, contents if img_data else None)
    model_kind = 'simulated' if model.mode == 'simulated' else 'model'
    claim = extract_claim(text)   # long articles: the most check-worthy sentence, not the whole document
    evidence, providers = await gather_evidence(claim)
    # C1-C3: stance, calibrated Trust Score, explanation audit. Each part is off until configured.
    evidence, assessment = await run_in_threadpool(assess, claim, evidence, model)
    prob = model.fused_probability_real
    score, prediction = decide(round(prob * 100), assessment)
    explanation, explanation_source = await explain(claim, evidence)
    domain = (urlparse(url).hostname or '') if url else ''
    metadata = {'source_domain': domain or None, 'https': url.startswith('https://') if url else None,
                'word_count': len(text.split()), 'image': img_meta,
                'source_notes': 'URL structure and uploaded image properties only; publisher reputation is not verified.'}
    signals = [
        {'name': 'Text analysis', 'score': round(model.text_probability_real * 100), 'kind': model_kind},
        {'name': 'Image alignment', 'score': round(model.image_text_alignment * 100) if model.image_text_alignment is not None else None,
         'kind': model_kind if model.image_text_alignment is not None else 'unavailable'},
        {'name': 'Retrieved references', 'score': None, 'value': len(evidence), 'kind': 'measured'},
        {'name': 'Source metadata', 'score': None, 'kind': 'unavailable'},
    ]
    record = {
        '_id': str(uuid4()), 'user_id': user['_id'], 'created_at': utcnow(),
        'title': re.sub(r'\s+', ' ', text[:95]).strip() + ('…' if len(text) > 95 else ''),
        'text': text, 'claim': claim, 'url': url, 'source_domain': domain, 'image_data': img_data,
        'image_metadata': img_meta, 'metadata': metadata, 'prediction': prediction,
        'trust_score': score, 'model_simulated': model.mode == 'simulated',
        'model': model.to_dict(), 'signals': signals,
        'evidence': evidence, 'providers': providers, 'explanation': explanation,
        'evidence_score': assessment['evidence_score'], 'trust': assessment['trust'],
        'explanation_audit': assessment['explanation_audit'], 'assessment_notes': assessment['notes'],
        'explanation_source': explanation_source, 'notes': '', 'tags': [],
    }
    await database().analyses.insert_one(record)
    return present(record)


@app.get('/api/analyses')
async def analyses(user: dict = Depends(current_user), page: int = Query(1, ge=1), limit: int = Query(12, ge=1, le=50),
                   q: str = Query('', max_length=150), category: str = Query('all')):
    criteria: dict = {'user_id': user['_id'], 'is_seed': {'$ne': True}}
    if q:
        criteria['$or'] = [{'title': {'$regex': re.escape(q), '$options': 'i'}}, {'source_domain': {'$regex': re.escape(q), '$options': 'i'}}]
    if category in ('Likely authentic', 'Needs review', 'Potentially misleading'):
        criteria['prediction'] = {'$in': [category, 'DEMO · ' + category]}
    count = await database().analyses.count_documents(criteria)
    cursor = database().analyses.find(criteria, {'image_data': 0, 'text': 0}).sort('created_at', -1).skip((page-1)*limit).limit(limit)
    items = [present(doc) async for doc in cursor]
    return {'items': items, 'total': count, 'page': page, 'pages': max(1, (count + limit - 1) // limit)}


async def find_owned(analysis_id: str, user_id: str) -> dict:
    doc = await database().analyses.find_one({'_id': analysis_id, 'user_id': user_id, 'is_seed': {'$ne': True}})
    if not doc:
        raise HTTPException(404, 'Analysis not found')
    return doc


@app.get('/api/analyses/{analysis_id}')
async def get_analysis(analysis_id: str, user: dict = Depends(current_user)):
    return present(await find_owned(analysis_id, user['_id']))


@app.patch('/api/analyses/{analysis_id}/notes')
async def update_note(analysis_id: str, body: NoteUpdate, user: dict = Depends(current_user)):
    result = await database().analyses.update_one(
        {'_id': analysis_id, 'user_id': user['_id'], 'is_seed': {'$ne': True}},
        {'$set': {'notes': body.notes}},
    )
    if not result.matched_count:
        raise HTTPException(404, 'Analysis not found')
    return {'notes': body.notes}


@app.delete('/api/analyses/{analysis_id}', status_code=204)
async def delete_analysis(analysis_id: str, user: dict = Depends(current_user)):
    result = await database().analyses.delete_one(
        {'_id': analysis_id, 'user_id': user['_id'], 'is_seed': {'$ne': True}},
    )
    if not result.deleted_count:
        raise HTTPException(404, 'Analysis not found')


@app.get('/api/dashboard/overview')
async def overview(user: dict = Depends(current_user)):
    now = utcnow()
    first_day = (now - timedelta(days=13)).replace(hour=0, minute=0, second=0, microsecond=0)
    cursor = await database().analyses.aggregate([
        {'$match': {'user_id': user['_id'], 'is_seed': {'$ne': True}}},
        {'$sort': {'created_at': -1}},
        {'$project': {
            'created_at': 1, 'trust_score': 1, 'prediction': 1, 'title': 1, 'source_domain': 1,
            'has_evidence': {'$gt': [{'$size': {'$ifNull': ['$evidence', []]}}, 0]},
        }},
        {'$facet': {
            'totals': [{'$group': {
                '_id': None, 'total': {'$sum': 1}, 'average_score': {'$avg': '$trust_score'},
                'evidence_found': {'$sum': {'$cond': ['$has_evidence', 1, 0]}},
            }}],
            'categories': [{'$group': {'_id': '$prediction', 'total': {'$sum': 1}}}],
            'days': [
                {'$match': {'created_at': {'$gte': first_day, '$lte': now}}},
                {'$group': {
                    '_id': {'$dateToString': {'date': '$created_at', 'format': '%Y-%m-%d'}},
                    'total': {'$sum': 1}, 'average': {'$avg': '$trust_score'},
                }},
            ],
            'recent': [{'$limit': 5}],
        }},
    ])
    result = await cursor.next()
    totals = next(iter(result['totals']), {})
    counts = {}
    for row in result['categories']:
        category = row['_id'].removeprefix('DEMO · ')
        counts[category] = counts.get(category, 0) + row['total']
    days = {row['_id']: row for row in result['days']}
    weekly = []
    for d in range(13, -1, -1):
        day = (now - timedelta(days=d)).date()
        daily = days.get(day.isoformat(), {})
        average = daily.get('average')
        weekly.append({'day': day.strftime('%d %b'), 'total': daily.get('total', 0),
                       'average': round(average) if average is not None else None})
    average_score = totals.get('average_score')
    return {'total': totals.get('total', 0), 'average_score': round(average_score) if average_score is not None else None,
            'needs_review': counts.get('Needs review', 0), 'evidence_found': totals.get('evidence_found', 0),
            'distribution': [{'name': k, 'value': counts.get(k, 0)} for k in ('Likely authentic', 'Needs review', 'Potentially misleading')],
            'activity': weekly, 'recent': [present(v) for v in result['recent']]}


@app.get('/api/integrations')
async def integrations(user: dict = Depends(current_user)):
    s = get_settings()
    return {'connectors': [
        {'id': 'google_factcheck', 'name': 'Google Fact Check Tools', 'category': 'Fact-checking', 'connected': bool(s.google_factcheck_api_key), 'description': 'Search published ClaimReview fact checks.', 'env_name': 'GOOGLE_FACTCHECK_API_KEY'},
        {'id': 'tavily', 'name': 'Tavily Search & Extract', 'category': 'Evidence retrieval', 'connected': bool(s.tavily_api_key), 'description': 'Retrieve news evidence and extract article URLs.', 'env_name': 'TAVILY_API_KEY'},
        {'id': 'gemini', 'name': 'Google Gemini', 'category': 'AI explanation', 'connected': bool(s.gemini_api_key), 'description': 'Generate constrained summaries from actual retrieved citations.', 'env_name': 'GEMINI_API_KEY'},
        {'id': 'mongo', 'name': 'MongoDB', 'category': 'Data storage', 'connected': True, 'description': 'Accounts with hashed passwords and saved analysis history.', 'env_name': 'MONGODB_URI'},
        {'id': 'colab', 'name': 'Custom BERT + CLIP', 'category': 'Local trained model', 'connected': False, 'description': 'Classifier inference is simulated while trained artifacts are unavailable.', 'env_name': 'MODEL_MODE'},
    ], 'model_mode': s.model_mode,
        'disclaimer': 'Classifications and trust indicators are simulated. Review retrieved references independently.'}


@app.get('/api/reports/export.csv')
async def export_csv(user: dict = Depends(current_user)):
    cursor = database().analyses.find(
        {'user_id': user['_id'], 'is_seed': {'$ne': True}},
        {'created_at': 1, 'title': 1, 'prediction': 1, 'trust_score': 1, 'evidence': 1, 'url': 1, 'notes': 1},
    ).sort('created_at', -1)

    def spreadsheet_text(value):
        return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value

    async def rows():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Created (UTC)', 'Headline', 'Simulated classification', 'Simulated trust score',
                         'Evidence count', 'Source URL', 'Notes'])
        yield output.getvalue()
        async with cursor:
            async for item in cursor:
                output.seek(0)
                output.truncate()
                writer.writerow([item['_id'], iso(item['created_at']), spreadsheet_text(item['title']),
                                 item['prediction'].removeprefix('DEMO · '), item['trust_score'],
                                 len(item.get('evidence', [])), spreadsheet_text(item.get('url', '')),
                                 spreadsheet_text(item.get('notes', ''))])
                yield output.getvalue()

    return StreamingResponse(rows(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="deepshield_history.csv"'})


@app.get('/api/analyses/{analysis_id}/report.pdf')
async def export_pdf(analysis_id: str, user: dict = Depends(current_user)):
    doc = await find_owned(analysis_id, user['_id'])
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=(595.27, 841.89), rightMargin=44, leftMargin=44, topMargin=44, bottomMargin=45)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='HeroDS', parent=styles['Title'], fontSize=20, leading=26, textColor=colors.HexColor('#10243c'), spaceAfter=14))
    styles.add(ParagraphStyle(name='BodyDS', parent=styles['BodyText'], leading=15, spaceAfter=10))
    story = [Paragraph('DEEPSHIELD / ANALYSIS REPORT', styles['HeroDS']),
             Paragraph('<b>MODEL SIMULATION</b>', styles['Heading2']), Spacer(1, 10),
             Paragraph(f"Created: {escape(iso(doc['created_at']))}", styles['BodyDS']),
             Paragraph(f"Headline: {escape(doc['title'])}", styles['BodyDS']),
             Paragraph(f"Simulated verdict: {escape(doc['prediction'].removeprefix('DEMO · '))}", styles['BodyDS']),
             Paragraph(f"Simulated trust score: {doc['trust_score']}/100", styles['BodyDS']),
             Paragraph('The classifier and trust indicator are simulated pending model training. Review the cited sources independently.', styles['BodyDS']),
             Paragraph('Submitted text', styles['Heading2']),
             Paragraph(escape(doc['text']).replace('\n', '<br/>'), styles['BodyDS']),
             Paragraph('Evidence and cited sources', styles['Heading2'])]
    if doc['evidence']:
        for index, ev in enumerate(doc['evidence'][:8], 1):
            story.append(Paragraph(f"{index}. {escape(ev['title'][:130])} — {escape(ev['publisher'][:100])}", styles['BodyDS']))
            story.append(Paragraph(escape(ev['url']), styles['BodyDS']))
    else:
        story.append(Paragraph('No external evidence was retrieved.', styles['BodyDS']))
    story += [Paragraph('Evidence explanation', styles['Heading2']), Paragraph(escape(doc['explanation'][:2500]), styles['BodyDS'])]
    if doc.get('notes'):
        story += [Paragraph('Analyst notes', styles['Heading2']), Paragraph(escape(doc['notes']), styles['BodyDS'])]
    pdf.build(story)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='application/pdf', headers={'Content-Disposition': f'attachment; filename="deepshield-{analysis_id[:8]}.pdf"'})
