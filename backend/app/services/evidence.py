"""Real external evidence sources; no synthetic/fabricated citations."""
import asyncio
import html
import re
from functools import lru_cache
from urllib.parse import quote, urlparse
import httpx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # TF-IDF cosine fallback if FAISS is not installed
from app.config import get_settings


async def google_fact_checks(claim: str) -> tuple[list[dict], str]:
    key = get_settings().google_factcheck_api_key
    if not key:
        return [], 'not_configured'
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.get('https://factchecktools.googleapis.com/v1alpha1/claims:search', params={'query': claim[:300], 'key': key, 'pageSize': 6})
            response.raise_for_status()
            results = []
            for item in response.json().get('claims', []):
                for review in item.get('claimReview', [])[:2]:
                    url = review.get('url', '')
                    if not url.startswith(('https://', 'http://')):
                        continue
                    results.append({'title': review.get('title') or item.get('text') or 'Fact check', 'url': url,
                                    'publisher': review.get('publisher', {}).get('name') or urlparse(url).netloc,
                                    'snippet': item.get('text', '')[:600], 'rating': review.get('textualRating', ''),
                                    'source_type': 'fact_check', 'published_at': review.get('reviewDate')})
            return results[:8], 'connected'
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return [], 'provider_error'  # Never reveal request URL/API keys to the browser


async def tavily_search(claim: str) -> tuple[list[dict], str]:
    key = get_settings().tavily_api_key
    if not key:
        return [], 'not_configured'
    try:
        async with httpx.AsyncClient(timeout=14) as client:
            response = await client.post('https://api.tavily.com/search', headers={'Authorization': f'Bearer {key}'}, json={
                'query': claim[:350], 'topic': 'news', 'search_depth': 'basic', 'max_results': 6, 'include_answer': False})
            response.raise_for_status()
            results = []
            for item in response.json().get('results', []):
                url = item.get('url', '')
                if not url.startswith(('https://', 'http://')):
                    continue
                results.append({'title': item.get('title') or 'Search result', 'url': url,
                                'publisher': urlparse(url).netloc.removeprefix('www.'),
                                'snippet': (item.get('content') or '')[:700], 'rating': '',
                                'source_type': 'web_search', 'published_at': item.get('published_date')})
            return results, 'connected'
    except (httpx.HTTPError, ValueError) as e:
        return [], 'provider_error'  # Never reveal request URL/API keys to the browser


# Reporting words that start many headlines but rarely help a Wikipedia search.
_ATTRIBUTION_WORDS = {'scientists', 'confirm', 'confirms', 'confirmed', 'says', 'said', 'report', 'reports',
                      'reportedly', 'breaking', 'study', 'experts', 'according'}


def wikipedia_queries(claim: str) -> list[str]:
    """Keyword queries for Wikipedia search.

    Wikipedia search requires every word to match, so a full sentence returns
    unrelated pages. A short and a slightly longer keyword query are used so one
    of them is likely to match.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", claim)
             if w.lower() not in ENGLISH_STOP_WORDS and w.lower() not in _ATTRIBUTION_WORDS]
    queries = []
    for size in (4, 6):
        query = ' '.join(words[:size])
        if query and query not in queries:
            queries.append(query)
    return queries


async def wikipedia_search(claim: str) -> tuple[list[dict], str]:
    """Keyless evidence source: Wikipedia article search (MediaWiki API)."""
    queries = wikipedia_queries(claim)
    if not queries:
        return [], 'connected'
    try:
        async with httpx.AsyncClient(timeout=10, headers={'User-Agent': 'DeepShield/1.0 (https://localhost; local development)'}) as client:
            async def fetch(query):
                for attempt in range(3):   # Wikipedia answers 429 with a short Retry-After when it is asked too often
                    response = await client.get('https://en.wikipedia.org/w/api.php', params={
                        'action': 'query', 'list': 'search', 'srsearch': query, 'srlimit': 5, 'format': 'json', 'utf8': 1})
                    if response.status_code != 429 or attempt == 2:
                        return response
                    await asyncio.sleep(min(float(response.headers.get('retry-after', 5) or 5), 10))
            responses = await asyncio.gather(*[fetch(query) for query in queries])
            results = []
            for response in responses:
                response.raise_for_status()
                for item in response.json().get('query', {}).get('search', []):
                    title = item.get('title', '')
                    if not title:
                        continue
                    # Search snippets contain <span class="searchmatch"> markup.
                    snippet = html.unescape(re.sub(r'<[^>]+>', '', item.get('snippet', '')))
                    results.append({'title': title, 'url': 'https://en.wikipedia.org/wiki/' + quote(title.replace(' ', '_')),
                                    'publisher': 'Wikipedia', 'snippet': snippet[:700], 'rating': '',
                                    'source_type': 'web_search', 'published_at': None})
            if get_settings().wikipedia_intro and results:
                results = await _with_intros(client, results)
            return results, 'connected'
    except (httpx.HTTPError, ValueError):
        return [], 'provider_error'


def apply_intros(results: list[dict], extracts: dict[str, str]) -> list[dict]:
    """Replace each result's search fragment by its page's opening sentences when we have them."""
    for item in results:
        intro = (extracts.get(item['title']) or '').strip()
        if intro:
            item['snippet'] = intro[:700]
    return results


async def _with_intros(client, results: list[dict]) -> list[dict]:
    """The MediaWiki search snippet is a keyword-in-context fragment, not a statement; a page's lead paragraph is.
    One extra request for the top pages; on any problem the original snippets are kept."""
    titles = list(dict.fromkeys(item['title'] for item in results))[:8]
    try:
        response = await client.get('https://en.wikipedia.org/w/api.php', params={
            'action': 'query', 'prop': 'extracts', 'exintro': 1, 'explaintext': 1, 'exsentences': 4, 'exlimit': 'max',
            'titles': '|'.join(titles), 'format': 'json', 'utf8': 1})
        response.raise_for_status()
        pages = response.json().get('query', {}).get('pages', {}).values()
        return apply_intros(results, {page.get('title', ''): page.get('extract', '') for page in pages})
    except (httpx.HTTPError, ValueError, AttributeError):
        return results


async def with_wikipedia_intros(evidence: list[dict]) -> list[dict]:
    """Copies of the evidence in which Wikipedia items carry their page's opening sentences instead of the search
    fragment. Used for the evidence judge only, so stance and the Trust Score keep the evidence they were fitted on."""
    items = [dict(item) for item in evidence]
    wiki = [item for item in items if item.get('publisher') == 'Wikipedia' and item.get('title')]
    if not wiki:
        return items
    try:
        async with httpx.AsyncClient(timeout=8, headers={'User-Agent': 'DeepShield/1.0 (https://localhost; local development)'}) as client:
            await _with_intros(client, wiki)
    except httpx.HTTPError:
        pass
    return items


@lru_cache(maxsize=1)
def _embedder(name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval(), torch


def embed_texts(documents: list[str], name: str):
    """L2-normalised sentence embeddings (mean pooling) from the named model. Raises if the model cannot load."""
    tokenizer, model, torch = _embedder(name)
    with torch.no_grad():
        batch = tokenizer(documents, padding=True, truncation=True, max_length=256, return_tensors='pt')
        hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    dense = pooled.numpy().astype(np.float32)
    return dense / np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-9)


def _semantic_vectors(documents: list[str]):
    """Embeddings from EMBEDDING_MODEL, or None to use TF-IDF.
    Off by default: the Trust Score (C2) was fitted on TF-IDF relevance, so turning this on changes the
    relevance scale and the Trust Score should be refitted (research.eval_live / eval_trust)."""
    name = get_settings().embedding_model
    if not name:
        return None
    try:
        return embed_texts(documents, name)
    except Exception:   # model missing or offline: fall back to TF-IDF rather than failing the analysis
        return None


def rank_evidence(claim: str, items: list[dict]) -> list[dict]:
    """Rank evidence by relevance to the claim (fact checks first, top 8).

    Relevance is cosine similarity searched with FAISS: TF-IDF vectors by default, or sentence embeddings
    when EMBEDDING_MODEL is set (falls back to TF-IDF if the model cannot load).
    """
    dedup = {i['url']: i for i in items if i.get('url')}
    items = list(dedup.values())
    if not items:
        return []
    documents = [claim] + [(i.get('title', '') + ' ' + i.get('snippet', '')).strip() for i in items]
    semantic = _semantic_vectors(documents)
    if semantic is not None:
        scores = semantic[1:] @ semantic[0]
        for item, score in zip(items, scores):
            item['relevance'] = round(float(score), 3)
        return sorted(items, key=lambda i: (i['source_type'] == 'fact_check', i['relevance']), reverse=True)[:8]
    try:
        vectors = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(documents)
        if faiss is not None:
            dense = np.asarray(vectors.toarray(), dtype=np.float32)
            faiss.normalize_L2(dense)
            index = faiss.IndexFlatIP(dense.shape[1])
            index.add(dense[1:])
            similarities, indices = index.search(dense[0:1], len(items))
            scores = np.zeros(len(items), dtype=np.float32)
            for similarity, index_id in zip(similarities[0], indices[0]):
                if index_id >= 0:
                    scores[index_id] = similarity
        else:
            scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        for item, score in zip(items, scores):
            item['relevance'] = round(float(score), 3)
    except ValueError:
        for item in items:
            item['relevance'] = 0.0
    return sorted(items, key=lambda i: (i['source_type'] == 'fact_check', i['relevance']), reverse=True)[:8]


async def gather_evidence(claim: str):
    (facts, fs), (web, ws), (wiki, wk) = await asyncio.gather(
        google_fact_checks(claim), tavily_search(claim), wikipedia_search(claim))
    return rank_evidence(claim, facts + web + wiki), {'google_factcheck': fs, 'tavily': ws, 'wikipedia': wk}
