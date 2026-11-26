"""URL extraction is delegated to Tavily, avoiding server-side arbitrary URL fetch/SSRF."""
import httpx
from urllib.parse import urlparse
from app.config import get_settings


def valid_public_url(url: str) -> bool:
    from ipaddress import ip_address
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return False
    if len(url) > 2048 or port not in (None, 80, 443):
        return False
    if parsed.scheme not in {'https', 'http'} or not parsed.hostname or parsed.username or parsed.password:
        return False
    hostname = parsed.hostname.lower().rstrip('.')
    if hostname in {'localhost', 'localhost.localdomain'} or hostname.endswith(('.local', '.internal', '.localhost')):
        return False
    try:
        address = ip_address(hostname)
        if not address.is_global:
            return False
    except ValueError:
        pass
    return True


async def extract_article(url: str) -> tuple[str, str | None]:
    if not valid_public_url(url):
        return '', 'Invalid or non-public article URL'
    if not get_settings().tavily_api_key:
        return '', 'Tavily API key is required for URL-only extraction. You can also paste the article text.'
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post('https://api.tavily.com/extract',
                                  headers={'Authorization': f'Bearer {get_settings().tavily_api_key}'},
                                  json={'urls': [url], 'extract_depth': 'basic'})
            r.raise_for_status()
            result = r.json().get('results', [])
            if not result or not result[0].get('raw_content'):
                return '', 'No article content could be extracted from this URL'
            return result[0]['raw_content'][:9000], None
    except (httpx.HTTPError, ValueError):
        return '', 'Article extraction is unavailable. Paste the article text to continue.'
