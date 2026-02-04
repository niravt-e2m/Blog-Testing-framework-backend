"""Reference URL validation and content extraction service"""

import asyncio
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ReferenceValidationResult:
    """Result of validating a reference URL"""
    url: str
    accessible: bool
    response_time_ms: Optional[int] = None
    content_type: Optional[str] = None
    title: Optional[str] = None
    main_content: Optional[str] = None
    key_claims: Optional[List[str]] = None
    credibility_score: Optional[float] = None
    error: Optional[str] = None


class ReferenceValidator:
    """
    Service for validating and analyzing reference URLs.
    
    Responsibilities:
    - Validate URL accessibility
    - Scrape and extract content
    - Identify key claims from references
    - Cache results for performance
    """
    
    def __init__(
        self,
        timeout: int = 10,
        max_content_length: int = 50000,
        redis_client = None,
        cache_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize the reference validator.
        
        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to process
            redis_client: Optional Redis client for caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        
        # In-memory cache fallback
        self._memory_cache: Dict[str, tuple[ReferenceValidationResult, datetime]] = {}

    
    def _get_url_hash(self, url: str) -> str:
        """Generate a hash key for caching"""
        return hashlib.md5(url.encode()).hexdigest()
    
    async def _get_cached(self, url: str) -> Optional[ReferenceValidationResult]:
        """Get cached validation result"""
        cache_key = f"reference:{self._get_url_hash(url)}"
        
        # Try Redis first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    import json
                    data = json.loads(cached)
                    return ReferenceValidationResult(**data)
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Fall back to memory cache
        if url in self._memory_cache:
            result, timestamp = self._memory_cache[url]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return result
            else:
                del self._memory_cache[url]
        
        return None
    
    async def _set_cached(self, url: str, result: ReferenceValidationResult):
        """Cache validation result"""
        cache_key = f"reference:{self._get_url_hash(url)}"
        
        # Try Redis first
        if self.redis_client:
            try:
                import json
                data = {
                    "url": result.url,
                    "accessible": result.accessible,
                    "response_time_ms": result.response_time_ms,
                    "content_type": result.content_type,
                    "title": result.title,
                    "key_claims": result.key_claims,
                    "credibility_score": result.credibility_score,
                    "error": result.error,
                }
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
        
        # Also store in memory cache
        self._memory_cache[url] = (result, datetime.now())
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and supported"""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
        except Exception:
            return False
    
    async def validate_url(self, url: str) -> ReferenceValidationResult:
        """
        Validate a single URL and extract content.
        
        Args:
            url: URL to validate
            
        Returns:
            ReferenceValidationResult with validation details
        """
        # Check URL validity
        if not self._is_valid_url(url):
            return ReferenceValidationResult(
                url=url,
                accessible=False,
                error="Invalid URL format"
            )
        
        # Check cache first
        cached = await self._get_cached(url)
        if cached:
            logger.debug(f"Cache hit for {url}")
            return cached
        
        # Make request
        start_time = asyncio.get_event_loop().time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; BlogEvaluator/1.0)',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    },
                    allow_redirects=True,
                ) as response:
                    response_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
                    
                    if response.status != 200:
                        result = ReferenceValidationResult(
                            url=url,
                            accessible=False,
                            response_time_ms=response_time,
                            error=f"HTTP {response.status}"
                        )
                    else:
                        content_type = response.headers.get('content-type', '')
                        
                        # Only process HTML content
                        if 'text/html' in content_type or 'application/xhtml' in content_type:
                            html = await response.text()
                            html = html[:self.max_content_length]  # Limit content size
                            
                            # Parse and extract content
                            parsed = self._parse_html(html)
                            claims = self._extract_claims(parsed.get('content', ''))
                            credibility = self._assess_credibility(url, parsed)
                            
                            result = ReferenceValidationResult(
                                url=url,
                                accessible=True,
                                response_time_ms=response_time,
                                content_type='article',
                                title=parsed.get('title'),
                                main_content=parsed.get('content', '')[:5000],  # Limit stored content
                                key_claims=claims[:10],  # Limit claims
                                credibility_score=credibility,
                            )
                        else:
                            # Non-HTML content (PDF, etc.)
                            result = ReferenceValidationResult(
                                url=url,
                                accessible=True,
                                response_time_ms=response_time,
                                content_type=self._classify_content_type(content_type),
                                credibility_score=70.0,  # Default for non-HTML
                            )
                    
                    # Cache the result
                    await self._set_cached(url, result)
                    return result
                    
        except asyncio.TimeoutError:
            return ReferenceValidationResult(
                url=url,
                accessible=False,
                error=f"Timeout after {self.timeout}s"
            )
        except aiohttp.ClientError as e:
            return ReferenceValidationResult(
                url=url,
                accessible=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error validating {url}: {e}")
            return ReferenceValidationResult(
                url=url,
                accessible=False,
                error=f"Error: {str(e)}"
            )
    
    def _parse_html(self, html: str) -> Dict[str, Any]:
        """Parse HTML and extract structured content"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = None
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Try to find main content
        content = ""
        
        # Look for article or main content
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find('div', {'class': re.compile(r'content|article|post', re.I)}) or
            soup.find('div', {'id': re.compile(r'content|article|post', re.I)})
        )
        
        if main_content:
            content = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to body text
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return {
            'title': title,
            'content': content,
        }
    
    def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from content"""
        claims = []
        
        # Look for sentences with factual indicators
        factual_patterns = [
            r'[^.]*\d+%[^.]*\.',  # Percentages
            r'[^.]*\d{4}[^.]*\.',  # Years
            r'[^.]*\$[\d,]+[^.]*\.',  # Dollar amounts
            r'[^.]*according to[^.]*\.',  # Citations
            r'[^.]*study\s+(shows?|found|reveals?)[^.]*\.',  # Studies
            r'[^.]*research\s+(shows?|indicates?|suggests?)[^.]*\.',  # Research
            r'[^.]*statistics?\s+(show|indicate|suggest)[^.]*\.',  # Statistics
        ]
        
        for pattern in factual_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                claim = match.strip()
                if 20 < len(claim) < 500:  # Reasonable claim length
                    claims.append(claim)
        
        # Deduplicate and limit
        seen = set()
        unique_claims = []
        for claim in claims:
            claim_lower = claim.lower()
            if claim_lower not in seen:
                seen.add(claim_lower)
                unique_claims.append(claim)
        
        return unique_claims
    
    def _assess_credibility(self, url: str, parsed: Dict[str, Any]) -> float:
        """Assess credibility of a reference source"""
        score = 70.0  # Base score
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Boost for known credible domains
        credible_domains = [
            '.edu', '.gov', '.org',
            'wikipedia.org', 'britannica.com',
            'nature.com', 'science.org', 'sciencedirect.com',
            'reuters.com', 'apnews.com', 'bbc.com',
            'nytimes.com', 'washingtonpost.com',
            'docs.python.org', 'developer.mozilla.org',
        ]
        
        for cd in credible_domains:
            if cd in domain:
                score += 15
                break
        
        # Check for HTTPS
        if parsed_url.scheme == 'https':
            score += 5
        
        # Check content length (longer usually more substantive)
        content_length = len(parsed.get('content', ''))
        if content_length > 1000:
            score += 5
        if content_length > 5000:
            score += 5
        
        return min(100.0, score)
    
    def _classify_content_type(self, content_type: str) -> str:
        """Classify content type into friendly name"""
        if 'pdf' in content_type.lower():
            return 'pdf'
        elif 'json' in content_type.lower():
            return 'data'
        elif 'xml' in content_type.lower():
            return 'data'
        elif 'image' in content_type.lower():
            return 'image'
        else:
            return 'other'
    
    async def validate_urls(
        self, 
        urls: List[str],
        max_concurrent: int = 5,
    ) -> List[ReferenceValidationResult]:
        """
        Validate multiple URLs concurrently.
        
        Args:
            urls: List of URLs to validate
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of validation results
        """
        if not urls:
            return []
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_semaphore(url: str) -> ReferenceValidationResult:
            async with semaphore:
                return await self.validate_url(url)
        
        # Run validations concurrently
        tasks = [validate_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                processed_results.append(ReferenceValidationResult(
                    url=url,
                    accessible=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_all_claims(self, results: List[ReferenceValidationResult]) -> List[str]:
        """Get all claims from validation results"""
        all_claims = []
        for result in results:
            if result.key_claims:
                all_claims.extend(result.key_claims)
        return all_claims
