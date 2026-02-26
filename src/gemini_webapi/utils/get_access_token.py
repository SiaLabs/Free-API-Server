import re

from httpx import AsyncClient, Response

from ..constants import Endpoint, Headers
from ..exceptions import AuthError
from .logger import logger


async def send_request(
    cookies: dict, proxy: str | None = None
) -> tuple[Response | None, dict]:
    """
    Send http request with provided cookies.
    """

    async with AsyncClient(
        proxy=proxy,
        headers=Headers.GEMINI.value,
        cookies=cookies,
        follow_redirects=True,
        verify=False,
    ) as client:
        response = await client.get(Endpoint.INIT.value)
        response.raise_for_status()
        return response, cookies


async def get_access_token(
    base_cookies: dict, proxy: str | None = None, verbose: bool = False
) -> tuple[str, dict]:
    """
    Send a get request to gemini.google.com with provided cookies and return
    the value of "SNlM0e" as access token.

    Parameters
    ----------
    base_cookies : `dict`
        Base cookies to be used in the request (from .env or browser extension).
    proxy: `str`, optional
        Proxy URL.
    verbose: `bool`, optional
        If `True`, will print more infomation in logs.

    Returns
    -------
    `str`
        Access token.
    `dict`
        Cookies of the successful request.

    Raises
    ------
    `gemini_webapi.AuthError`
        If request failed.
    """

    # Validate required cookies
    if "__Secure-1PSID" not in base_cookies or "__Secure-1PSIDTS" not in base_cookies:
        raise AuthError(
            "Missing required cookies. Please ensure __Secure-1PSID and __Secure-1PSIDTS are set in .env file."
        )

    try:
        response, cookies = await send_request(base_cookies, proxy=proxy)
        
        # Try multiple token formats - Google may use different tokens
        snlm0e = re.search(r'"SNlM0e":"(.*?)"', response.text)
        cfb2h = re.search(r'"cfb2h":"(.*?)"', response.text)
        fdrfje = re.search(r'"FdrFJe":"(.*?)"', response.text)
        
        if snlm0e or cfb2h or fdrfje:
            token = snlm0e.group(1) if snlm0e else (cfb2h.group(1) if cfb2h else fdrfje.group(1))
            if verbose:
                logger.success("Successfully obtained access token.")
            return token, cookies
        else:
            raise AuthError("Failed to extract access token from response.")
    except Exception as e:
        if verbose:
            logger.error(f"Failed to initialize: {e}")
        raise AuthError(
            "Failed to initialize client. Please ensure cookies are valid and up to date. "
            "The browser extension should automatically update cookies when they change."
        )
