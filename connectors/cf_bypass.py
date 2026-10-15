"""
Monkey-patch py_clob_client pour contourner le blocage Cloudflare.
Remplace le client HTTP (httpx) par curl_cffi qui imite l'empreinte TLS de Chrome.

Usage : importer ce module AVANT d'utiliser ClobClient.
    import connectors.cf_bypass
    from py_clob_client.client import ClobClient
    ...
"""
import json
from curl_cffi import requests as cf_requests
import py_clob_client.http_helpers.helpers as helpers
from py_clob_client.exceptions import PolyApiException

_session = cf_requests.Session(impersonate="chrome")


def _patched_request(endpoint: str, method: str, headers=None, data=None):
    headers = helpers.overloadHeaders(method, headers)
    headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    if isinstance(data, str):
        resp = _session.request(method, endpoint, headers=headers, data=data.encode("utf-8"))
    elif data is not None:
        resp = _session.request(method, endpoint, headers=headers, data=json.dumps(data).encode("utf-8"))
    else:
        resp = _session.request(method, endpoint, headers=headers)

    if resp.status_code != 200:
        raise PolyApiException(resp)

    try:
        return resp.json()
    except ValueError:
        return resp.text


# Patch appliqué à l'import
helpers.request = _patched_request
