from __future__ import annotations

import os
from http.cookiejar import CookieJar
from typing import Iterable


try:
    import browser_cookie3  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency failure
    browser_cookie3 = None


_BROWSER_LOADERS = {
    "safari": "safari",
    "chrome": "chrome",
    "chromium": "chromium",
    "edge": "edge",
    "brave": "brave",
    "firefox": "firefox",
    "opera": "opera",
}


def _iter_cookie_header(jar: CookieJar) -> list[str]:
    items: list[str] = []
    for cookie in jar:
        if not cookie.name:
            continue
        items.append(f"{cookie.name}={cookie.value}")
    return items


def _load_from_browser(name: str, domain: str) -> CookieJar | None:
    if browser_cookie3 is None:
        return None
    loader_name = _BROWSER_LOADERS.get(name.lower())
    if not loader_name:
        return None
    loader = getattr(browser_cookie3, loader_name, None)
    if loader is None:
        return None
    try:
        return loader(domain_name=domain)
    except Exception:
        return None


def _order_from_env(env_value: str | None) -> list[str]:
    if not env_value:
        return ["safari", "chrome", "brave", "edge", "chromium", "firefox", "opera"]
    return [item.strip().lower() for item in env_value.split(",") if item.strip()]


def load_cookie_header(
    *,
    domain: str,
    order: Iterable[str] | None = None,
    env_cookie_header: str | None = None,
) -> tuple[str, str] | None:
    if env_cookie_header:
        return env_cookie_header.strip(), "manual"

    if browser_cookie3 is None:
        return None

    preferred = list(order or _order_from_env(os.environ.get("CONTEXT_RECON_BROWSER_COOKIE_ORDER")))
    for browser in preferred:
        jar = _load_from_browser(browser, domain)
        if jar is None:
            continue
        header_parts = _iter_cookie_header(jar)
        if not header_parts:
            continue
        return "; ".join(header_parts), browser
    return None
