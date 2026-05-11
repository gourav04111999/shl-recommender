"""
scraper.py — One-time catalog builder for SHL Individual Test Solutions.

Usage:
    pip install playwright beautifulsoup4
    playwright install chromium
    python scraper.py

Writes the scraped catalog to data/catalog.json, merging with any
existing data so manual additions are preserved.

Why Playwright?
    SHL's catalog page renders via JavaScript; simple HTTP requests
    return 403. Playwright controls a real headless browser and handles
    JS rendering, cookies, and anti-bot headers automatically.
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeoutError
except ImportError:
    print("Playwright not installed. Run:  pip install playwright && playwright install chromium")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not installed. Run:  pip install beautifulsoup4")
    sys.exit(1)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).parent / "data" / "catalog.json"

# SHL test-type letter codes (from their icon titles)
_TYPE_MAP = {
    "ability": "A",
    "cognitive": "A",
    "knowledge": "K",
    "skills": "K",
    "personality": "P",
    "behavioural": "B",
    "behavioral": "B",
    "motivation": "P",
    "situational": "S",
    "simulation": "S",
    "biodata": "B",
    "competency": "C",
    "development": "D",
}


def _infer_type(text: str) -> str:
    lower = text.lower()
    for keyword, code in _TYPE_MAP.items():
        if keyword in lower:
            return code
    return "A"


async def _scroll_to_bottom(page: Page):
    """Scroll page to trigger lazy-loaded content."""
    prev_height = 0
    for _ in range(20):
        height = await page.evaluate("document.body.scrollHeight")
        if height == prev_height:
            break
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.8)
        prev_height = height


async def _get_product_detail(page: Page, url: str) -> Optional[Dict[str, Any]]:
    """
    Visit an individual product page and extract:
    description, duration, remote testing flag, job levels.
    Returns None on failure (scraping is best-effort).
    """
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        html = await page.content()
        soup = BeautifulSoup(html, "lxml")

        # Description — meta or first substantial paragraph
        desc = ""
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            desc = meta_desc["content"].strip()
        else:
            for tag in soup.find_all(["p", "div"], class_=re.compile(r"desc|overview|intro", re.I)):
                text = tag.get_text(" ", strip=True)
                if len(text) > 60:
                    desc = text
                    break

        # Duration — look for patterns like "25 minutes" in the page
        duration = None
        duration_match = re.search(r"(\d+)\s*(?:–|-|to)\s*(\d+)\s*min", html, re.I)
        if duration_match:
            duration = int(duration_match.group(2))
        else:
            single = re.search(r"(\d+)\s*min", html, re.I)
            if single:
                duration = int(single.group(1))

        # Remote testing
        remote = "remote" in html.lower()

        return {
            "description": desc[:300],
            "duration_minutes": duration,
            "remote_testing": remote,
        }
    except Exception as exc:
        print(f"  [warn] could not fetch detail for {url}: {exc}")
        return None


async def scrape_catalog_page(page: Page, start: int = 0) -> List[Dict[str, Any]]:
    """
    Fetch one page of the catalog table (12 items per page).
    Returns list of raw item dicts.
    """
    url = f"{CATALOG_URL}?start={start}&type=1"  # type=1 = Individual Test Solutions
    try:
        await page.goto(url, wait_until="networkidle", timeout=30_000)
    except PWTimeoutError:
        # Partial load is still useful
        pass

    await _scroll_to_bottom(page)
    html = await page.content()
    soup = BeautifulSoup(html, "lxml")

    items: List[Dict[str, Any]] = []

    # SHL uses a <table> with class containing "catalogue" for its product listing
    table = soup.find("table", class_=re.compile(r"catalogue", re.I))
    if not table:
        # Fallback: any table with product-looking rows
        table = soup.find("table")

    if not table:
        print(f"  [warn] no table found at start={start}")
        return items

    for row in table.find_all("tr")[1:]:  # skip header
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        # First cell usually has the linked product name
        name_cell = cells[0]
        link = name_cell.find("a", href=True)
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link["href"]
        if not href.startswith("http"):
            href = BASE_URL + href

        # Test type icons — look for class attributes with type hints
        test_type = "A"
        for cell in cells[1:]:
            icons = cell.find_all(["span", "div", "img"])
            for icon in icons:
                title = (
                    icon.get("title", "")
                    or icon.get("alt", "")
                    or icon.get("aria-label", "")
                    or icon.get_text(strip=True)
                )
                if title:
                    inferred = _infer_type(title)
                    test_type = inferred
                    break

        items.append({"name": name, "url": href, "test_type": test_type})

    return items


async def _main():
    all_items: List[Dict[str, Any]] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = await ctx.new_page()

        # Page through the catalog (SHL shows 12 per page)
        start = 0
        consecutive_empty = 0
        while consecutive_empty < 2:
            print(f"Scraping catalog page (start={start}) ...")
            items = await scrape_catalog_page(page, start)
            if not items:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
                all_items.extend(items)
            start += 12
            await asyncio.sleep(1.2)

        # Enrich with detail pages
        print(f"\nEnriching {len(all_items)} items with detail pages ...")
        for idx, item in enumerate(all_items):
            print(f"  [{idx+1}/{len(all_items)}] {item['name']}")
            detail = await _get_product_detail(page, item["url"])
            if detail:
                item.update({k: v for k, v in detail.items() if v})
            await asyncio.sleep(0.8)

        await browser.close()

    # Remove duplicates by URL
    seen_urls: set = set()
    deduped: List[Dict[str, Any]] = []
    for item in all_items:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            deduped.append(item)

    # Merge with existing catalog (preserve manual additions)
    existing: List[Dict[str, Any]] = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as fh:
            existing = json.load(fh)

    existing_urls = {e["url"] for e in existing}
    for scraped in deduped:
        if scraped["url"] not in existing_urls:
            existing.append(scraped)
        else:
            # Update description if empty
            for ex in existing:
                if ex["url"] == scraped["url"] and not ex.get("description"):
                    ex.update(scraped)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(existing)} items saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(_main())
