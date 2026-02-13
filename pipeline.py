#!/usr/bin/env python3
"""
MULTILATERAL AI GOVERNANCE DAILY DIGEST PIPELINE (FREE)
========================================================
Zero API costs. Monitors 50+ sources across international organizations,
governments, think tanks, civil society, industry, and academia for
developments in AI governance, regulation, and policy.
"""

import os
import sys
import json
import yaml
import hashlib
import logging
import argparse
import smtplib
import re
import random
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "max_items_per_section": 5,
    "max_total_items": 30,
    "lookback_hours": 28,
    "request_timeout": 15,
    "digest_sections": [
        "multilateral",
        "regulation",
        "canada",
        "bangladesh",
        "safety",
        "rights_ethics",
        "standards",
        "geopolitics",
    ],
    "email_recipients": [],
    "archive_dir": "archive",
    "output_dir": "output",
}

KEYWORDS = {
    "high": [
        "AI governance", "AI regulation", "AI treaty", "AI convention",
        "EU AI Act", "AI safety institute", "frontier AI",
        "GPAI", "global partnership on AI", "AI advisory body",
        "responsible AI", "AI framework convention",
        "Hiroshima process", "AI safety summit", "Bletchley declaration",
        "OECD AI principles", "UNESCO AI recommendation",
        "AI executive order", "AIDA", "artificial intelligence and data act",
        "compute governance", "foundation model regulation",
        "general purpose AI", "high-risk AI",
        "pan-canadian AI strategy", "CIFAR", "Canada AI safety institute",
        "automated decision-making directive", "privacy commissioner AI",
        "smart Bangladesh", "digital Bangladesh", "Bangladesh AI strategy",
        "a2i programme", "Bangladesh national AI",
    ],
    "medium": [
        "AI policy", "AI ethics", "AI standards", "AI accountability",
        "algorithmic governance", "AI transparency", "AI audit",
        "AI risk management", "trustworthy AI", "AI alignment",
        "AI safety", "existential risk AI", "AI moratorium",
        "open source AI", "AI sovereignty", "digital sovereignty",
        "AI liability", "AI intellectual property",
        "AI and human rights", "AI bias", "AI discrimination",
        "AI surveillance", "facial recognition ban",
        "ISO 42001", "NIST AI RMF",
        "AI code of practice", "AI compliance",
    ],
    "low": [
        "artificial intelligence", "AI", "machine learning",
        "large language model", "LLM", "generative AI",
        "governance", "regulation", "multilateral",
        "United Nations", "OECD", "UNESCO", "ITU",
        "European Commission", "Council of Europe",
        "G7", "G20", "summit", "treaty", "convention",
        "policy", "framework", "standard",
    ],
}

AI_GOVERNANCE_QUIPS = [
    "Why did the AI governance framework cross the road? To get to the other signatory.",
    "How many stakeholders does it take to draft an AI treaty? We'll form a working group to find out.",
    "The EU AI Act walks into a bar. The bartender says 'Are you high-risk?' It replies, 'Depends on the annex.'",
    "What's an AI ethicist's favourite movie? The Unbearable Lightness of Regulation.",
    "GPAI, OECD, UNESCO, ITU — the only field where the acronyms outnumber the agreements.",
    "Why don't frontier models attend UN summits? They can't agree on alignment.",
    "AI governance is like a potluck — everyone brings principles, nobody brings enforcement.",
    "What did the OECD AI Principles say to the EU AI Act? 'I was soft law before it was cool.'",
    "How do you know an AI treaty is close? The footnotes are longer than the articles.",
    "Responsible AI: where 'we need more research' is both the problem and the conclusion.",
    "The Bletchley Declaration proved 28 countries can agree on AI risk. Next step: agreeing on anything else.",
    "AI safety summits: where existential risk meets catering logistics.",
    "Why did the algorithm refuse to testify before parliament? It claimed its weights were proprietary.",
    "ISO 42001: because nothing says 'innovation' like a management system standard.",
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SourceItem:
    title: str
    url: str
    source_name: str
    source_category: str
    published: Optional[datetime] = None
    snippet: str = ""
    relevance_score: float = 0.0
    keywords_matched: list = field(default_factory=list)
    digest_section: str = ""
    item_hash: str = ""

    def __post_init__(self):
        raw = f"{self.title.lower().strip()}{self.url.strip()}"
        self.item_hash = hashlib.md5(raw.encode()).hexdigest()


# ============================================================================
# FETCHER
# ============================================================================

class Fetcher:
    def __init__(self, config: dict, lookback: timedelta):
        self.config = config
        self.lookback = lookback
        self.cutoff = datetime.now(timezone.utc) - lookback
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Governance-Digest/1.0 (policy research)"
        })

    def fetch_rss(self, source: dict) -> list[SourceItem]:
        items = []
        try:
            feed = feedparser.parse(
                source.get("rss_url", source["url"]),
                request_headers={"User-Agent": self.session.headers["User-Agent"]},
            )
            for entry in feed.entries:
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                if pub_date and pub_date < self.cutoff:
                    continue
                snippet = ""
                if hasattr(entry, "summary"):
                    snippet = BeautifulSoup(entry.summary, "html.parser").get_text()[:500]
                items.append(SourceItem(
                    title=entry.get("title", "Untitled"),
                    url=entry.get("link", source["url"]),
                    source_name=source["name"],
                    source_category=source.get("_category", "general"),
                    published=pub_date,
                    snippet=snippet,
                ))
        except Exception as e:
            logging.warning(f"RSS fetch failed for {source['name']}: {e}")
        return items

    def fetch_api(self, source: dict) -> list[SourceItem]:
        items = []
        try:
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            api_url = source["api_url"].replace("{yesterday}", yesterday)
            resp = self.session.get(api_url, timeout=self.config["request_timeout"])
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", data if isinstance(data, list) else [])
            for doc in results:
                pub_date = None
                if "publication_date" in doc:
                    pub_date = datetime.strptime(
                        doc["publication_date"], "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc)
                items.append(SourceItem(
                    title=doc.get("title", "Untitled"),
                    url=doc.get("html_url", doc.get("url", source["url"])),
                    source_name=source["name"],
                    source_category=source.get("_category", "general"),
                    published=pub_date,
                    snippet=doc.get("abstract", doc.get("excerpt", ""))[:500],
                ))
        except Exception as e:
            logging.warning(f"API fetch failed for {source['name']}: {e}")
        return items

    def fetch_web_scrape(self, source: dict) -> list[SourceItem]:
        items = []
        try:
            resp = self.session.get(source["url"], timeout=self.config["request_timeout"])
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            selectors = [
                "article", ".news-item", ".press-release",
                ".post", ".entry", ".blog-post", "li.views-row",
            ]
            elements = []
            for sel in selectors:
                elements.extend(soup.select(sel))
            if not elements:
                elements = soup.find_all("a", href=True)
            for el in elements[:50]:
                title = el.get_text(strip=True)[:200]
                if len(title) < 10:
                    continue
                link = el.get("href", "")
                if link and not link.startswith("http"):
                    link = urljoin(source["url"], link)
                items.append(SourceItem(
                    title=title,
                    url=link or source["url"],
                    source_name=source["name"],
                    source_category=source.get("_category", "general"),
                    snippet=title,
                ))
        except Exception as e:
            logging.warning(f"Web scrape failed for {source['name']}: {e}")
        return items

    def fetch_source(self, source: dict) -> list[SourceItem]:
        source_type = source.get("type", "web_scrape")
        if source_type == "rss":
            return self.fetch_rss(source)
        elif source_type == "api":
            return self.fetch_api(source)
        else:
            return self.fetch_web_scrape(source)


# ============================================================================
# FILTER & SCORER
# ============================================================================

class Filter:
    WEIGHTS = {"high": 3, "medium": 2, "low": 1}

    def __init__(self, keywords: dict, source_keywords: list[str] = None):
        self.keywords = keywords
        self.source_keywords = source_keywords or []

    def score_item(self, item: SourceItem) -> SourceItem:
        text = f"{item.title} {item.snippet}".lower()
        score = 0.0
        matched = []
        for tier, terms in self.keywords.items():
            weight = self.WEIGHTS[tier]
            for term in terms:
                if term.lower() in text:
                    score += weight
                    matched.append(term)
        for kw in self.source_keywords:
            if kw.lower() in text:
                score += 1.5
                if kw not in matched:
                    matched.append(kw)
        item.relevance_score = score
        item.keywords_matched = matched
        return item

    def filter_items(self, items: list[SourceItem], min_score: float = 1.0) -> list[SourceItem]:
        scored = [self.score_item(item) for item in items]
        return [i for i in scored if i.relevance_score >= min_score]


# ============================================================================
# DEDUPLICATOR
# ============================================================================

class Deduplicator:
    def __init__(self):
        self.seen_hashes = set()
        self.seen_titles = []

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def _is_similar_title(self, title: str, threshold: float = 0.7) -> bool:
        norm = self._normalize(title)
        words_new = set(norm.split())
        if not words_new:
            return False
        for seen in self.seen_titles:
            words_seen = set(seen.split())
            if not words_seen:
                continue
            jaccard = len(words_new & words_seen) / len(words_new | words_seen)
            if jaccard >= threshold:
                return True
        return False

    def deduplicate(self, items: list[SourceItem]) -> list[SourceItem]:
        sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)
        unique = []
        for item in sorted_items:
            if item.item_hash in self.seen_hashes:
                continue
            if self._is_similar_title(item.title):
                continue
            self.seen_hashes.add(item.item_hash)
            self.seen_titles.append(self._normalize(item.title))
            unique.append(item)
        return unique


# ============================================================================
# CATEGORIZER
# ============================================================================

SECTION_RULES = {
    "multilateral": {
        "source_categories": ["international_orgs"],
        "keywords": ["UN", "OECD", "UNESCO", "ITU", "GPAI", "G7", "G20",
                      "multilateral", "treaty", "convention", "summit",
                      "declaration", "advisory body", "global compact",
                      "Hiroshima", "Bletchley", "Seoul"],
    },
    "regulation": {
        "source_categories": ["national_government"],
        "keywords": ["regulation", "legislation", "law", "act", "bill",
                      "EU AI Act", "executive order", "compliance",
                      "enforcement", "liability", "mandatory", "requirement"],
    },
    "canada": {
        "source_categories": ["canada_ai"],
        "keywords": ["canada", "canadian", "AIDA", "CIFAR", "Mila", "Vector",
                      "Amii", "pan-canadian", "Ottawa", "ISED",
                      "privacy commissioner", "automated decision",
                      "treasury board", "INDU committee", "Montreal",
                      "Alberta", "Ontario", "Quebec", "IRPP", "CIGI"],
    },
    "bangladesh": {
        "source_categories": ["bangladesh_ai"],
        "keywords": ["bangladesh", "bangladeshi", "Dhaka", "a2i",
                      "digital Bangladesh", "smart Bangladesh",
                      "ICT division", "BTRC", "BRAC", "CPD",
                      "fourth industrial", "hi-tech park",
                      "Bangladesh computer council"],
    },
    "safety": {
        "source_categories": ["national_government", "industry"],
        "keywords": ["safety", "frontier", "evaluation", "testing", "red team",
                      "alignment", "existential", "catastrophic", "compute",
                      "foundation model", "general purpose", "capability"],
    },
    "rights_ethics": {
        "source_categories": ["civil_society", "international_orgs"],
        "keywords": ["rights", "ethics", "bias", "discrimination", "fairness",
                      "surveillance", "facial recognition", "transparency",
                      "accountability", "justice", "equity", "inclusion"],
    },
    "standards": {
        "source_categories": ["standards"],
        "keywords": ["standard", "ISO", "IEEE", "NIST", "framework",
                      "certification", "audit", "assessment", "benchmark",
                      "interoperability", "technical", "specification"],
    },
    "geopolitics": {
        "source_categories": ["think_tanks", "media", "academic"],
        "keywords": ["geopolitics", "sovereignty", "competition", "race",
                      "China", "US", "Europe", "export control", "chip",
                      "semiconductor", "compute", "strategic", "national security",
                      "open source", "decoupling"],
    },
}


def categorize_item(item: SourceItem) -> str:
    scores = {}
    text = f"{item.title} {item.snippet}".lower()
    for section, rules in SECTION_RULES.items():
        score = 0
        if item.source_category in rules["source_categories"]:
            score += 2
        for kw in rules["keywords"]:
            if kw.lower() in text:
                score += 1
        scores[section] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "geopolitics"


# ============================================================================
# PUBLISHER
# ============================================================================

SECTION_LABELS = {
    "multilateral": "🌐 Multilateral Governance",
    "regulation": "📜 National Regulation",
    "canada": "🍁 Canada AI Governance",
    "bangladesh": "🇧🇩 Bangladesh AI & Digital",
    "safety": "🛡️ AI Safety",
    "rights_ethics": "⚖️ Rights & Ethics",
    "standards": "📐 Standards & Frameworks",
    "geopolitics": "🌍 Geopolitics & Strategy",
}


def generate_html_digest(items_by_section: dict, date: str) -> str:
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("digest.html")

    joke = random.choice(AI_GOVERNANCE_QUIPS)
    total_items = sum(len(v) for v in items_by_section.values())
    source_names = set()
    for items in items_by_section.values():
        for item in items:
            source_names.add(item.source_name)

    return template.render(
        date=date,
        joke=joke,
        sections=items_by_section,
        section_labels=SECTION_LABELS,
        total_items=total_items,
        total_sources=len(source_names),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def send_email(html: str, subject: str, recipients: list[str]):
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    from_addr = os.environ.get("SMTP_FROM", smtp_user)
    if not smtp_user or not smtp_pass:
        logging.info("SMTP not configured — skipping email delivery")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logging.info(f"Email sent to {len(recipients)} recipients")
    except Exception as e:
        logging.error(f"Email delivery failed: {e}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_sources(path: str = None) -> list[dict]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "sources.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)
    sources = []
    for category, source_list in raw.items():
        for source in source_list:
            source["_category"] = category
            sources.append(source)
    return sources


def run_pipeline(dry_run: bool = False, target_date: Optional[str] = None,
                 section_filter: Optional[str] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("pipeline")

    date_str = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log.info(f"=== Multilateral AI Governance Digest — {date_str} ===")

    sources = load_sources()
    log.info(f"Loaded {len(sources)} sources")

    lookback = timedelta(hours=CONFIG["lookback_hours"])
    fetcher = Fetcher(CONFIG, lookback)
    all_items = []
    for source in sources:
        items = fetcher.fetch_source(source)
        log.info(f"  {source['name']}: {len(items)} items")
        all_items.extend(items)
    log.info(f"Total fetched: {len(all_items)} items")

    relevant = []
    for source in sources:
        source_items = [i for i in all_items if i.source_name == source["name"]]
        source_filter = Filter(KEYWORDS, source.get("keywords", []))
        relevant.extend(source_filter.filter_items(source_items))
    log.info(f"After keyword filter: {len(relevant)} items")

    deduper = Deduplicator()
    unique = deduper.deduplicate(relevant)
    log.info(f"After dedup: {len(unique)} items")

    for item in unique:
        item.digest_section = categorize_item(item)

    items_by_section = {}
    for section in CONFIG["digest_sections"]:
        section_items = sorted(
            [i for i in unique if i.digest_section == section],
            key=lambda x: x.relevance_score, reverse=True,
        )[:CONFIG["max_items_per_section"]]
        if section_items:
            items_by_section[section] = section_items

    if section_filter:
        items_by_section = {k: v for k, v in items_by_section.items() if k == section_filter}

    total = sum(len(v) for v in items_by_section.values())
    log.info(f"Digest items: {total} across {len(items_by_section)} sections")

    if dry_run:
        log.info("=== DRY RUN ===")
        for section, items in items_by_section.items():
            print(f"\n### {SECTION_LABELS.get(section, section)}")
            for item in items:
                print(f"  [{item.relevance_score:.1f}] {item.title}")
                print(f"         {item.url}")
                print(f"         Keywords: {', '.join(item.keywords_matched[:5])}")
        return

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["archive_dir"], exist_ok=True)

    html = generate_html_digest(items_by_section, date_str)

    output_path = os.path.join(CONFIG["output_dir"], "index.html")
    with open(output_path, "w") as f:
        f.write(html)
    log.info(f"HTML saved to {output_path}")

    archive_path = os.path.join(CONFIG["archive_dir"], f"{date_str}.html")
    with open(archive_path, "w") as f:
        f.write(html)

    data_path = os.path.join(CONFIG["archive_dir"], f"{date_str}.json")
    with open(data_path, "w") as f:
        json.dump({
            "date": date_str,
            "sections": {
                section: [asdict(item) for item in items]
                for section, items in items_by_section.items()
            },
        }, f, indent=2, default=str)

    subject = f"🌐 AI Governance Digest — {date_str}"
    if CONFIG["email_recipients"]:
        send_email(html, subject, CONFIG["email_recipients"])

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilateral AI Governance Daily Digest (Free)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--section", type=str, default=None, choices=CONFIG["digest_sections"])
    args = parser.parse_args()
    run_pipeline(dry_run=args.dry_run, target_date=args.date, section_filter=args.section)
