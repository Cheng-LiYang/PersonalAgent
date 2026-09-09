"""Read-only client for the public Yiguan calligraphy dictionary."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import time
from typing import Any

import httpx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from backend.models import MediaItem


_API_URL = "https://api.ygsf.com/v2.4/glyph/query"
_SITE_URL = "https://web.ygsf.com/#/home?VNK=7fc30e811~"
_AES_KEY = b"PkT!ihpN^QkQ62k%"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_INTENT_RE = re.compile(r"草书.*(?:写法|怎么写|查询|查找|图片|字形)|(?:写法|怎么写|查询|查找).{0,12}草书")
_WHOLE_POEM_RE = re.compile(r"(?:这|那|整|全)(?:一)?首(?:古)?(?:诗|词)|(?:全|整)(?:诗|词|文)|全文")
_DIRECT_TEXT_RE = re.compile(
    r"(?:上面|前面|上述|刚才)(?:的)?(?:这)?句(?:话|诗)?|这句(?:话|诗)?|这句话|这段(?:话|文字)|这行(?:字|诗)"
)


def _transparent_ink_touches_edge(content: bytes, edge_ratio: float = 0.035) -> bool:
    """Detect tightly cropped transparent glyphs that need the full-color fallback."""
    try:
        import pymupdf

        pixmap = pymupdf.Pixmap(content)
    except Exception:
        return False
    if not pixmap.alpha or pixmap.n < 4 or pixmap.width < 2 or pixmap.height < 2:
        return False
    width, height, channels = pixmap.width, pixmap.height, pixmap.n
    samples = pixmap.samples
    band = max(2, round(min(width, height) * edge_ratio))

    def visible(x: int, y: int) -> bool:
        return samples[(y * width + x) * channels + channels - 1] > 12

    for y in range(height):
        for x in range(band):
            if visible(x, y) or visible(width - 1 - x, y):
                return True
    for x in range(band, width - band):
        for y in range(band):
            if visible(x, y) or visible(x, height - 1 - y):
                return True
    return False


@dataclass(slots=True)
class PoemResolution:
    status: str
    title: str = ""
    author: str = ""
    content: str = ""
    lookup_text: str = ""
    note: str = ""


@dataclass(slots=True)
class CalligraphyIntent:
    """Structured result of model-first calligraphy intent analysis."""

    intent: str
    action: str
    query_text: str = ""
    work_title: str = ""
    needs_work_resolution: bool = False
    confidence: float = 0.0
    reason: str = ""
    model_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "action": self.action,
            "query_text": self.query_text,
            "work_title": self.work_title,
            "needs_work_resolution": self.needs_work_resolution,
            "confidence": self.confidence,
            "reason": self.reason,
            "model_used": self.model_used,
        }


def _clean_direct_request(question: str) -> str:
    """Remove request scaffolding without discarding inline first lines."""

    text = question.strip()
    prefix = re.compile(
        r"^(?:请|麻烦|帮我|请帮我)?\s*(?:查询|查找|查一下|看看|展示|显示|给出)?\s*"
        r"(?:下面|以下|下方)?\s*(?:这|该)?\s*"
        r"(?:一?段(?:文字|文本)?|些文字|句话|行字|首诗|首词|文字|文本|内容)?\s*"
        r"(?:的)?\s*草书(?:写法|怎么写|字形|图片)?\s*"
        r"(?:[-—=]*[>》→]+|[:：])?\s*",
        re.IGNORECASE,
    )
    cleaned = prefix.sub("", text, count=1)
    if cleaned != text:
        return cleaned
    separated = re.split(r"(?:[-—=]*[>》→]+|[:：])", text, maxsplit=1)
    if len(separated) == 2 and _INTENT_RE.search(separated[0]):
        return separated[1]
    return text


def extract_calligraphy_text(question: str, max_characters: int = 160) -> str | None:
    """Return the requested Chinese text when this is clearly a cursive lookup."""

    normalized = question.strip()
    if not _INTENT_RE.search(normalized):
        return None

    candidates: list[str] = []
    quoted = re.findall(r"[《「『“\"]([^》」』”\"]+)[》」』”\"]", normalized)
    candidates.extend(quoted)
    referenced = re.match(
        r"^(?P<text>.+?)[\s，,。]*(?:(?:上面|前面|上述|刚才)(?:的)?(?:这)?句(?:话|诗)?|这句(?:话|诗)?)(?:的)?草书(?:写法|怎么写)?",
        normalized,
    )
    if referenced:
        candidates.append(referenced.group("text"))
    cleaned = _clean_direct_request(normalized)
    if cleaned != normalized and _CJK_RE.search(cleaned):
        candidates.append(cleaned)
    parts = re.split(r"[:：]\s*|\n+", normalized, maxsplit=1)
    if len(parts) == 2 and _CJK_RE.search(parts[1]):
        candidates.append(parts[1])
    if not candidates:
        candidates.append(re.sub(
            r"(?:请|帮我|麻烦|Agent|agent|查询|查找|查一下|看看|展示|显示|给出|上面|前面|上述|刚才|下面|这首|这句话|这句|话|诗词?|内容|文字|的|草书|写法|怎么写|图片)",
            "",
            normalized,
            flags=re.IGNORECASE,
        ))

    candidate = max(candidates, key=lambda value: len(_CJK_RE.findall(value)))
    characters = _CJK_RE.findall(candidate)
    return "".join(characters[:max_characters])


def should_resolve_poem_title(question: str, extracted_text: str) -> bool:
    """Whether the user supplied a title but requested the complete work."""

    characters = _CJK_RE.findall(extracted_text)
    trailing_parts = re.split(r"[:：]\s*|\n+", question.strip())[1:]
    has_explicit_body = any(
        len(_CJK_RE.findall(part)) >= 4 and not _INTENT_RE.search(part)
        for part in trailing_parts
    )
    explicitly_requests_whole_work = bool(_WHOLE_POEM_RE.search(question))
    # Chinese book-title marks are a useful title signal.  Ordinary quotation
    # marks are also routinely used around a pasted verse, so treating all
    # quoted text as a title would send direct lines through the LLM.
    has_quoted_title = bool(re.search(r"《[^》]+》", question))
    directly_references_a_line = bool(_DIRECT_TEXT_RE.search(question))
    return (
        not has_explicit_body
        and bool(characters)
        and len(characters) <= 20
        and (explicitly_requests_whole_work or (has_quoted_title and not directly_references_a_line))
    )


def analyze_calligraphy_intent(
    model: Any,
    question: str,
    max_characters: int = 160,
    *,
    assume_intent: bool = False,
) -> CalligraphyIntent | None:
    """Analyze a calligraphy request as JSON, with a safe extraction fallback."""

    fallback_text = extract_calligraphy_text(question, max_characters)
    if fallback_text is None and not assume_intent:
        return None
    fallback_text = fallback_text or ""
    system = (
        "You are an intent router for a Chinese assistant. Return JSON only, never Markdown. "
        "Do not answer, rewrite, complete, translate, or invent the user's requested text."
    )
    prompt = (
        "Return exactly one object with this schema: "
        '{"intent":"calligraphy_lookup|other","action":"query_calligraphy_api|none",'
        '"query_text":"","work_title":"","needs_work_resolution":false,'
        '"confidence":0.0,"reason":""}. '
        "For pasted text, query_text must contain every pasted Chinese character in original order, including the "
        "first line after an arrow/colon or on the same line as the instruction. Remove only request wording and "
        "punctuation. For a title-only classical work, use its title and set needs_work_resolution=true. "
        "Never add characters absent from the request.\nUser request:\n" + question
    )
    try:
        raw = model.generate(system, prompt)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no JSON object")
        payload = json.loads(raw[start:end + 1])
        if not isinstance(payload, dict):
            raise ValueError("intent payload is not an object")
        intent = str(payload.get("intent") or "other").strip().lower()
        action = str(payload.get("action") or "none").strip().lower()
        query_text = "".join(_CJK_RE.findall(str(payload.get("query_text") or "")))[:max_characters]
        original_characters = "".join(_CJK_RE.findall(question))
        if intent != "calligraphy_lookup" or action != "query_calligraphy_api":
            raise ValueError("calligraphy route not selected")
        if query_text and query_text not in original_characters:
            raise ValueError("query_text was not copied from the user request")
        if not query_text and fallback_text:
            raise ValueError("explicit text was omitted")
        work_title = "".join(_CJK_RE.findall(str(payload.get("work_title") or "")))
        if work_title and work_title not in original_characters:
            work_title = ""
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        return CalligraphyIntent(
            intent="calligraphy_lookup",
            action="query_calligraphy_api",
            query_text=query_text,
            work_title=work_title,
            needs_work_resolution=bool(payload.get("needs_work_resolution", False)),
            confidence=confidence,
            reason=str(payload.get("reason") or "").strip(),
            model_used=True,
        )
    except Exception:
        needs_resolution = should_resolve_poem_title(question, fallback_text)
        return CalligraphyIntent(
            intent="calligraphy_lookup",
            action="query_calligraphy_api",
            query_text=fallback_text,
            work_title=fallback_text if needs_resolution else "",
            needs_work_resolution=needs_resolution,
            confidence=0.6,
            reason="model intent unavailable; deterministic fallback used",
            model_used=False,
        )


def resolve_poem_with_model(model: Any, question: str, title_hint: str, max_characters: int = 200) -> PoemResolution:
    """Ask the configured model to turn a poem title into verified-looking structured text."""

    system = (
        "You resolve Chinese classical poem and ci titles for a calligraphy lookup. "
        "Return JSON only; never include Markdown. Do not invent commentary, translations, or modern paraphrases."
    )
    prompt = (
        "The user wants the complete original work, but may have supplied only a title or cipai. "
        "Identify the intended work and return exactly one JSON object with keys: "
        '"status" (resolved, ambiguous, or not_found), "title", "author", "content", and "note". '
        "For a generic cipai with many works, select the best-known work normally meant by an unqualified request, "
        "and explain that choice briefly in note. content must contain the complete original body only, excluding "
        "title, author, translation, analysis, and Markdown. If you cannot identify one work confidently, use ambiguous "
        "and leave content empty.\n"
        f"Title hint: {title_hint}\nOriginal request: {question}"
    )
    raw = model.generate(system, prompt)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise CalligraphyLookupError("模型没有返回可解析的诗词信息")
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise CalligraphyLookupError("模型返回的诗词信息格式无效") from exc
    if not isinstance(payload, dict):
        raise CalligraphyLookupError("模型返回的诗词信息格式无效")

    status = str(payload.get("status") or "not_found").strip().lower()
    if status not in {"resolved", "ambiguous", "not_found"}:
        status = "not_found"
    title = str(payload.get("title") or title_hint).strip()
    author = str(payload.get("author") or "").strip()
    content = str(payload.get("content") or "").strip()
    note = str(payload.get("note") or "").strip()
    lookup_text = "".join(_CJK_RE.findall(content))
    normalized_title = "".join(_CJK_RE.findall(title))
    normalized_hint = "".join(_CJK_RE.findall(title_hint))
    if status == "resolved":
        if len(lookup_text) < 16:
            raise CalligraphyLookupError("模型没有生成足够完整的诗词正文")
        if lookup_text == normalized_title:
            raise CalligraphyLookupError("模型只返回了诗词标题，没有生成正文")
        if normalized_title and normalized_hint and not (
            normalized_hint in normalized_title or normalized_title in normalized_hint
        ):
            raise CalligraphyLookupError("模型返回的作品标题与用户请求不一致")
        if len(lookup_text) > max_characters:
            raise CalligraphyLookupError(f"诗词正文超过 {max_characters} 个汉字的查询上限")
    else:
        content = ""
        lookup_text = ""
    return PoemResolution(status, title, author, content, lookup_text, note)


class CalligraphyLookupError(RuntimeError):
    """The external dictionary could not return a usable result."""


class YiguanCalligraphyClient:
    """Query one representative cursive glyph per Chinese character."""

    def __init__(self, timeout_seconds: float = 20.0, max_workers: int = 6) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_workers = max(1, int(max_workers))

    def lookup_text(self, text: str) -> tuple[list[MediaItem], list[str]]:
        unique = list(dict.fromkeys(_CJK_RE.findall(text)))
        found: dict[str, MediaItem] = {}
        errors: dict[str, str] = {}
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(unique) or 1)) as pool:
                futures = {pool.submit(self._lookup_character, client, character): character for character in unique}
                for future in as_completed(futures):
                    character = futures[future]
                    try:
                        item = future.result()
                        if item is not None:
                            found[character] = item
                        else:
                            errors[character] = "未找到草书字形"
                    except Exception as exc:  # each missing glyph should not discard the rest
                        errors[character] = str(exc)

        media: list[MediaItem] = []
        emitted: set[str] = set()
        for position, character in enumerate(text):
            item = found.get(character)
            if item is None:
                continue
            first = character not in emitted
            emitted.add(character)
            media.append(MediaItem(
                id=item.id if first else f"{item.id}-repeat-{position}",
                type=item.type,
                mime_type=item.mime_type,
                data_base64=item.data_base64 if first else "",
                source_url=item.source_url,
                title=item.title,
                alt=item.alt,
                metadata={**item.metadata, "position": position, "resource_id": item.id},
            ))
        missing = [character for character in dict.fromkeys(text) if character in errors]
        return media, missing

    def _lookup_character(self, client: httpx.Client, character: str) -> MediaItem | None:
        started = time.perf_counter()
        payload = {
            "_plat": "web",
            "_channel": "pc",
            "_brand": "",
            "key": character,
            "kind": 1,
            "type": 2,
            "font": "草",
            "author": "",
            "orderby": "hot",
            "strict": 0,
            "loaded": 0,
            "_token": "",
        }
        encoded = _encrypt_payload(payload)
        response = client.post(_API_URL, data={"p": encoded})
        response.raise_for_status()
        decoded = _decrypt_response(response.text)
        if decoded.get("stat") != 0:
            raise CalligraphyLookupError("书法字典返回查询错误")
        rows = decoded.get("data", {}).get("list", [])
        if not rows:
            return None
        row = rows[0]
        clear_image_url = str(row.get("_clear_image") or "")
        color_image_url = str(row.get("_color_image") or "")
        image_url = clear_image_url or color_image_url
        if not image_url.startswith("https://"):
            return None
        image_response = client.get(image_url)
        image_response.raise_for_status()
        image_content = image_response.content
        content_type = image_response.headers.get("content-type", "image/png").split(";", 1)[0]
        image_variant = "clear" if image_url == clear_image_url else "color"
        if (
            image_variant == "clear"
            and color_image_url.startswith("https://")
            and _transparent_ink_touches_edge(image_content)
        ):
            try:
                color_response = client.get(color_image_url)
                color_response.raise_for_status()
                color_type = color_response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
                if len(color_response.content) <= 2_000_000 and color_type.startswith("image/"):
                    image_url = color_image_url
                    image_content = color_response.content
                    content_type = color_type
                    image_variant = "color"
            except httpx.HTTPError:
                pass
        if len(image_content) > 2_000_000:
            raise CalligraphyLookupError("字形图片超过大小限制")
        if not content_type.startswith("image/"):
            raise CalligraphyLookupError("字形资源不是图片")
        return MediaItem(
            id=f"ygsf-{row.get('_id') or character}",
            type="image",
            mime_type=content_type,
            data_base64=base64.b64encode(image_content).decode("ascii"),
            source_url=image_url,
            title=f"{character} · {row.get('_author') or '佚名'}",
            alt=f"{character}的草书字形",
            metadata={
                "character": character,
                "style": "草书",
                "author": row.get("_author") or "佚名",
                "work": row.get("_from") or "来源未注明",
                "dynasty": row.get("_dynasty") or "",
                "provider": "以观书法",
                "provider_url": _SITE_URL,
                "image_variant": image_variant,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )


def _encrypt_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padder = PKCS7(128).padder()
    padded = padder.update(raw) + padder.finalize()
    encryptor = Cipher(algorithms.AES(_AES_KEY), modes.ECB()).encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(encrypted).decode("ascii").replace("+", "-").replace("/", "_").replace("=", "!")


def _decrypt_response(value: str) -> dict[str, Any]:
    decryptor = Cipher(algorithms.AES(_AES_KEY), modes.ECB()).decryptor()
    decrypted = decryptor.update(base64.b64decode(value.strip())) + decryptor.finalize()
    unpadder = PKCS7(128).unpadder()
    raw = unpadder.update(decrypted) + unpadder.finalize()
    result = json.loads(raw.decode("utf-8"))
    if not isinstance(result, dict):
        raise CalligraphyLookupError("书法字典返回格式无效")
    return result
