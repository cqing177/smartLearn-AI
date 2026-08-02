import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You answer messages only from the supplied PDF text. "
    "Cite factual claims with [Page X]. "
    "If the answer is not in the PDF, say that the document does not provide enough information. "
    "Never invent a page number."
)


def answer_from_pages(pages: list[dict], message: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    document_text = "\n\n".join(
        f"### [Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"PDF text:\n{document_text}\n\nmessage: {message}"},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Cannot reach OpenRouter: {e}")
    except requests.exceptions.Timeout:
        raise RuntimeError("OpenRouter request timed out after 30s")
    except requests.exceptions.HTTPError as e:
        body = e.response.text[:300] if e.response is not None else "no body"
        raise RuntimeError(f"OpenRouter HTTP {e.response.status_code}: {body}")
    except Exception as e:
        raise RuntimeError(f"OpenRouter call failed: {type(e).__name__}: {e}")
