from __future__ import annotations


def validate_gemini_rerank_payload(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Gemini payload must be a JSON object")

    snippets = payload.get("snippets")
    if not isinstance(snippets, list):
        raise ValueError("Gemini payload must include a snippets list")

    normalized: list[dict] = []
    for item in snippets:
        if not isinstance(item, dict):
            raise ValueError("Each snippet must be an object")
        snippet_id = item.get("id")
        score = item.get("score")
        rationale = item.get("rationale")

        if not isinstance(snippet_id, int):
            raise ValueError("snippet.id must be an integer")
        if not isinstance(score, (int, float)):
            raise ValueError("snippet.score must be numeric")
        if not isinstance(rationale, str):
            raise ValueError("snippet.rationale must be a string")

        normalized.append(
            {
                "id": snippet_id,
                "score": float(score),
                "rationale": rationale.strip(),
            }
        )

    return normalized
