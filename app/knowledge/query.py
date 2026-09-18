from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from typing import Protocol

from app.errors import ServiceError
from app.knowledge.contracts import QueryPlan
from app.knowledge.gateway import (
    NormalizationOutput,
    NormalizationRequestError,
    NormalizationResponseError,
)


logger = logging.getLogger(__name__)
_ALPHANUMERIC_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?=[A-Za-z0-9._/-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/-]*\d)"
    r"[A-Za-z0-9]+(?:[-._/][A-Za-z0-9]+)*"
    r"(?![A-Za-z0-9])"
)
_NUMBER = re.compile(r"(?<!\d)\d+(?:\.\d+)?(?!\d)")
_NEGATION = re.compile(
    r"不支持|不能|不可以|不会|没有|不得|不建议|不含|不附送|未|无|不(?!是)"
)


class _Gateway(Protocol):
    async def normalize(self, question: str) -> NormalizationOutput: ...


def _canonical(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _protected_identifiers(value: str) -> set[str]:
    canonical = _canonical(value)
    return {match.group(0) for match in _ALPHANUMERIC_IDENTIFIER.finditer(canonical)}


def _numbers(value: str) -> set[str]:
    canonical = _canonical(value)
    return {match.group(0) for match in _NUMBER.finditer(canonical)}


def protected_terms_preserved(original: str, normalized: str) -> bool:
    original_identifiers = _protected_identifiers(original)
    normalized_identifiers = _protected_identifiers(normalized)
    if original_identifiers != normalized_identifiers:
        return False
    if _numbers(original) != _numbers(normalized):
        return False
    if (_NEGATION.search(original) is not None) != (
        _NEGATION.search(normalized) is not None
    ):
        return False
    return True


def _introduces_protected_terms(original: str, candidate: str) -> bool:
    return (
        not _protected_identifiers(candidate).issubset(
            _protected_identifiers(original)
        )
        or not _numbers(candidate).issubset(_numbers(original))
        or (_NEGATION.search(candidate) is not None)
        != (_NEGATION.search(original) is not None)
    )


class UnsafeNormalizationError(ValueError):
    pass


def _fallback_reason(exc: Exception) -> str:
    if isinstance(exc, UnsafeNormalizationError):
        return "protected_terms"
    if isinstance(exc, ServiceError) and exc.code == "INPUT_TOO_LONG":
        return "budget"
    if isinstance(exc, NormalizationResponseError):
        return "invalid_response"
    if isinstance(exc, NormalizationRequestError):
        return "request_error"
    if isinstance(exc, TimeoutError):
        return "deadline"
    return "gateway_error"


class QueryNormalizer:
    def __init__(self, gateway: _Gateway) -> None:
        self._gateway = gateway

    async def prepare(
        self,
        question: str,
        category: str | None,
        *,
        deadline: float,
    ) -> QueryPlan:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("query normalization deadline exceeded")
            async with asyncio.timeout(remaining):
                output = await self._gateway.normalize(question)
            if not protected_terms_preserved(question, output.normalized):
                raise UnsafeNormalizationError(
                    "normalization changed protected query terms"
                )
            if any(
                _introduces_protected_terms(question, synonym)
                for synonym in output.synonyms
            ):
                raise UnsafeNormalizationError(
                    "normalization synonym added protected query terms"
                )
        except Exception as exc:
            logger.info(
                "query normalization fallback",
                extra={"fallback_reason": _fallback_reason(exc)},
            )
            return QueryPlan(
                original=question,
                normalized=question,
                synonyms=(),
                category=category,
                fallback=True,
            )

        synonyms: list[str] = []
        seen = {_canonical(question.strip()), _canonical(output.normalized)}
        for synonym in output.synonyms:
            value = synonym.strip()
            key = _canonical(value)
            if key and key not in seen:
                seen.add(key)
                synonyms.append(value)
        return QueryPlan(
            original=question,
            normalized=output.normalized,
            synonyms=tuple(synonyms),
            category=category,
        )
