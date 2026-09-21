"""Only code chooses graph destinations; model output never names a node."""
import math

from app.workflow.contracts import IntentResult

_ROUTES = {
    'logistics': 'business', 'order': 'business', 'after_sales': 'business',
    'product': 'knowledge', 'return_refund': 'knowledge',
    'complaint': 'complaint', 'chitchat': 'chitchat',
}


def route_intent(intent: IntentResult) -> str:
    return _ROUTES[intent.intent]


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def knowledge_band(score: float, *, lower: float = .7, upper: float = .8) -> str:
    if not _finite_number(score) or not 0 <= score <= 1:
        raise ValueError('invalid reranker score')
    if not _finite_number(lower) or not _finite_number(upper) or not 0 <= lower < upper <= 1:
        raise ValueError('invalid knowledge thresholds')
    return 'high' if score > upper else 'middle' if score >= lower else 'low'
