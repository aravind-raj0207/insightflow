"""A cheap first line of defence: refuse questions that ask to change or destroy data.

This is NOT the only protection. The SQL validator and the read-only database role
also block writes. This check just gives the user a clear message early.
"""
import re

_MODIFY_VERBS = r"(delete|drop|truncate|alter|update|insert|erase|wipe|destroy|overwrite)"
_TARGETS = r"(table|tables|rows?|records?|data|database|dataset|columns?|schema|everything)"

_PATTERNS = [
    re.compile(rf"\b{_MODIFY_VERBS}\b.{{0,60}}\b{_TARGETS}\b", re.IGNORECASE),
    re.compile(rf"^\s*(please\s+)?{_MODIFY_VERBS}\b", re.IGNORECASE),
    re.compile(r";\s*(delete|drop|update|insert|truncate|alter|create)\b", re.IGNORECASE),
]


def looks_like_modification_request(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PATTERNS)
