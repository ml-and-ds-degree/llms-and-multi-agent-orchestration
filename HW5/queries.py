"""
Test queries for BOI.pdf (Bank of Israel / Beneficial Ownership Information)
These queries are designed to test different aspects of RAG retrieval and
provide lightweight heuristics for automatic accuracy checks.
"""

from typing import Optional

# Test queries categorized by difficulty/type
TEST_QUERIES = [
    # Direct factual questions
    {
        "query": "How to report BOI?",
        "type": "direct",
        "expected_context": "Should mention reporting process/procedures",
        "keywords": ["boi", "report", "file", "portal"],
        "min_keywords": 2,
    },
    {
        "query": "What is the document about?",
        "type": "general",
        "expected_context": "Should provide high-level overview",
        "keywords": ["manual", "beneficial ownership", "report", "requirements"],
        "min_keywords": 2,
    },
    {
        "query": "What are the main points as a business owner I should be aware of?",
        "type": "synthesis",
        "expected_context": "Should summarize key requirements for business owners",
        "keywords": [
            "beneficial owners",
            "reporting company",
            "company applicants",
            "requirements",
        ],
        "min_keywords": 2,
    },
]

# Additional queries for more comprehensive testing
EXTENDED_QUERIES = [
    {
        "query": "Who needs to file BOI reports?",
        "type": "direct",
        "expected_context": "Should mention reporting companies and exemptions",
        "keywords": ["reporting companies", "exempt", "must report"],
        "min_keywords": 2,
    },
    {
        "query": "What information must be reported about beneficial owners?",
        "type": "detailed",
        "expected_context": "Should list required information fields",
        "keywords": ["name", "address", "identification", "dob"],
        "min_keywords": 2,
    },
    {
        "query": "What are the deadlines for BOI reporting?",
        "type": "specific",
        "expected_context": "Should mention filing deadlines and timeframes",
        "keywords": ["deadline", "timeframe", "days", "calendar"],
        "min_keywords": 1,
    },
    {
        "query": "What are the penalties for non-compliance?",
        "type": "specific",
        "expected_context": "Should describe civil and criminal penalties",
        "keywords": ["penalties", "fines", "civil", "criminal"],
        "min_keywords": 2,
    },
]


def get_baseline_queries():
    """Get the baseline set of queries used in the video."""
    return [q["query"] for q in TEST_QUERIES]


def get_all_queries():
    """Get all available test queries."""
    return [q["query"] for q in TEST_QUERIES + EXTENDED_QUERIES]


def get_query_info(query: str):
    """Get metadata about a specific query."""
    all_queries = TEST_QUERIES + EXTENDED_QUERIES
    for q in all_queries:
        if q["query"] == query:
            return q
    return None


def evaluate_query_accuracy(query: str, response: str) -> Optional[bool]:
    """Heuristically evaluate whether a response covers the expected context."""
    metadata = get_query_info(query)
    if not metadata:
        return None

    keywords = metadata.get("keywords")
    if not keywords:
        return None

    response_text = response.lower()
    matches = sum(1 for keyword in keywords if keyword.lower() in response_text)
    min_required = metadata.get("min_keywords", len(keywords))
    return matches >= min_required
