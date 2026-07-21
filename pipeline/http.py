"""Shared HTTP retry configuration.

Provides a single :func:`http_retry` decorator used by every module that makes
external HTTP calls, so retry behavior is consistent across the pipeline:
3 attempts, exponential backoff with jitter.
"""

from __future__ import annotations

import logging

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# Number of attempts (including the first) for external HTTP calls.
HTTP_MAX_ATTEMPTS = 3


def http_retry(func):
    """Decorate an HTTP-calling function with standard retry behavior.

    Retries up to :data:`HTTP_MAX_ATTEMPTS` times on connection errors, timeouts,
    and HTTP errors, using exponential backoff with jitter.

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function.
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(HTTP_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type(
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            )
        ),
        before_sleep=lambda state: logger.warning(
            "Retrying %s (attempt %d) after error: %s",
            func.__name__,
            state.attempt_number,
            state.outcome.exception() if state.outcome else "unknown",
        ),
    )(func)
