"""Event-driven messaging abstractions for Cleave v2.

Enables asynchronous, distributed job dispatch via Redpanda/Kafka
with an in-memory queue fallback.
"""

from .producer import EventProducer, get_event_producer

__all__ = ["EventProducer", "get_event_producer"]
