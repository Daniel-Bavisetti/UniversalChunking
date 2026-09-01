"""Tests for entity and topic enrichment with Gemini and fallback."""

import json
from unittest.mock import MagicMock, patch

from cleave.enrich_entities import enrich_entities_batch
from cleave.models import (
    ChunkingDecision,
    Context,
    KnowledgeUnit,
    Modality,
    Provenance,
)


def test_enrich_entities_batch_with_mock_llm():
    unit = KnowledgeUnit(
        id="ku_0001",
        content="Google DeepMind developed AlphaFold, revolutionizing structural biology and protein prediction.",
        modality=Modality.DOCUMENT,
        context=Context(document_title="DeepMind Biology"),
        provenance=Provenance(source_uri="paper.pdf"),
        decision=ChunkingDecision(strategy="structural", reason="section boundary"),
        token_count=45,
    )

    mock_llm_response = {
        "entities": ["Google DeepMind", "AlphaFold", "structural biology"],
        "topics": ["Artificial Intelligence", "Protein Folding"],
        "summary": "DeepMind developed AlphaFold for protein prediction.",
    }

    mock_provider = MagicMock()
    mock_provider.is_configured.return_value = True
    mock_provider.name = "gemini"
    mock_provider.complete_json.return_value = (json.dumps(mock_llm_response), {})

    with patch("cleave.enrich_entities.get_provider", return_value=mock_provider):
        enriched = enrich_entities_batch([unit], max_enrich=5)
        assert len(enriched) == 1
        assert "AlphaFold" in enriched[0].entities
        assert "Google DeepMind" in enriched[0].entities
        assert "topics" in enriched[0].metadata
        assert "Protein Folding" in enriched[0].metadata["topics"]
