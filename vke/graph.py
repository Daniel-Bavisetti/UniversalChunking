"""The event / entity graph.

Five node types and six edge types - enough to express the relationships the
knowledge units actually have, and small enough that every edge is one a query
can use. There is no graph library here: an adjacency dict over a few hundred
nodes is faster than the import, and it serialises straight to JSON for the UI.

The graph earns its place by doing real work in retrieval (`expand` below), not
by being drawn. A query for "authentication" reaches events that never say the
word, via the entities they share.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .schemas import KnowledgeUnit, SceneCut, VideoMeta

# Node types
VIDEO, SCENE, EVENT, ENTITY, SPEAKER = "Video", "Scene", "Event", "Entity", "Speaker"

# Edge types
CONTAINS = "CONTAINS"              # Video -> Scene, Video -> Event
PRECEDES = "PRECEDES"              # Event -> Event (temporal order)
OCCURS_DURING = "OCCURS_DURING"    # Event -> Scene
MENTIONS = "MENTIONS"              # Event -> Entity
SPOKEN_BY = "SPOKEN_BY"            # Event -> Speaker
RELATED_TO = "RELATED_TO"          # Event -> Event (shared entities)

MIN_SHARED_ENTITIES = 2


@dataclass
class Node:
    id: str
    type: str
    label: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    type: str
    weight: float = 1.0


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _adj: dict[str, list[tuple[str, str, float]]] = field(
        default_factory=lambda: defaultdict(list))

    # --- construction ------------------------------------------------------ #
    def add_node(self, node: Node) -> None:
        self.nodes.setdefault(node.id, node)

    def add_edge(self, source: str, target: str, type_: str, weight: float = 1.0) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        self.edges.append(Edge(source, target, type_, weight))
        # Undirected adjacency: retrieval walks relationships in both directions.
        self._adj[source].append((target, type_, weight))
        self._adj[target].append((source, type_, weight))

    # --- queries ----------------------------------------------------------- #
    def neighbours(self, node_id: str, types: Iterable[str] | None = None
                   ) -> list[tuple[str, str, float]]:
        out = self._adj.get(node_id, [])
        if types is None:
            return out
        allowed = set(types)
        return [e for e in out if e[1] in allowed]

    def expand(self, seeds: Iterable[str], hops: int = 1,
               node_types: Iterable[str] | None = None) -> dict[str, float]:
        """Breadth-first expansion with distance decay.

        Returns {node_id: score}. This is what makes the graph useful rather than
        decorative: seed it with the entities a query matched and it returns the
        events that involve them, including ones whose transcript never contains
        the query terms.
        """
        wanted = set(node_types) if node_types else None
        scores: dict[str, float] = {}
        frontier: list[tuple[str, int]] = [(s, 0) for s in seeds if s in self.nodes]
        visited: set[str] = {s for s, _ in frontier}

        while frontier:
            node_id, depth = frontier.pop(0)
            if depth > 0 and (wanted is None or self.nodes[node_id].type in wanted):
                scores[node_id] = max(scores.get(node_id, 0.0), 1.0 / (1 + depth))
            if depth >= hops:
                continue
            for neighbour, _type, weight in self.neighbours(node_id):
                if neighbour not in visited:
                    visited.add(neighbour)
                    frontier.append((neighbour, depth + 1))
        return scores

    def path_between(self, a: str, b: str, max_hops: int = 4) -> list[str] | None:
        """Shortest path, used to explain *why* two events are related."""
        if a not in self.nodes or b not in self.nodes:
            return None
        queue: list[list[str]] = [[a]]
        seen = {a}
        while queue:
            path = queue.pop(0)
            if path[-1] == b:
                return path
            if len(path) > max_hops:
                continue
            for neighbour, _t, _w in self.neighbours(path[-1]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append([*path, neighbour])
        return None

    # --- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            counts[node.type] += 1
        edge_counts: dict[str, int] = defaultdict(int)
        for edge in self.edges:
            edge_counts[edge.type] += 1
        return {
            "nodes": [
                {"id": n.id, "type": n.type, "label": n.label, **n.data}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "type": e.type,
                 "weight": round(e.weight, 3)}
                for e in self.edges
            ],
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "nodes_by_type": dict(counts),
                "edges_by_type": dict(edge_counts),
            },
        }


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def build(
    meta: VideoMeta,
    units: list[KnowledgeUnit],
    cuts: list[SceneCut],
) -> Graph:
    g = Graph()

    video_id = f"video:{meta.video_id}"
    g.add_node(Node(video_id, VIDEO, meta.filename,
                    {"duration": meta.duration, "unit_count": len(units)}))

    # Scenes are a visual segmentation, distinct from events (see docs).
    scene_starts = [0.0, *[c.ts for c in cuts]]
    for i, start in enumerate(scene_starts):
        end = scene_starts[i + 1] if i + 1 < len(scene_starts) else meta.duration
        sid = f"scene:{i}"
        g.add_node(Node(sid, SCENE, f"scene {i}", {"start": start, "end": end}))
        g.add_edge(video_id, sid, CONTAINS)

    for unit in units:
        eid = f"event:{unit.id}"
        g.add_node(Node(eid, EVENT, unit.title, {
            "start": unit.span.start,
            "end": unit.span.end,
            "quality": unit.quality,
            "keyframe": unit.keyframe_url,
        }))
        g.add_edge(video_id, eid, CONTAINS)

        for scene_index in unit.scene_ids:
            g.add_edge(eid, f"scene:{scene_index}", OCCURS_DURING)

        for entity in unit.entities:
            nid = f"entity:{entity}"
            g.add_node(Node(nid, ENTITY, entity))
            g.add_edge(eid, nid, MENTIONS)

        for speaker in unit.speakers:
            nid = f"speaker:{speaker}"
            g.add_node(Node(nid, SPEAKER, speaker))
            g.add_edge(eid, nid, SPOKEN_BY)

    # Temporal chain: what came before and after.
    for prev, nxt in zip(units, units[1:]):
        g.add_edge(f"event:{prev.id}", f"event:{nxt.id}", PRECEDES)

    # Non-adjacent events that share substantive vocabulary. This is the edge
    # that answers "which events relate to authentication?" across the video.
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            shared = set(a.entities) & set(b.entities)
            if len(shared) >= MIN_SHARED_ENTITIES and b.id != a.next_unit_id:
                g.add_edge(f"event:{a.id}", f"event:{b.id}", RELATED_TO,
                           weight=len(shared))
    return g
