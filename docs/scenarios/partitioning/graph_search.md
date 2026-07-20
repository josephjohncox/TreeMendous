# Distributed graph frontier search

`GraphSearchEngine` performs deterministic breadth-first search. Construction copies adjacency lists, removes duplicate edges, sorts neighbors lexically, requires every referenced vertex to exist, and verifies the start vertex. The frontier is FIFO; a discovered vertex is queued once, expanded once, and assigned the shortest BFS distance. Expansion ordinals are claimed independently from application vertex names.

`expand(width=...)` processes a bounded frontier band and `run()` continues until no reachable vertices remain. `snapshot()` reports expansion order, visited distances, and the pending frontier; `checkpoint()` adds local claim/event evidence. Dangling edges, invalid names, empty graphs, invalid starts, and non-sequence adjacency fail before execution. Disconnected vertices remain unvisited by correct BFS semantics.

The queue, visited set, graph, and claims are process-local. A genuine distributed frontier needs durable atomic discovered/visited transitions, stable level ordering (or documented weaker ordering), work leases, and fencing-token enforcement. This engine does not provide consensus, remote worker liveness, or storage.

The example demonstrates lexical tie-breaking. The smoke expands a 300-vertex graph and compares complete order and distances with an independent list-based BFS oracle.
