# Map-reduce input splits

`MapReduceEngine` creates real input splits in two modes. `records` uses `bytes.splitlines(keepends=True)` and groups a configured number of complete records while retaining exact byte spans. `bytes` creates fixed-size raw chunks. Workers claim split IDs and invoke the injected mapper on each split unit. Emissions must have nonempty string keys. A failed map band is discarded and its claim abandoned.

Mapped emissions are retained by split ID. `reduce()` groups keys and invokes the injected reducer in stable split/emission order; keys are output lexically. The reducer is contractually associative because arbitrary Python behavior cannot be proved associative. `run()` and `snapshot()` expose execution; `audit_snapshot()` adds local coordination evidence but is not restorable because it omits the input and injected code. Empty bytes, invalid modes/sizes, malformed emissions, and callback failures are explicit.

This engine does not distribute input or shuffle data. Mapped values remain process-local. A deployment needs immutable split identity, durable/fenced map output, serialization, shuffle transport, reducer placement, retry policy, and a genuinely associative reducer.

The example performs record-split word count. The smoke maps 500 records and compares final counts with an independent whole-input oracle.
