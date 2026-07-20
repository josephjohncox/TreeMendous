# Distributed genetic search

`GeneticSearchEngine` models actual genetic evolution over equal-width bit strings. Each generation scores the population with an injected finite-valued fitness function, sorts by descending fitness and lexical candidate tie-break, selects the top half (at least two), performs seeded parent selection and single-point crossover, and flips each bit according to the configured mutation probability.

Generation ordinals are claimed from the private ledger. `step()` executes one claimed generation; `run()` completes the configured count. `best()` rescoring is deterministic. The checkpoint contains generation number, population, history, Python PRNG state, and claim/event state, making an in-process lineage replayable when paired with the same fitness function. Invalid population shapes, bit alphabets, rates, seeds, generation counts, and non-finite fitness outputs are rejected.

Fitness callables are local code and are not serialized. Neither population storage nor checkpoint persistence is distributed. A cluster integration must version the fitness implementation, durably persist the PRNG/checkpoint and population, assign generation ownership, and fence stale generation writes. It must not run multiple dependent generations as though they were independent candidate bands.

The example is seeded and deterministic. The smoke performs eight real selection/crossover/mutation generations and checks the initial ranking against a structurally independent oracle.
