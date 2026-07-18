// Portable optimized facade. The previous copy duplicated the core algorithm and
// drifted semantically; keep one checked implementation until a measured storage
// specialization earns a separate implementation.
#define TREE_MENDOUS_USE_FLAT_MAP 0
#define TREE_MENDOUS_USE_SMALL_VECTOR 0
#define TREE_MENDOUS_USE_SIMD 0
#define TREE_MENDOUS_PREALLOCATE_VECTORS 0

#include "boundary.cpp"

class IntervalManagerOptimized : public IntervalManager {
public:
    using IntervalManager::IntervalManager;
};
