#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace treemendous {
using Coordinate = std::int64_t;
using Measure = std::int64_t;

inline void validate_span(Coordinate start, Coordinate end) {
    if (start >= end) {
        throw std::invalid_argument("span must satisfy start < end");
    }
}

inline void validate_length(Measure length) {
    if (length <= 0) {
        throw std::invalid_argument("length must be greater than zero");
    }
}

inline Measure checked_length(Coordinate start, Coordinate end) {
    validate_span(start, end);
    if (start < 0 && end > std::numeric_limits<Coordinate>::max() + start) {
        throw std::overflow_error("interval length exceeds signed 64-bit measure");
    }
    return end - start;
}

inline Coordinate checked_end(Coordinate start, Measure length) {
    validate_length(length);
    if (start > std::numeric_limits<Coordinate>::max() - length) {
        throw std::overflow_error("start + length exceeds signed 64-bit coordinate");
    }
    return start + length;
}

inline Measure checked_add(Measure left, Measure right) {
    if ((right > 0 && left > std::numeric_limits<Measure>::max() - right) ||
        (right < 0 && left < std::numeric_limits<Measure>::min() - right)) {
        throw std::overflow_error("interval measure overflow");
    }
    return left + right;
}
}  // namespace treemendous
