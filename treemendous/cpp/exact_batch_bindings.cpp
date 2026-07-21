// Stable whole-batch-atomic exact geometry mutation engine.
#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
using Coordinate = std::int64_t;
using Interval = std::pair<Coordinate, Coordinate>;

static_assert(sizeof(Coordinate) == 8);
static_assert(std::is_trivially_copyable_v<Coordinate>);
constexpr std::size_t kOperationFields = 3;
constexpr std::size_t kOperationBytes = kOperationFields * sizeof(Coordinate);
constexpr std::size_t kSignalPeriod = 1024;

class BatchLimitExceeded : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

std::size_t checked_size_add(std::size_t left, std::size_t right,
                             const char* description) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
        throw std::overflow_error(description);
    return left + right;
}

std::size_t checked_size_multiply(std::size_t left, std::size_t right,
                                  const char* description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        throw std::overflow_error(description);
    return left * right;
}

py::ssize_t checked_shape(std::size_t value, const char* description) {
    if (value > static_cast<std::size_t>(PY_SSIZE_T_MAX))
        throw std::overflow_error(description);
    return static_cast<py::ssize_t>(value);
}

void validate_limit(std::size_t value, const char* name) {
    if (value == 0) throw py::value_error(std::string(name) + " must be positive");
    if (value > static_cast<std::size_t>(PY_SSIZE_T_MAX))
        throw std::overflow_error(std::string(name) +
                                  " exceeds signed size range");
}

Coordinate decode_operation_field(const std::vector<std::uint8_t>& bytes,
                                  std::size_t row, std::size_t field) {
    Coordinate value;
    const std::size_t row_offset =
        checked_size_multiply(row, kOperationBytes, "operation offset overflow");
    const std::size_t field_offset = checked_size_multiply(
        field, sizeof(Coordinate), "operation field offset overflow");
    const std::size_t offset = checked_size_add(
        row_offset, field_offset, "operation field offset overflow");
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

class PackedBuffer {
  public:
    PackedBuffer(const py::bytes& storage, const std::string& format,
                 const std::vector<py::ssize_t>& shape,
                 const std::vector<py::ssize_t>& strides)
        : storage_(storage), format_(format), shape_(shape), strides_(strides) {}

    py::buffer_info buffer_info() {
        return py::buffer_info(
            PyBytes_AS_STRING(storage_.ptr()), format_ == "B" ? 1 : 8, format_,
            checked_shape(shape_.size(), "buffer dimension count overflow"), shape_,
            strides_, true);
    }

  private:
    py::bytes storage_;
    std::string format_;
    std::vector<py::ssize_t> shape_;
    std::vector<py::ssize_t> strides_;
};

class PackedMutationResults {
  public:
    PackedMutationResults(const py::bytes& offsets, const py::bytes& spans,
                          const py::bytes& lengths, const py::bytes& covered,
                          std::size_t rows, std::size_t changed_count)
        : offsets_(offsets), spans_(spans), lengths_(lengths), covered_(covered),
          rows_(rows), changed_count_(changed_count) {}

    std::size_t size() const noexcept { return rows_; }

    py::object offsets() const {
        const auto count = checked_size_add(rows_, 1, "offset shape overflow");
        return make_view(offsets_, "Q", {checked_shape(count, "offset shape overflow")},
                         {8});
    }
    py::object lengths() const {
        return make_view(lengths_, "q", {checked_shape(rows_, "length shape overflow")},
                         {8});
    }
    py::object covered() const {
        return make_view(covered_, "B", {checked_shape(rows_, "coverage shape overflow")},
                         {1});
    }
    py::object spans() const {
        return make_view(spans_, "q",
                         {checked_shape(changed_count_, "span shape overflow"), 2},
                         {16, 8});
    }

    py::tuple materialize() const {
        py::object domain = py::module_::import("treemendous.domain");
        py::object span_type = domain.attr("Span");
        py::object result_type = domain.attr("MutationResult");
        py::tuple output(checked_shape(rows_, "result tuple shape overflow"));
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::uint64_t begin = binary_at<std::uint64_t>(offsets_, row);
            const std::size_t next_row =
                checked_size_add(row, 1, "offset index overflow");
            const std::uint64_t end =
                binary_at<std::uint64_t>(offsets_, next_row);
            if (end < begin || end > changed_count_)
                throw std::runtime_error("corrupt packed result offsets");
            const auto changed_size = static_cast<std::size_t>(end - begin);
            py::tuple changed(
                checked_shape(changed_size, "changed tuple shape overflow"));
            for (std::uint64_t index = begin; index < end; ++index) {
                const auto span_index = checked_size_multiply(
                    static_cast<std::size_t>(index), 2, "span index overflow");
                const std::size_t span_end_index =
                    checked_size_add(span_index, 1, "span index overflow");
                changed[static_cast<std::size_t>(index - begin)] = span_type(
                    binary_at<Coordinate>(spans_, span_index),
                    binary_at<Coordinate>(spans_, span_end_index));
            }
            output[row] = result_type(
                changed, binary_at<Coordinate>(lengths_, row),
                binary_at<std::uint8_t>(covered_, row) != 0);
        }
        return output;
    }

  private:
    template <typename T>
    static T binary_at(const py::bytes& storage, std::size_t index) {
        const std::size_t offset =
            checked_size_multiply(index, sizeof(T), "packed result index overflow");
        T value;
        std::memcpy(&value, PyBytes_AS_STRING(storage.ptr()) + offset, sizeof(T));
        return value;
    }

    static py::object make_view(const py::bytes& storage, const char* format,
                                const std::vector<py::ssize_t>& shape,
                                const std::vector<py::ssize_t>& strides) {
        auto owner = std::make_shared<PackedBuffer>(storage, format, shape, strides);
        py::object prepared_owner = py::cast(owner);
        return py::module_::import("builtins").attr("memoryview")(prepared_owner);
    }

    py::bytes offsets_;
    py::bytes spans_;
    py::bytes lengths_;
    py::bytes covered_;
    std::size_t rows_;
    std::size_t changed_count_;
};

class MutationGuard {
  public:
    explicit MutationGuard(std::atomic<bool>& active) : active_(active) {
        if (active_.exchange(true, std::memory_order_acq_rel)) {
            throw std::runtime_error(
                "overlapping or reentrant mutation on the same ExactBatchRangeSet");
        }
    }
    ~MutationGuard() { active_.store(false, std::memory_order_release); }
    MutationGuard(const MutationGuard&) = delete;
    MutationGuard& operator=(const MutationGuard&) = delete;

  private:
    std::atomic<bool>& active_;
};

[[noreturn]] void operation_error(std::size_t index, const std::string& message) {
    throw std::invalid_argument("operation " + std::to_string(index) + ": " +
                                message);
}

[[noreturn]] void operation_overflow(std::size_t index,
                                     const std::string& message) {
    throw std::overflow_error("operation " + std::to_string(index) + ": " +
                              message);
}

[[noreturn]] void limit_error(const char* limit) {
    throw BatchLimitExceeded(std::string(limit) + " limit exceeded");
}

Coordinate checked_length(Coordinate start, Coordinate end, std::size_t index) {
    if (start >= end) operation_error(index, "span must satisfy start < end");
    Coordinate result;
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_sub_overflow(end, start, &result) || result < 0)
        operation_overflow(index, "signed span measure overflow");
#else
    if (start < 0 && end > std::numeric_limits<Coordinate>::max() + start)
        operation_overflow(index, "signed span measure overflow");
    result = end - start;
#endif
    return result;
}

void checked_coordinate_add(Coordinate& value, Coordinate delta,
                            std::size_t index, const char* description) {
    Coordinate result;
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_add_overflow(value, delta, &result))
        operation_overflow(index, description);
#else
    if (delta > 0 && value > std::numeric_limits<Coordinate>::max() - delta)
        operation_overflow(index, description);
    result = value + delta;
#endif
    value = result;
}

void checked_coordinate_subtract(Coordinate& value, Coordinate delta,
                                 std::size_t index, const char* description) {
    Coordinate result;
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_sub_overflow(value, delta, &result))
        operation_overflow(index, description);
#else
    if (delta < 0 && value > std::numeric_limits<Coordinate>::max() + delta)
        operation_overflow(index, description);
    if (delta > 0 && value < std::numeric_limits<Coordinate>::min() + delta)
        operation_overflow(index, description);
    result = value - delta;
#endif
    value = result;
}

template <typename T>
void append_binary(std::string& output, const T& value) {
    const std::size_t old_size = output.size();
    const std::size_t next = checked_size_add(
        old_size, sizeof(T), "packed result byte count overflow");
    output.resize(next);
    std::memcpy(output.data() + old_size, &value, sizeof(T));
}

bool fully_covered(const std::vector<Interval>& state, Coordinate start,
                   Coordinate end) {
    auto item = std::upper_bound(
        state.begin(), state.end(), start,
        [](Coordinate point, const Interval& span) { return point < span.first; });
    if (item != state.begin()) --item;
    return item != state.end() && item->first <= start && end <= item->second;
}

std::vector<Interval> add_span(std::vector<Interval>& state, Coordinate start,
                               Coordinate end) {
    std::vector<Interval> changed;
    Coordinate cursor = start;
    for (const auto& interval : state) {
        if (interval.second <= cursor) continue;
        if (interval.first >= end) break;
        if (interval.first > cursor)
            changed.emplace_back(cursor, std::min(interval.first, end));
        cursor = std::max(cursor, std::min(interval.second, end));
        if (cursor >= end) break;
    }
    if (cursor < end) changed.emplace_back(cursor, end);

    std::vector<Interval> next;
    next.reserve(checked_size_add(state.size(), 1, "live interval count overflow"));
    Coordinate merged_start = start;
    Coordinate merged_end = end;
    bool inserted = false;
    for (const auto& interval : state) {
        if (interval.second < merged_start) {
            next.push_back(interval);
        } else if (merged_end < interval.first) {
            if (!inserted) {
                next.emplace_back(merged_start, merged_end);
                inserted = true;
            }
            next.push_back(interval);
        } else {
            merged_start = std::min(merged_start, interval.first);
            merged_end = std::max(merged_end, interval.second);
        }
    }
    if (!inserted) next.emplace_back(merged_start, merged_end);
    state.swap(next);
    return changed;
}

std::vector<Interval> discard_span(std::vector<Interval>& state, Coordinate start,
                                   Coordinate end) {
    std::vector<Interval> changed;
    std::vector<Interval> next;
    next.reserve(checked_size_add(state.size(), 1, "live interval count overflow"));
    for (const auto& interval : state) {
        if (interval.second <= start || interval.first >= end) {
            next.push_back(interval);
            continue;
        }
        changed.emplace_back(std::max(interval.first, start),
                             std::min(interval.second, end));
        if (interval.first < start) next.emplace_back(interval.first, start);
        if (interval.second > end) next.emplace_back(end, interval.second);
    }
    state.swap(next);
    return changed;
}

struct Limits {
    std::size_t operations;
    std::size_t live_intervals;
    std::size_t changed_spans;
    std::size_t result_bytes;
    std::size_t work_units;
};

class ExactBatchManager {
  public:
    ExactBatchManager(const std::vector<Interval>& domain, bool initially_available,
                      std::size_t max_operations,
                      std::size_t max_live_intervals,
                      std::size_t max_changed_spans,
                      std::size_t max_result_bytes,
                      std::size_t max_work_units)
        : domain_(domain),
          state_(initially_available ? domain_ : std::vector<Interval>{}),
          limits_{max_operations, max_live_intervals, max_changed_spans,
                  max_result_bytes, max_work_units} {
        validate_limit(max_operations, "max_operations");
        validate_limit(max_live_intervals, "max_live_intervals");
        validate_limit(max_changed_spans, "max_changed_spans");
        validate_limit(max_result_bytes, "max_result_bytes");
        validate_limit(max_work_units, "max_work_units");
        if (state_.size() > limits_.live_intervals)
            limit_error("max_live_intervals");

        Coordinate total = 0;
        Coordinate previous_end = 0;
        bool first = true;
        for (const auto& span : domain_) {
            if (span.first >= span.second)
                throw py::value_error("domain span must satisfy start < end");
            if (!first && span.first <= previous_end)
                throw py::value_error("domain spans must be normalized and disjoint");
            const auto length = constructor_length(span.first, span.second);
            if (length > std::numeric_limits<Coordinate>::max() - total)
                throw std::overflow_error("signed managed-domain measure overflow");
            total += length;
            previous_end = span.second;
            first = false;
        }
        if (domain_.empty()) throw py::value_error("managed domain must not be empty");
        total_ = initially_available ? total : 0;
    }

    py::object mutate_packed(const py::handle& operations) {
        return mutate_impl(operations, false);
    }

    py::tuple mutate_materialized(const py::handle& operations) {
        return mutate_impl(operations, true).cast<py::tuple>();
    }

    py::object mutate_impl(const py::handle& operations, bool materialize) {
        if (!PyBytes_CheckExact(operations.ptr()))
            throw py::type_error("operations must be exact immutable bytes");
        MutationGuard mutation_guard(mutation_active_);
        const Py_ssize_t raw_length = PyBytes_GET_SIZE(operations.ptr());
        if (raw_length < 0) throw std::overflow_error("operation byte length overflow");
        const auto byte_length = static_cast<std::size_t>(raw_length);
        if (byte_length % kOperationBytes != 0)
            throw py::value_error("operation byte length must be a multiple of 24");
        const std::size_t operation_count = byte_length / kOperationBytes;
        if (operation_count > limits_.operations) limit_error("max_operations");
        const std::size_t expected = checked_size_multiply(
            operation_count, kOperationBytes, "operation byte length overflow");
        if (expected != byte_length)
            throw py::value_error("operation byte length is inconsistent");

        maybe_fail("operations_copy");
        std::vector<std::uint8_t> operation_bytes(expected);
        if (expected != 0)
            std::memcpy(operation_bytes.data(), PyBytes_AS_STRING(operations.ptr()),
                        expected);

        std::vector<Interval> scratch;
        Coordinate scratch_total;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            maybe_fail("state_copy");
            scratch = state_;
            scratch_total = total_;
        }

        std::vector<std::uint64_t> offsets;
        std::vector<Interval> changed_spans;
        std::vector<Coordinate> changed_lengths;
        std::vector<std::uint8_t> covered_values;
        maybe_fail("result_reserve");
        const std::size_t offset_count =
            checked_size_add(operation_count, 1, "offset count overflow");
        offsets.reserve(offset_count);
        changed_lengths.reserve(operation_count);
        covered_values.reserve(operation_count);
        offsets.push_back(0);

        const std::size_t offsets_bytes = checked_size_multiply(
            offset_count, sizeof(std::uint64_t), "offset byte count overflow");
        const std::size_t lengths_bytes = checked_size_multiply(
            operation_count, sizeof(Coordinate), "length byte count overflow");
        const std::size_t base_result_bytes = checked_size_add(
            checked_size_add(offsets_bytes, lengths_bytes,
                             "packed result byte count overflow"),
            operation_count, "packed result byte count overflow");
        if (base_result_bytes > limits_.result_bytes)
            limit_error("max_result_bytes");

        std::size_t work_units = 0;
        {
            py::gil_scoped_release release;
            for (std::size_t index = 0; index < operation_count; ++index) {
                if (index % kSignalPeriod == 0) {
                    py::gil_scoped_acquire acquire;
                    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
                }
                maybe_fail("row_staging");
                const std::size_t row_work = checked_size_add(
                    scratch.size(), 1, "work unit count overflow");
                work_units = checked_size_add(work_units, row_work,
                                              "work unit count overflow");
                if (work_units > limits_.work_units) limit_error("max_work_units");

                const Coordinate opcode =
                    decode_operation_field(operation_bytes, index, 0);
                const Coordinate start =
                    decode_operation_field(operation_bytes, index, 1);
                const Coordinate end =
                    decode_operation_field(operation_bytes, index, 2);
                if (opcode < 0 || opcode > 2)
                    operation_error(index, "unknown mutation opcode");
                (void)checked_length(start, end, index);
                if (!inside_component(start, end))
                    operation_error(
                        index,
                        "span is outside or crosses a managed-domain component");
                const bool covered = fully_covered(scratch, start, end);
                std::vector<Interval> changed;
                if (opcode == 0) {
                    changed = add_span(scratch, start, end);
                } else if (opcode == 2 && !covered) {
                    // Strict rejection is an exact no-op.
                } else {
                    changed = discard_span(scratch, start, end);
                }
                if (scratch.size() > limits_.live_intervals)
                    limit_error("max_live_intervals");

                Coordinate changed_length = 0;
                for (const auto& span : changed) {
                    const Coordinate part =
                        checked_length(span.first, span.second, index);
                    checked_coordinate_add(changed_length, part, index,
                                           "signed changed-length overflow");
                }
                if (opcode == 0)
                    checked_coordinate_add(scratch_total, changed_length, index,
                                           "signed total overflow");
                else
                    checked_coordinate_subtract(scratch_total, changed_length, index,
                                                "signed total overflow");
                if (scratch_total < 0)
                    operation_overflow(index, "signed total overflow");

                const std::size_t next_changed = checked_size_add(
                    changed_spans.size(), changed.size(),
                    "packed changed-span offset overflow");
                if (next_changed > limits_.changed_spans)
                    limit_error("max_changed_spans");
                const std::size_t span_bytes = checked_size_multiply(
                    next_changed, 2 * sizeof(Coordinate),
                    "span byte count overflow");
                const std::size_t exact_result_bytes = checked_size_add(
                    base_result_bytes, span_bytes,
                    "packed result byte count overflow");
                if (exact_result_bytes > limits_.result_bytes)
                    limit_error("max_result_bytes");
                if (next_changed > std::numeric_limits<std::uint64_t>::max())
                    operation_overflow(index,
                                       "packed changed-span offset overflow");
                changed_spans.insert(changed_spans.end(), changed.begin(),
                                     changed.end());
                offsets.push_back(static_cast<std::uint64_t>(next_changed));
                changed_lengths.push_back(changed_length);
                covered_values.push_back(covered ? 1 : 0);
            }
        }

        const std::size_t span_field_count = checked_size_multiply(
            changed_spans.size(), 2, "span field count overflow");
        const std::size_t spans_bytes = checked_size_multiply(
            span_field_count, sizeof(Coordinate), "span byte count overflow");
        const std::size_t exact_result_bytes = checked_size_add(
            base_result_bytes, spans_bytes, "packed result byte count overflow");
        if (exact_result_bytes > limits_.result_bytes)
            limit_error("max_result_bytes");

        maybe_fail("packed_storage");
        std::string offsets_storage;
        std::string spans_storage;
        std::string lengths_storage;
        std::string covered_storage;
        offsets_storage.reserve(offsets_bytes);
        spans_storage.reserve(spans_bytes);
        lengths_storage.reserve(lengths_bytes);
        covered_storage.reserve(operation_count);
        for (const auto value : offsets) append_binary(offsets_storage, value);
        for (const auto& span : changed_spans) {
            append_binary(spans_storage, span.first);
            append_binary(spans_storage, span.second);
        }
        for (const auto value : changed_lengths)
            append_binary(lengths_storage, value);
        for (const auto value : covered_values)
            append_binary(covered_storage, value);
        if (checked_size_add(
                checked_size_add(offsets_storage.size(), spans_storage.size(),
                                 "packed result byte count overflow"),
                checked_size_add(lengths_storage.size(), covered_storage.size(),
                                 "packed result byte count overflow"),
                "packed result byte count overflow") != exact_result_bytes)
            throw std::runtime_error("packed result byte count mismatch");

        maybe_fail("python_bytes");
        py::bytes packed_offsets(offsets_storage);
        py::bytes packed_spans(spans_storage);
        py::bytes packed_lengths(lengths_storage);
        py::bytes packed_covered(covered_storage);
        auto packed = std::make_shared<PackedMutationResults>(
            packed_offsets, packed_spans, packed_lengths, packed_covered,
            operation_count, changed_spans.size());
        py::object prepared;
        if (materialize) {
            maybe_fail("materialized_results");
            prepared = packed->materialize();
        } else {
            maybe_fail("wrapper_preparation");
            prepared = py::cast(packed);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.swap(scratch);  // allocation-free and noexcept
            total_ = scratch_total;
        }
        return prepared;
    }

    py::tuple snapshot_data() const {
        std::vector<Interval> copy;
        Coordinate total;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(state_mutex_);
            copy = state_;
            total = total_;
        }
        py::tuple intervals(
            checked_shape(copy.size(), "snapshot tuple shape overflow"));
        for (std::size_t index = 0; index < copy.size(); ++index)
            intervals[index] =
                py::make_tuple(copy[index].first, copy[index].second);
        return py::make_tuple(intervals, total);
    }

    void set_failpoint(const std::string& name, std::size_t occurrence) {
        static const std::vector<std::string> names = {
            "operations_copy", "state_copy",       "result_reserve",
            "row_staging",    "packed_storage",   "python_bytes",
            "wrapper_preparation", "materialized_results"};
        if (std::find(names.begin(), names.end(), name) == names.end())
            throw py::value_error("unknown exact-batch failpoint");
        std::lock_guard<std::mutex> lock(failpoint_mutex_);
        failpoint_name_ = name;
        failpoint_occurrence_ = occurrence;
    }

    void clear_failpoint() {
        std::lock_guard<std::mutex> lock(failpoint_mutex_);
        failpoint_name_.clear();
        failpoint_occurrence_ = 0;
    }

  private:
    static Coordinate constructor_length(Coordinate start, Coordinate end) {
        Coordinate length;
#if defined(__GNUC__) || defined(__clang__)
        if (__builtin_sub_overflow(end, start, &length) || length < 0)
            throw std::overflow_error("signed managed-domain measure overflow");
#else
        if (start < 0 && end > std::numeric_limits<Coordinate>::max() + start)
            throw std::overflow_error("signed managed-domain measure overflow");
        length = end - start;
#endif
        return length;
    }

    bool inside_component(Coordinate start, Coordinate end) const {
        auto item = std::upper_bound(
            domain_.begin(), domain_.end(), start,
            [](Coordinate point, const Interval& span) { return point < span.first; });
        if (item != domain_.begin()) --item;
        return item != domain_.end() && item->first <= start && end <= item->second;
    }

    void maybe_fail(const char* name) {
        std::lock_guard<std::mutex> lock(failpoint_mutex_);
        if (failpoint_name_ != name) return;
        if (failpoint_occurrence_ != 0) {
            --failpoint_occurrence_;
            return;
        }
        failpoint_name_.clear();
        throw std::bad_alloc();
    }

    std::vector<Interval> domain_;
    mutable std::mutex state_mutex_;
    std::vector<Interval> state_;
    Coordinate total_ = 0;
    Limits limits_;
    std::atomic<bool> mutation_active_{false};
    std::mutex failpoint_mutex_;
    std::string failpoint_name_;
    std::size_t failpoint_occurrence_ = 0;
};
}  // namespace

#if PYBIND11_VERSION_HEX >= 0x03000000
PYBIND11_MODULE(_exact_batch, module,
                py::multiple_interpreters::per_interpreter_gil()) {
#else
PYBIND11_MODULE(_exact_batch, module) {
#endif
    module.doc() = "Stable exact whole-batch-atomic sorted-vector geometry engine";
    py::object limit_exception =
        py::register_exception<BatchLimitExceeded>(module, "BatchLimitError",
                                                   PyExc_ValueError);
    limit_exception.attr("__module__") = "treemendous.exact_batch";

    py::class_<PackedBuffer, std::shared_ptr<PackedBuffer>>(
        module, "_PackedBuffer", py::buffer_protocol())
        .def_buffer(&PackedBuffer::buffer_info);
    auto packed = py::class_<PackedMutationResults,
                             std::shared_ptr<PackedMutationResults>>(
                      module, "PackedMutationResults")
                      .def_property_readonly("changed_offsets",
                                             &PackedMutationResults::offsets)
                      .def_property_readonly("changed_spans",
                                             &PackedMutationResults::spans)
                      .def_property_readonly("changed_lengths",
                                             &PackedMutationResults::lengths)
                      .def_property_readonly("fully_covered",
                                             &PackedMutationResults::covered)
                      .def("materialize", &PackedMutationResults::materialize)
                      .def("__len__", &PackedMutationResults::size);
    packed.attr("__module__") = "treemendous.exact_batch";

    py::class_<ExactBatchManager>(module, "ExactBatchManager")
        .def(py::init<std::vector<Interval>, bool, std::size_t, std::size_t,
                      std::size_t, std::size_t, std::size_t>(),
             py::arg("domain"), py::arg("initially_available"),
             py::arg("max_operations"), py::arg("max_live_intervals"),
             py::arg("max_changed_spans"), py::arg("max_result_bytes"),
             py::arg("max_work_units"))
        .def("mutate_packed", &ExactBatchManager::mutate_packed,
             py::arg("operations"))
        .def("mutate_materialized", &ExactBatchManager::mutate_materialized,
             py::arg("operations"))
        .def("snapshot_data", &ExactBatchManager::snapshot_data)
        .def("_set_failpoint", &ExactBatchManager::set_failpoint,
             py::arg("name"), py::arg("occurrence") = 0)
        .def("_clear_failpoint", &ExactBatchManager::clear_failpoint);
}
#endif  // __has_include(<pybind11/pybind11.h>)
