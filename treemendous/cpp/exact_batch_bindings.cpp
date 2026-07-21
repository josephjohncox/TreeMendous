// Experimental whole-batch-atomic exact geometry mutation engine.
#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
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

Coordinate decode_operation_field(const std::vector<std::uint8_t>& bytes,
                                  std::size_t row, std::size_t field) {
    Coordinate value;
    const std::size_t offset = row * kOperationBytes + field * sizeof(Coordinate);
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
        return py::buffer_info(PyBytes_AS_STRING(storage_.ptr()),
                               format_ == "B" ? 1 : 8, format_, shape_.size(),
                               shape_, strides_, true);
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
        return make_view(offsets_, "Q", {static_cast<py::ssize_t>(rows_ + 1)},
                         {8});
    }
    py::object lengths() const {
        return make_view(lengths_, "q", {static_cast<py::ssize_t>(rows_)}, {8});
    }
    py::object covered() const {
        return make_view(covered_, "B", {static_cast<py::ssize_t>(rows_)}, {1});
    }
    py::object spans() const {
        return make_view(spans_, "q",
                         {static_cast<py::ssize_t>(changed_count_), 2}, {16, 8});
    }

    py::tuple materialize() const {
        py::object domain = py::module_::import("treemendous.domain");
        py::object span_type = domain.attr("Span");
        py::object result_type = domain.attr("MutationResult");
        py::tuple output(rows_);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::uint64_t begin = binary_at<std::uint64_t>(offsets_, row);
            const std::uint64_t end = binary_at<std::uint64_t>(offsets_, row + 1);
            py::tuple changed(end - begin);
            for (std::uint64_t index = begin; index < end; ++index) {
                changed[index - begin] = span_type(
                    binary_at<Coordinate>(spans_, index * 2),
                    binary_at<Coordinate>(spans_, index * 2 + 1));
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
        T value;
        std::memcpy(&value, PyBytes_AS_STRING(storage.ptr()) + index * sizeof(T),
                    sizeof(T));
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

class BufferView {
  public:
    explicit BufferView(const py::handle& object) {
        if (PyObject_GetBuffer(object.ptr(), &view_,
                               PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
            throw py::error_already_set();
        }
        active_ = true;
    }
    ~BufferView() {
        if (active_) PyBuffer_Release(&view_);
    }
    BufferView(const BufferView&) = delete;
    BufferView& operator=(const BufferView&) = delete;
    const Py_buffer& get() const noexcept { return view_; }

  private:
    Py_buffer view_{};
    bool active_ = false;
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

bool native_int64_format(const char* format) {
    if (format == nullptr) return false;
    const std::string value(format);
    if (value == "q" || value == "l" || value == "@q" || value == "@l" ||
        value == "=q" || value == "=l") {
        return true;
    }
    if constexpr (std::endian::native == std::endian::little)
        return value == "<q" || value == "<l";
    if constexpr (std::endian::native == std::endian::big)
        return value == ">q" || value == ">l";
    return false;
}

[[noreturn]] void operation_error(std::size_t index, const std::string& message) {
    throw std::invalid_argument("operation " + std::to_string(index) + ": " +
                                message);
}

[[noreturn]] void operation_overflow(std::size_t index,
                                     const std::string& message) {
    throw std::overflow_error("operation " + std::to_string(index) + ": " +
                              message);
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

void checked_add(Coordinate& value, Coordinate delta, std::size_t index,
                 const char* description) {
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

template <typename T>
void append_binary(std::string& output, const T& value) {
    const std::size_t old_size = output.size();
    output.resize(old_size + sizeof(T));
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
    next.reserve(state.size() + 1);
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
    next.reserve(state.size() + 1);
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

class ExactBatchManager {
  public:
    ExactBatchManager(const std::vector<Interval>& domain, bool initially_available)
        : domain_(domain), state_(initially_available ? domain_ : std::vector<Interval>{}) {
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

    py::object mutate(const py::object& exporter) {
        // Deliberately precedes even buffer protocol dispatch: exporters may call back.
        MutationGuard mutation_guard(mutation_active_);
        BufferView buffer(exporter);
        const Py_buffer& view = buffer.get();
        if (view.itemsize != static_cast<Py_ssize_t>(sizeof(Coordinate)) ||
            !native_int64_format(view.format)) {
            throw py::type_error(
                "operations must have native-endian signed-int64 elements");
        }
        if (view.ndim != 1 && view.ndim != 2)
            throw py::type_error("operations must be flat 3*N or N by 3");
        if (!PyBuffer_IsContiguous(&view, 'C'))
            throw py::type_error("operations must be C-contiguous");
        Py_ssize_t rows = 0;
        if (view.ndim == 1) {
            if (view.shape[0] % 3 != 0)
                throw py::value_error("flat operations length must be a multiple of 3");
            rows = view.shape[0] / 3;
        } else {
            if (view.shape[1] != 3)
                throw py::value_error("two-dimensional operations must have shape N by 3");
            rows = view.shape[0];
        }
        if (rows < 0 || static_cast<std::uint64_t>(rows) >
                            std::numeric_limits<std::size_t>::max() / kOperationBytes)
            throw std::overflow_error("operation buffer is too large");
        const std::size_t operation_count = static_cast<std::size_t>(rows);
        const auto expected = operation_count * kOperationBytes;
        if (view.len < 0 || static_cast<std::size_t>(view.len) != expected)
            throw py::value_error("operation buffer has inconsistent shape and length");
        std::vector<std::uint8_t> operation_bytes(expected);
        if (expected != 0) std::memcpy(operation_bytes.data(), view.buf, expected);

        std::unique_lock<std::mutex> lock(state_mutex_);
        std::vector<Interval> scratch = state_;
        Coordinate scratch_total = total_;
        std::vector<std::uint64_t> offsets;
        std::vector<Interval> changed_spans;
        std::vector<Coordinate> changed_lengths;
        std::vector<std::uint8_t> covered_values;
        offsets.reserve(operation_count + 1);
        changed_lengths.reserve(operation_count);
        covered_values.reserve(operation_count);
        offsets.push_back(0);

        {
            py::gil_scoped_release release;
            for (std::size_t index = 0; index < operation_count; ++index) {
                const Coordinate opcode = decode_operation_field(operation_bytes, index, 0);
                const Coordinate start = decode_operation_field(operation_bytes, index, 1);
                const Coordinate end = decode_operation_field(operation_bytes, index, 2);
                if (opcode < 0 || opcode > 2)
                    operation_error(index, "unknown mutation opcode");
                const Coordinate target_length = checked_length(start, end, index);
                if (!inside_component(start, end))
                    operation_error(index,
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
                Coordinate changed_length = 0;
                for (const auto& span : changed) {
                    Coordinate part = checked_length(span.first, span.second, index);
                    checked_add(changed_length, part, index,
                                "signed changed-length overflow");
                }
                if (opcode == 0)
                    checked_add(scratch_total, changed_length, index,
                                "signed total overflow");
                else
                    scratch_total -= changed_length;
                if (scratch_total < 0)
                    operation_overflow(index, "signed total overflow");
                if (changed_spans.size() >
                    std::numeric_limits<std::uint64_t>::max() - changed.size())
                    operation_overflow(index, "packed changed-span offset overflow");
                changed_spans.insert(changed_spans.end(), changed.begin(), changed.end());
                offsets.push_back(static_cast<std::uint64_t>(changed_spans.size()));
                changed_lengths.push_back(changed_length);
                covered_values.push_back(covered ? 1 : 0);
                (void)target_length;
            }
        }

        std::string offsets_bytes;
        std::string spans_bytes;
        std::string lengths_bytes;
        std::string covered_bytes;
        offsets_bytes.reserve(offsets.size() * sizeof(std::uint64_t));
        spans_bytes.reserve(changed_spans.size() * 2 * sizeof(Coordinate));
        lengths_bytes.reserve(changed_lengths.size() * sizeof(Coordinate));
        covered_bytes.reserve(covered_values.size());
        for (const auto value : offsets) append_binary(offsets_bytes, value);
        for (const auto& span : changed_spans) {
            append_binary(spans_bytes, span.first);
            append_binary(spans_bytes, span.second);
        }
        for (const auto value : changed_lengths) append_binary(lengths_bytes, value);
        for (const auto value : covered_values) append_binary(covered_bytes, value);

        auto packed = std::make_shared<PackedMutationResults>(
            py::bytes(offsets_bytes), py::bytes(spans_bytes), py::bytes(lengths_bytes),
            py::bytes(covered_bytes), operation_count, changed_spans.size());
        // Force allocation of the Python wrapper before the commit point.
        py::object prepared = py::cast(packed);
        state_.swap(scratch);  // allocation-free and noexcept
        total_ = scratch_total;
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
        py::tuple intervals(copy.size());
        for (std::size_t index = 0; index < copy.size(); ++index)
            intervals[index] = py::make_tuple(copy[index].first, copy[index].second);
        return py::make_tuple(intervals, total);
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

    std::vector<Interval> domain_;
    mutable std::mutex state_mutex_;
    std::vector<Interval> state_;
    Coordinate total_ = 0;
    std::atomic<bool> mutation_active_{false};
};
}  // namespace

PYBIND11_MODULE(_exact_batch, module) {
    module.doc() = "Experimental exact whole-batch-atomic sorted-vector engine";
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
    packed.attr("__module__") = "treemendous.experimental.exact_batch";

    py::class_<ExactBatchManager>(module, "ExactBatchManager")
        .def(py::init<std::vector<Interval>, bool>(), py::arg("domain"),
             py::arg("initially_available"))
        .def("mutate_packed", &ExactBatchManager::mutate, py::arg("operations"))
        .def("snapshot_data", &ExactBatchManager::snapshot_data);
}
#endif  // __has_include(<pybind11/pybind11.h>)
