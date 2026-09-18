#pragma once

#include <cstddef>
#include <iterator>
#include <map>

namespace quactlize {

// A load is complete only after each byte of the local GGUF shard arrived once.
class upload_ranges {
    std::map<size_t, size_t> ranges;

    bool vacant(size_t begin, size_t end) const {
        auto next = ranges.lower_bound(begin);
        return (next == ranges.end() || next->first >= end) &&
               (next == ranges.begin() || std::prev(next)->second <= begin);
    }

    void insert(size_t begin, size_t end) {
        auto next = ranges.lower_bound(begin);
        if (next != ranges.begin()) {
            auto previous = std::prev(next);
            if (previous->second == begin) {
                begin = previous->first;
                ranges.erase(previous);
            }
        }
        if (next != ranges.end() && next->first == end) {
            end = next->second;
            ranges.erase(next);
        }
        ranges.emplace(begin, end);
    }

public:
    bool add(size_t offset, size_t width, size_t rows, size_t pitch, size_t total) {
        if (!width || !rows || pitch < width || offset > total || width > total - offset ||
                rows - 1 > (total - offset - width) / pitch) {
            return false;
        }
        if (pitch == width) {
            width *= rows;
            rows = 1;
        }
        for (size_t row = 0; row < rows; ++row) {
            const size_t begin = offset + row * pitch;
            if (!vacant(begin, begin + width)) { return false; }
        }
        for (size_t row = 0; row < rows; ++row) {
            const size_t begin = offset + row * pitch;
            insert(begin, begin + width);
        }
        return true;
    }

    bool complete(size_t total) const {
        return ranges.size() == 1 && ranges.begin()->first == 0 && ranges.begin()->second == total;
    }

    void clear() { ranges.clear(); }
};

} // namespace quactlize
