// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>

#include "openvino/core/version.hpp"

namespace ov {
namespace intel_gpu {

// GPU compiled cache trailer (written after the plugin's serialized blob body):
//
//   +------------------------------------------------+
//   | [ blob body produced by export_model() ]       |
//   +------------------------------------------------+
//   | metadata_version  : uint32                     |
//   | ov_version        : uint16 major,minor,patch   |
//   | driver_version    : uint32 len + bytes         |
//   | device_name       : uint32 len + bytes         |
//   | blob_data_size    : uint64 (size of blob body) |
//   | magic             : "OVGPU" (5 bytes)          |
//   +------------------------------------------------+
//
// Read order: locate the magic at the end of the stream, step back to read
// blob_data_size, then seek to blob_data_size to parse the rest.
//
// Versioning:
//   - Bump MAJOR when an existing field is removed or a field is inserted in the middle.
//     Older blobs are rejected on import.
//   - Bump MINOR when a new field is appended at the end. Older blobs remain
//     readable - newer fields are only read when the blob's minor is >= the
//     minor that introduced them.
constexpr std::string_view MAGIC_BYTES = "OVGPU";

constexpr uint32_t make_version(uint16_t major, uint16_t minor) {
    return (static_cast<uint32_t>(major) << 16) | minor;
}

constexpr uint16_t get_major(uint32_t version) {
    return static_cast<uint16_t>(version >> 16);
}

constexpr uint16_t get_minor(uint32_t version) {
    return static_cast<uint16_t>(version);
}

constexpr uint32_t CURRENT_METADATA_VERSION = make_version(1, 0);

class OpenvinoVersion final {
public:
    constexpr OpenvinoVersion(uint16_t major, uint16_t minor, uint16_t patch)
        : m_major(major),
          m_minor(minor),
          m_patch(patch) {}

    void read(std::istream& stream);
    void write(std::ostream& stream) const;

    bool operator!=(const OpenvinoVersion& other) const {
        return m_major != other.m_major || m_minor != other.m_minor || m_patch != other.m_patch;
    }

    uint16_t get_major() const { return m_major; }
    uint16_t get_minor() const { return m_minor; }
    uint16_t get_patch() const { return m_patch; }

private:
    uint16_t m_major;
    uint16_t m_minor;
    uint16_t m_patch;
};

constexpr OpenvinoVersion CURRENT_OPENVINO_VERSION{OPENVINO_VERSION_MAJOR,
                                                   OPENVINO_VERSION_MINOR,
                                                   OPENVINO_VERSION_PATCH};

void write_metadata(std::ostream& stream,
                    std::streampos blob_start,
                    const std::string& driver_version,
                    const std::string& device_name);

void verify_metadata(std::istream& stream,
                     std::streampos blob_start,
                     const std::string& driver_version,
                     const std::string& device_name,
                     uint64_t& out_blob_data_size);

}  // namespace intel_gpu
}  // namespace ov
