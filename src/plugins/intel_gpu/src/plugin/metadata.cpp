// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/metadata.hpp"

#include <istream>
#include <ostream>

#include "openvino/core/except.hpp"
#include "openvino/core/version.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void write_string(std::ostream& stream, const std::string& value) {
    const uint32_t len = static_cast<uint32_t>(value.size());
    stream.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len > 0) {
        stream.write(value.data(), len);
    }
}

std::string read_string(std::istream& stream) {
    uint32_t len = 0;
    stream.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string value(len, '\0');
    if (len > 0) {
        stream.read(value.data(), len);
    }
    return value;
}

}  // namespace

void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&m_major), sizeof(m_major));
    stream.read(reinterpret_cast<char*>(&m_minor), sizeof(m_minor));
    stream.read(reinterpret_cast<char*>(&m_patch), sizeof(m_patch));
}

void OpenvinoVersion::write(std::ostream& stream) const {
    stream.write(reinterpret_cast<const char*>(&m_major), sizeof(m_major));
    stream.write(reinterpret_cast<const char*>(&m_minor), sizeof(m_minor));
    stream.write(reinterpret_cast<const char*>(&m_patch), sizeof(m_patch));
}

void write_metadata(std::ostream& stream,
                    std::streampos blob_start,
                    const std::string& driver_version,
                    const std::string& device_name) {
    const uint64_t blob_data_size = static_cast<uint64_t>(stream.tellp() - blob_start);

    constexpr uint32_t metadata_version = CURRENT_METADATA_VERSION;
    stream.write(reinterpret_cast<const char*>(&metadata_version), sizeof(metadata_version));
    CURRENT_OPENVINO_VERSION.write(stream);
    write_string(stream, driver_version);
    write_string(stream, device_name);
    stream.write(reinterpret_cast<const char*>(&blob_data_size), sizeof(blob_data_size));
    stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size());
}

void verify_metadata(std::istream& stream,
                     std::streampos blob_start,
                     const std::string& driver_version,
                     const std::string& device_name,
                     uint64_t& out_blob_data_size) {
    const std::streampos cur = stream.tellg();
    stream.seekg(0, std::ios::end);
    const std::streampos stream_end = stream.tellg();

    constexpr std::streamoff magic_size = static_cast<std::streamoff>(MAGIC_BYTES.size());
    constexpr std::streamoff size_size = sizeof(uint64_t);

    OPENVINO_ASSERT(stream_end - blob_start >= magic_size + size_size,
                    "[GPU] Incompatible cache blob: too small to contain GPU metadata trailer. "
                    "The blob was likely produced by an older OpenVINO version. "
                    "Please regenerate the cache with the current OpenVINO version.");

    stream.seekg(stream_end - magic_size);
    char magic_buf[MAGIC_BYTES.size()];
    stream.read(magic_buf, magic_size);
    OPENVINO_ASSERT(std::string_view(magic_buf, MAGIC_BYTES.size()) == MAGIC_BYTES,
                    "[GPU] Incompatible cache blob: missing GPU metadata magic. "
                    "The blob was likely produced by an older OpenVINO version. "
                    "Please regenerate the cache with the current OpenVINO version.");

    uint64_t blob_data_size = 0;
    stream.seekg(stream_end - magic_size - size_size);
    stream.read(reinterpret_cast<char*>(&blob_data_size), sizeof(blob_data_size));

    stream.seekg(blob_start + std::streamoff(blob_data_size));
    uint32_t metadata_version = 0;
    stream.read(reinterpret_cast<char*>(&metadata_version), sizeof(metadata_version));
    OPENVINO_ASSERT(get_major(metadata_version) == get_major(CURRENT_METADATA_VERSION),
                    "[GPU] Incompatible cache blob: metadata major version ",
                    get_major(metadata_version),
                    ".",
                    get_minor(metadata_version),
                    " is not supported by this OpenVINO build (current ",
                    get_major(CURRENT_METADATA_VERSION),
                    ".",
                    get_minor(CURRENT_METADATA_VERSION),
                    "). Please regenerate the cache with the current OpenVINO version.");
    OPENVINO_ASSERT(get_minor(metadata_version) <= get_minor(CURRENT_METADATA_VERSION),
                    "[GPU] Incompatible cache blob: metadata minor version ",
                    get_major(metadata_version),
                    ".",
                    get_minor(metadata_version),
                    " is newer than this OpenVINO build (current ",
                    get_major(CURRENT_METADATA_VERSION),
                    ".",
                    get_minor(CURRENT_METADATA_VERSION),
                    "). Please use a newer OpenVINO version to consume the cache.");

    OpenvinoVersion blob_ov_version{0, 0, 0};
    blob_ov_version.read(stream);
    OPENVINO_ASSERT(!(blob_ov_version != CURRENT_OPENVINO_VERSION),
                    "[GPU] Incompatible cache blob: OpenVINO version mismatch (blob: ",
                    blob_ov_version.get_major(),
                    ".",
                    blob_ov_version.get_minor(),
                    ".",
                    blob_ov_version.get_patch(),
                    ", current: ",
                    CURRENT_OPENVINO_VERSION.get_major(),
                    ".",
                    CURRENT_OPENVINO_VERSION.get_minor(),
                    ".",
                    CURRENT_OPENVINO_VERSION.get_patch(),
                    "). Please regenerate the cache with the current OpenVINO version.");

    const std::string blob_driver_version = read_string(stream);
    const std::string blob_device_name = read_string(stream);
    OPENVINO_ASSERT(blob_driver_version == driver_version,
                    "[GPU] Incompatible cache blob: GPU driver version mismatch (blob: '",
                    blob_driver_version,
                    "', current: '",
                    driver_version,
                    "'). Please regenerate the cache with the current driver.");
    OPENVINO_ASSERT(blob_device_name == device_name,
                    "[GPU] Incompatible cache blob: GPU device mismatch (blob: '",
                    blob_device_name,
                    "', current: '",
                    device_name,
                    "'). Please regenerate the cache for this device.");

    out_blob_data_size = blob_data_size;
    stream.seekg(cur);
}

}  // namespace intel_gpu
}  // namespace ov
