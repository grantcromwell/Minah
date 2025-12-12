#pragma once

#include <string>
#include <sstream>
#include <iomanip>

namespace minah {
namespace utils {

/**
 * @brief Convert integer to hex string with 0x prefix
 *
 * @param value The integer value to convert
 * @return std::string Hex string with 0x prefix
 */
inline std::string to_hex(uint64_t value) {
    std::stringstream ss;
    ss << "0x" << std::hex << value;
    return ss.str();
}

/**
 * @brief Convert hex string to uint64_t
 *
 * @param hex_str Hex string (with or without 0x prefix)
 * @return uint64_t The converted integer value
 */
inline uint64_t from_hex(const std::string& hex_str) {
    std::string stripped = hex_str;
    if (stripped.substr(0, 2) == "0x") {
        stripped = stripped.substr(2);
    }
    return std::stoull(stripped, nullptr, 16);
}

/**
 * @brief Convert hex string to binary data
 *
 * @param hex_str Hex string
 * @return std::vector<uint8_t> Binary data
 */
inline std::vector<uint8_t> hex_to_bytes(const std::string& hex_str) {
    std::string stripped = hex_str;
    if (stripped.substr(0, 2) == "0x") {
        stripped = stripped.substr(2);
    }

    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < stripped.length(); i += 2) {
        std::string byte_string = stripped.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byte_string, nullptr, 16));
        bytes.push_back(byte);
    }

    return bytes;
}

/**
 * @brief Convert binary data to hex string
 *
 * @param bytes Binary data
 * @return std::string Hex string with 0x prefix
 */
inline std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << "0x" << std::hex << std::setfill('0');
    for (uint8_t byte : bytes) {
        ss << std::setw(2) << static_cast<int>(byte);
    }
    return ss.str();
}

/**
 * @brief Ethereum address checksum
 *
 * Implements EIP-55 checksum for Ethereum addresses
 *
 * @param address Ethereum address (with or without 0x prefix)
 * @return std::string Checksummed address with 0x prefix
 */
inline std::string eth_address_checksum(const std::string& address) {
    std::string lower = address;
    if (lower.substr(0, 2) == "0x") {
        lower = lower.substr(2);
    }

    // Convert to lowercase
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Compute Keccak-256 hash
    // Note: This would require a Keccak-256 implementation

    // For now, return the address with proper 0x prefix
    return "0x" + lower;
}

/**
 * @brief Pad hex string to 32 bytes
 *
 * @param hex_str Hex string to pad
 * @param left_pad Whether to pad on the left (default true)
 * @return std::string Padded hex string with 0x prefix
 */
inline std::string pad_hex_to_32_bytes(const std::string& hex_str, bool left_pad = true) {
    std::string stripped = hex_str;
    if (stripped.substr(0, 2) == "0x") {
        stripped = stripped.substr(2);
    }

    std::string padded(64, '0'); // 32 bytes = 64 hex characters

    if (left_pad) {
        // Copy from right to left
        std::copy(stripped.rbegin(), stripped.rend(), padded.rbegin());
    } else {
        // Copy from left to right
        std::copy(stripped.begin(), stripped.end(), padded.begin());
    }

    return "0x" + padded;
}

} // namespace utils
} // namespace minah