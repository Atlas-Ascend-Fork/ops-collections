/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "roaring.h"

namespace {

void WriteFile(std::filesystem::path const& path, std::vector<char> const& contents)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.close();
    if (!output) {
        throw std::runtime_error("cannot write output file: " + path.string());
    }
}

void WritePortable(std::filesystem::path const& path, roaring_bitmap_t const* bitmap)
{
    size_t const size = roaring_bitmap_portable_size_in_bytes(bitmap);
    std::vector<char> buffer(size);
    size_t const written = roaring_bitmap_portable_serialize(bitmap, buffer.data());
    if (written != size) {
        throw std::runtime_error("unexpected portable serialization size for: " + path.string());
    }
    WriteFile(path, buffer);
}

void GenerateUint32Files(std::filesystem::path const& outputDirectory)
{
    roaring_bitmap_t* bitmap = roaring_bitmap_create();
    if (bitmap == nullptr) {
        throw std::runtime_error("cannot create uint32 RoaringBitmap");
    }

    try {
        for (uint32_t k = 0; k < 100000; k += 1000) {
            roaring_bitmap_add(bitmap, k);
        }
        for (uint32_t k = 100000; k < 200000; ++k) {
            roaring_bitmap_add(bitmap, 3U * k);
        }
        for (uint32_t k = 700000; k < 800000; ++k) {
            roaring_bitmap_add(bitmap, k);
        }

        WritePortable(outputDirectory / "bitmapwithoutruns.bin", bitmap);
        roaring_bitmap_run_optimize(bitmap);
        WritePortable(outputDirectory / "bitmapwithruns.bin", bitmap);
    } catch (...) {
        roaring_bitmap_free(bitmap);
        throw;
    }
    roaring_bitmap_free(bitmap);
}

void GeneratePortableUint64File(std::filesystem::path const& outputDirectory)
{
    roaring64_bitmap_t* bitmap = roaring64_bitmap_create();
    if (bitmap == nullptr) {
        throw std::runtime_error("cannot create uint64 RoaringBitmap");
    }

    try {
        for (uint64_t i = 0; i < 2; ++i) {
            uint64_t const base = i << 32U;
            roaring64_bitmap_add_range_closed(bitmap, base | 0x00000U, base | 0x09000U);
            roaring64_bitmap_add_range_closed(bitmap, base | 0x0A000U, base | 0x10000U);
            roaring64_bitmap_add(bitmap, base | 0x20000U);
            roaring64_bitmap_add(bitmap, base | 0x20005U);
            for (uint64_t j = 0; j < 0x10000U; j += 2U) {
                roaring64_bitmap_add(bitmap, base | (0x80000U + j));
            }
        }

        roaring64_bitmap_run_optimize(bitmap);
        size_t const size = roaring64_bitmap_portable_size_in_bytes(bitmap);
        std::vector<char> buffer(size);
        size_t const written = roaring64_bitmap_portable_serialize(bitmap, buffer.data());
        if (written != size) {
            throw std::runtime_error("unexpected portable uint64 serialization size");
        }
        WriteFile(outputDirectory / "portable_bitmap64.bin", buffer);
    } catch (...) {
        roaring64_bitmap_free(bitmap);
        throw;
    }
    roaring64_bitmap_free(bitmap);
}

std::filesystem::path ParseOutputDirectory(int argc, char** argv)
{
    if (argc != 3 || std::string(argv[1]) != "--output-dir") {
        throw std::invalid_argument("usage: roaring_bitmap_testdata_generator --output-dir <directory>");
    }
    return argv[2];
}

} // namespace

int main(int argc, char** argv)
{
    try {
        auto const outputDirectory = ParseOutputDirectory(argc, argv);
        std::filesystem::create_directories(outputDirectory);
        GenerateUint32Files(outputDirectory);
        GeneratePortableUint64File(outputDirectory);
        std::cout << "Generated RoaringBitmap test data in " << outputDirectory << '\n';
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "Failed to generate RoaringBitmap test data: " << error.what() << '\n';
        return 1;
    }
}
