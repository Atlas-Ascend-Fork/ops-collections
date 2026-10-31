/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#pragma once
#include <acl/acl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace aclco::detail::multimap {
inline void Check(aclError error, char const* operation)
{
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(std::string("StaticMultimap: ") + operation + " failed (ACL " + std::to_string(error) +
                                 ")");
    }
}
inline void Sync(aclrtStream stream) { Check(aclrtSynchronizeStream(stream), "synchronize"); }
inline void RequirePointer(void const* pointer)
{
    if (!pointer) {
        throw std::invalid_argument("StaticMultimap: null nonempty buffer");
    }
}
template <class T>
inline std::size_t Bytes(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("StaticMultimap: buffer byte size overflow");
    }
    return n * sizeof(T);
}
template <class T>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(std::size_t n) { Reserve(n); }
    // Caller must complete any device use before growing a live buffer.
    void Reserve(std::size_t n)
    {
        Bytes<T>(n);
        if (n <= capacity_) return;
        auto limit = std::numeric_limits<std::size_t>::max() / sizeof(T);
        auto grown = capacity_ <= limit / 2 ? capacity_ * 2 : n;
        auto capacity = n > grown ? n : grown;
        T* data = nullptr;
        Check(aclrtMalloc(reinterpret_cast<void**>(&data), Bytes<T>(capacity), ACL_MEM_MALLOC_HUGE_FIRST), "allocate");
        if (data_) (void)aclrtFree(data_);
        data_ = data;
        capacity_ = capacity;
    }
    ~Buffer()
    {
        if (data_) {
            (void)aclrtFree(data_);
        }
    }
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    Buffer(Buffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), capacity_(std::exchange(other.capacity_, 0)) {}
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this != &other) {
            if (data_) {
                (void)aclrtFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
        }
        return *this;
    }
    T* Data() const { return data_; }
    std::size_t Capacity() const { return capacity_; }

private:
    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};
template <class T>
inline T Read(T const* pointer)
{
    T result{};
    Check(aclrtMemcpy(&result, sizeof(T), pointer, sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST), "read scalar");
    return result;
}
} // namespace aclco::detail::multimap
