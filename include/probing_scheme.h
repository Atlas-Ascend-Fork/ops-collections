/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "macros.h"
#include "utility/is_same.h"

namespace aclco {
namespace detail {
template <typename T, typename = void>
struct IsEqualityComparable : FalseType {};

template <typename T>
struct IsEqualityComparable<T, std::void_t<decltype(std::declval<T const&>() == std::declval<T const&>())>>
    : TrueType {};
}  // namespace detail
/**
 * @brief 公共线性探测方案类
 *
 * @note 当冲突较少时，线性探测效率较高，例如在低负载率或低多重性情况下
 *
 * @note `Hash` 应该是可调用对象类型
 *
 * @tparam Hash 一元可调用类型
 */
template <typename Hash>
class LinearProbing {
 public:
  using Hasher = Hash; /// Hash function type

  /**
   *@brief 构造一个使用哈希函数可调用对象的线性探测方案
   *
   * @param hash 哈希函数对象
   */
   COLLECTION_HOST_DEVICE constexpr LinearProbing(Hash const& hash = {});

  /**
   *@brief 使用给定的哈希函数对象，创建当前探测方案的一个副本
   *
   * @tparam NewHash 新的哈希函数对象类型
   *
   * @param hash 哈希函数对象
   *
   * @return 返回当前探测方案的副本
   */
  template <typename NewHash>
  [[nodiscard]] COLLECTION_HOST_DEVICE constexpr LinearProbing<NewHash> RebindHash(NewHash const& hash) const noexcept;

  /**
   * @brief 返回一个探测迭代器
   *
   * @tparam BucketSize 桶的大小
   * @tparam ProbeKey 探测键的类型
   * @tparam Extent 边界类型
   *
   * @param probeKey 探测键
   * @param upperBound 迭代的上键
   * @return 一个迭代器，其 value_type 可转换为槽索引类型
   */
  template <int32_t BucketSize, typename ProbeKey, typename Extent>
  COLLECTION_HOST_DEVICE constexpr auto MakeIterator(ProbeKey probeKey,
                               Extent upperBound) const noexcept;

  /**
   * @brief 获取用于哈希键的函数
   *
   * @return 用于哈希键的函数对象
   */
  COLLECTION_HOST_DEVICE constexpr Hasher HashFunction() const noexcept;

  template <typename H = Hash, std::enable_if_t<detail::IsEqualityComparable<H>::value, int> = 0>
  COLLECTION_HOST_DEVICE constexpr bool operator==(LinearProbing const& other) const noexcept
  {
    return hash_ == other.hash_;
  }

 private:
  Hash hash_;
};

template <typename Hash1, typename Hash2 = Hash1>
class DoubleHashing {
 public:
  using Hasher = std::tuple<Hash1, Hash2>;  ///< Hash function type


  COLLECTION_HOST_DEVICE constexpr DoubleHashing(Hash1 const& hash1 = {}, Hash2 const& hash2 = {1});

  COLLECTION_HOST_DEVICE constexpr DoubleHashing(std::tuple<Hash1, Hash2> const& hash);

  template <typename NewHash>
  [[nodiscard]] COLLECTION_HOST_DEVICE constexpr auto RebindHash(NewHash const& hash) const;

  template <int32_t BucketSize, typename ProbeKey, typename Extent>
  COLLECTION_HOST_DEVICE constexpr auto MakeIterator(ProbeKey probeKey,
                                                   Extent upperBound) const noexcept;

  COLLECTION_HOST_DEVICE constexpr Hasher HashFunction() const noexcept;

  template <typename H1 = Hash1, typename H2 = Hash2,
            std::enable_if_t<detail::IsEqualityComparable<H1>::value && detail::IsEqualityComparable<H2>::value, int> = 0>
  COLLECTION_HOST_DEVICE constexpr bool operator==(DoubleHashing const& other) const noexcept
  {
    return hash1_ == other.hash1_ && hash2_ == other.hash2_;
  }

 private:
  Hash1 hash1_;
  Hash2 hash2_;
};

/**
 * @brief 用于判断给定探测方案是否为DoubleHashing类型
 *
 * @tparam T 输入的探测方案类型
 */
template <typename T>
struct IsDoubleHashing : FalseType {};

/**
 * @brief 用于判断给定探测方案是否为DoubleHashing类型
 *
 * @tparam Hash1 第一个哈希函数对象
 * @tparam Hash2 第二个哈希函数对象
 */
template <typename Hash1, typename Hash2>
struct IsDoubleHashing<aclco::DoubleHashing<Hash1, Hash2>> : TrueType {};
}  // namespace aclco

#include "detail/probing_scheme/probing_scheme_impl.inl"