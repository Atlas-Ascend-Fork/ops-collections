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

#include <tuple>

#include "utility/math_utils.h"

namespace aclco {
namespace detail {

/**
 * @brief 探测方案类(包含线性探测和双重哈希探测)
 * 
 * @tparam Extent 数据范围类型
 */
template <typename Extent>
class ProbingIterator {
 public:
  using ExtentType = Extent;                            ///< Extent type
  using SizeType   = ExtentType;                        ///< Size type, 后续需要换成 typename ExtentType::value_type 获取 std::array<std::uint64_t, 2>的type

  /**
   * @brief 构造探测迭代器
   * 
   * @note 此构造函数支持线性探测（probeStep = BuketSize）和双重哈希探测
   * 
   * @param start 标志迭代开始
   * @param probeStep 双重哈希探测的步长
   * @param upperBound 迭代的边界，此处为上界
   * 
   */
  COLLECTION_HOST_DEVICE constexpr ProbingIterator(SizeType start,
                             SizeType probeStep,
                             ExtentType upperBound)
    : idx_{start}, probeStep_{probeStep}, upperBound_{upperBound}
  {
    static_assert(!std::is_same_v<SizeType, bool>, "SizeType must not be bool");
    //将双重哈希探测引入aclco时，需要对其API进行修订，以支持新的冲突解决策略
  }

  /**
   * @brief 用于从指针中取出指向的值的操作符
   * 
   * @note 此操作符用于在循环中获取当前有效索引
   * 
   * @return 返回当前的索引
   * 
   */
  COLLECTION_HOST_DEVICE constexpr auto operator*() const { return idx_; }

  /**
   * @brief 后缀自增运算符
   * 
   * @return 自增前的迭代
   */
  COLLECTION_HOST_DEVICE constexpr auto operator++(int32_t)
  {
    auto temp = *this;
    ++(*this);
    return temp;
  }

  /**
   * @brief 前缀自增运算符
   * 
   * @return 当前的迭代
   */
  COLLECTION_HOST_DEVICE constexpr auto operator++()
  {

    // Valid iterators have idx_ < upperBound_ and probeStep_ <= upperBound_.
    // Subtract before adding so wrapping also works near the integer limit.
    auto const remaining = upperBound_ - probeStep_;
    idx_ = idx_ >= remaining ? idx_ - remaining : idx_ + probeStep_;
    return *this;
  }


 private:
  SizeType idx_;
  SizeType probeStep_;
  ExtentType upperBound_;
};
}
/**
 * @brief 线性探测策略，实现线性探测的冲突解决机制
 */
template <typename Hash>
COLLECTION_HOST_DEVICE constexpr LinearProbing<Hash>::LinearProbing(Hash const& hash)
  : hash_{hash}
{
}

/**
 * @brief 重新绑定哈希函数
 */
template <typename Hash>
template <typename NewHash>
COLLECTION_HOST_DEVICE constexpr LinearProbing<NewHash> LinearProbing<Hash>::RebindHash(
  NewHash const& hash) const noexcept
{
  return LinearProbing<NewHash>{hash};
}

/**
 * @brief 创建一个探测迭代器
 */
template <typename Hash>
template <int32_t BucketSize, typename ProbeKey, typename Extent>
COLLECTION_HOST_DEVICE constexpr auto LinearProbing<Hash>::MakeIterator(ProbeKey probeKey,
                                                   Extent upperBound) const noexcept
{
  static_assert(BucketSize > 0, "BucketSize must be greater than zero");
  using SizeType = Extent;
  SizeType const value = aclco::SanitizeHash<SizeType>(hash_(probeKey));
  SizeType const bucketCount = upperBound / BucketSize;
  SizeType const init = (value % bucketCount) * BucketSize;
  SizeType const step = static_cast<SizeType>(BucketSize);             
  return detail::ProbingIterator<Extent>{init, step, upperBound};
}

/**
 * @brief 获取当前绑定的哈希函数
 */
template <typename Hash>
COLLECTION_HOST_DEVICE constexpr typename LinearProbing<Hash>::Hasher LinearProbing<Hash>::HashFunction() const noexcept
{
  return hash_;
}

/**
 * @brief 双重哈希探测策略，基于两个独立哈希函数实现双重哈希探测机制
 */
template <typename Hash1, typename Hash2>
COLLECTION_HOST_DEVICE constexpr DoubleHashing<Hash1, Hash2>::DoubleHashing(
  Hash1 const& hash1, Hash2 const& hash2)
  : hash1_{hash1}, hash2_{hash2}
{
}

/**
 * @brief 构造函数
 */
template <typename Hash1, typename Hash2>
COLLECTION_HOST_DEVICE constexpr DoubleHashing<Hash1, Hash2>::DoubleHashing(std::tuple<Hash1, Hash2> const& hash)
    : hash1_{std::get<0>(hash)}, hash2_{std::get<1>(hash)}
{}

/**
 * @brief 重新绑定哈希函数
 */
template <typename Hash1, typename Hash2>
template <typename NewHash>
COLLECTION_HOST_DEVICE constexpr auto DoubleHashing<Hash1, Hash2>::RebindHash(
  NewHash const& hash) const
{
  auto const hash1         = std::get<0>(hash);
  auto const hash2         = std::get<1>(hash);
  using Hash1Type          = std::decay_t<decltype(hash1)>;
  using Hash2Type          = std::decay_t<decltype(hash2)>;
  return DoubleHashing<Hash1Type, Hash2Type>{hash1, hash2};
}

/**
 * @brief 创建一个探测迭代器
 */
template <typename Hash1, typename Hash2>
template <int32_t BucketSize, typename ProbeKey, typename Extent>
COLLECTION_HOST_DEVICE constexpr auto DoubleHashing<Hash1, Hash2>::MakeIterator(
  ProbeKey probeKey, Extent upperBound) const noexcept
{
  using SizeType = Extent;
  SizeType const value1 = aclco::SanitizeHash<SizeType>(hash1_(probeKey));
  SizeType const value2 = aclco::SanitizeHash<SizeType>(hash2_(probeKey));  
  SizeType const bucketCount = upperBound / BucketSize;

  SizeType const init = (value1 % bucketCount) * BucketSize;
  SizeType const step = (value2 % (bucketCount - 1) + 1) * BucketSize;

  return detail::ProbingIterator<Extent>{init, step, upperBound}; 
}

/**
 * @brief 获取当前绑定的哈希函数
 */
template <typename Hash1, typename Hash2>
COLLECTION_HOST_DEVICE constexpr typename DoubleHashing<Hash1, Hash2>::Hasher
DoubleHashing<Hash1, Hash2>::HashFunction() const noexcept
{
  return std::tuple{hash1_, hash2_};
}
}
