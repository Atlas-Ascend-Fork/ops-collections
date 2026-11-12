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

#include <type_traits>

#include "macros.h"

namespace aclco {
static constexpr std::size_t dynamicExtent = static_cast<std::size_t>(-1);

/**
 * @brief 静态 Extent 类
 *
 * @tparam SizeType 容器值的数据类型
 * @tparam N 容量大小
 */
template <typename SizeType, std::size_t N = dynamicExtent>
struct Extent {
  static_assert(std::is_integral_v<SizeType> && std::is_unsigned_v<SizeType>,
                "SizeType must be an unsigned integral type");
  using ValueType = SizeType;
  constexpr Extent() = default;

  /**
   * @brief 构造函数
   * 
   * @param size 容量的值
   */
  COLLECTION_HOST_DEVICE constexpr Extent(SizeType) noexcept {}

  /**
   * @brief 转化为 SizeType 的操作符
   * 
   * @return 返回模板参数 N 指定的容量值
   */
  COLLECTION_HOST_DEVICE constexpr operator SizeType() const noexcept { return N; };
};

  /**
   * @brief 动态 Extent 类 ，当模板参数 N 等于 dynamicExtent 时，使用此特化版本。
   * 
   * @note 包含一个运行时的大小成员size_
   * 
   * @tparam SizeType 容器值的数据类型
   */
template <typename SizeType>
struct Extent<SizeType, dynamicExtent> {
  static_assert(std::is_integral_v<SizeType> && std::is_unsigned_v<SizeType>,
                "SizeType must be an unsigned integral type");
  using ValueType = SizeType;

    /**
   * @brief 构造函数
   * 
   * @param size 容量的值，在运行时被存储
   */
  COLLECTION_HOST_DEVICE constexpr Extent(SizeType size) noexcept : size_(size) {}

  /**
   * @brief 转化为 SizeType 的操作符
   * 
   * @note 此操作用于将动态 Extent 对象转换为实际的容量数值
   * 
   * @return 返回运行时存储的 size_t 值
   */
  COLLECTION_HOST_DEVICE constexpr operator SizeType() const noexcept { return size_; };

 private:
  SizeType size_;
};

/**
 * @brief 创建一个与探测策略和存储策略兼容的 Extent
 * 
 * @note 哈希表的实际容量应完全由该工具函数的返回值决定，因为其输出取决于请求的下限大小、探测方法以及存储布局。
 * 该工具函数在容器构造过程中被内部调用；而对于容器的引用构造，则需由用户自行使用此函数，以确定容器构造函数所需
 * 的容量参数。
 * 
 * @tparam ProingScheme 探测策略类型
 * @tparam Storage 存储策略类型
 * @tparam SizeType Extent 的值类型
 * @tparam N 容量大小
 * 
 * @param ext 输入的 Extent 对象
 * 
 * @throw 输入的 Extent 对象不合理的时候
 * 
 * @return 返回一个与给定策略兼容的有效 Extent
 */
template <typename ProbingScheme, typename Storage, typename SizeType, std::size_t N>
[[nodiscard]] auto constexpr MakeValidExtent(Extent<SizeType, N> ext);
}

#include "detail/extent/extent.inl"