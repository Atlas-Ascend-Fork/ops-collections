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

#include <acl/acl.h>
#include <type_traits>
#include "detail/storages/storage_base.h"
#include "extent.h"
#include "utility/allocator.h"


// 1、实现基本的内存管理功能 2、自定义alloctor和释放
namespace aclco {
  /**
   * @brief 桶存储容器类，实现基本的内存管理功能
   * 
   * @note 实际容量会向上取整到BucketSize的倍数
   * 
   * @tparam T 存储的元素类型
   * @tparam BucketSize 每个桶中包含的元素个数
   * @tparam Extent 容量大小类型，默认为 std::size_t
   * @tparam Allocator 内存分配器类型，默认为 DefaultAllocator<T>
   *
   * @warning 根据 static_map 中的静态断言检查以及 Pair 结构的限制，元素类型 T 的大小不能超过8字节
   */
template <typename T,
          int32_t BucketSize,
          typename Extent = std::size_t,
          typename Allocator = DefaultAllocator<T>> //unfinish...
class BucketStorage {
 public:
  static_assert(std::is_integral_v<Extent> && std::is_unsigned_v<Extent>,
                "SizeType must be an unsigned integral type to keep probe indexes valid");
  static constexpr uint32_t bucketSize = BucketSize; ///< Number of elements processed per bucket

  using ExtentType = Extent;
  using SizeType = Extent; ///< Storage size type
  using ValueType = T; ///< Slot type
  using AllocatorType = typename std::allocator_traits<Allocator>::template rebind_alloc<ValueType>;
  //using ref_type = bucket_storage_ref<value_type, bucket_size, extent_type>;  // 后续补齐ref

  /**
   * @brief 创建指定容器的桶的存储容量
   * 
   * @note 实际容量会向上取整到 BucketSize 的倍数
   * 
   * @param capacity 容器的容量
   * @param allocator 内存分配器
   * @throw std::bad_alloc 非零容量所需的存储分配失败时抛出
   */
  explicit constexpr BucketStorage(Extent capacity,
                                   Allocator const& allocator = {});
  BucketStorage(BucketStorage&&) = default; //Move constructor
  BucketStorage& operator=(BucketStorage&&) = default;
  ~BucketStorage() = default;
  BucketStorage(BucketStorage const&) = delete;
  BucketStorage& operator=(BucketStorage const&) = delete;

  /**
   * @brief 获取Device侧容器内部存储的指针
   * 
   * @return 指向Device侧容器内部存储的指针
   * 
   * @warning 返回Device侧的内存指针，不能在host侧直接访问
   */
  constexpr ValueType* Data() const noexcept;

  /**
   * @brief 同步初始化桶存储
   * 
   * @note 这是一个同步操作，会阻塞直到插入完成
   * 
   * @param key Device侧指向键的指针
   * @param stream ACL流
   */
  void Initialize(ValueType* key, aclrtStream stream = nullptr);
  
  /**
   * @brief 异步初始化桶存储
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @param key Device侧指向键的指针
   * @param stream ACL流
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   */
  void InitializeAsync(ValueType* key, aclrtStream stream = nullptr);

  /**
   * @brief 获取容器的实际容量
   * 
   * @note 实际容量会向上取整到BuketSize的倍数
   * 
   * @return 实际容量
   */
  constexpr SizeType Capacity() const noexcept;

  /**
   * @brief 获取当前容器的容量范围
   * 
   * @note 该方法返回构造函数中指定的容量范围
   * 
   * @return 返回当前容量范围
   */
  constexpr ExtentType GetExtent() const noexcept;
  
 private:
  using SlotDeleterType = CustomDeleter<AllocatorType>;
  Extent extent_;
  AllocatorType allocator_;
  std::unique_ptr<ValueType, SlotDeleterType> slots_;
};
}
#include "detail/storages/bucket_storage.inl"
