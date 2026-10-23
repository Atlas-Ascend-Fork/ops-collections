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

#include <memory>

#include <acl/acl.h>

#include "detail/open_addressing/open_addressing_impl.h"
#include "hash_functions.h"
#include "macros.h"
#include "pair.h"
#include "extent.h"

namespace aclco {
static constexpr size_t defaultMapBucketSize = 5;

template <class Key,
          class T,
          class Extent = Extent<size_t>,
          class KeyEqual = aclco::EqualTo<Key>,
          class ProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<Key>>,
          class Storage = Storage<defaultMapBucketSize>>
class StaticMap {
  static_assert(sizeof(Key) <= 8, "Container dose not support key type size bigger than 8 bytes.");

  static_assert(sizeof(T) <= 8, "Container dose not support payload type size bigger than 8 bytes.");

  using ImplType = OpenAddressingImpl<Key,
                                      Pair<Key, T>,
                                      Extent,
                                      KeyEqual,
                                      ProbingScheme,
                                      Storage>;

 public:
  static constexpr auto bucketSize = ImplType::bucketSize;

  using SizeType = typename ImplType::SizeType;
  using KeyType = typename ImplType::KeyType;
  using ValueType = typename ImplType::ValueType;

  StaticMap(StaticMap const&) = delete;
  StaticMap& operator=(StaticMap const&) = delete;

  StaticMap(StaticMap&&) = default;
  StaticMap& operator=(StaticMap &&) = default;
  ~StaticMap() = default;

  /**
   * @brief 构造函数：创建指定容量的 static_map 容器
   * 
   * @param capacity map的容量
   * @param emptyKey 表示空键的标记值
   * @param emptyValue 表示空值的标记值
   * @param pred 键比较器，默认为 KeyEqual()
   * @param probingScheme 探测策略，默认为 ProbingScheme()
   * @param storage 存储策略，默认为 Storage<BucketSize>()
   * @param stream ACL流，默认为 nullptr
   * 
   * @note 实际容量会向上取整到BucketSize的倍数
   * 
   * @warning Key和Value类型大小不能超过8字节
   */
  constexpr StaticMap(Extent capacity,
                      Key emptyKey,
                      T emptyValue,
                      KeyEqual const& pred = {},
                      ProbingScheme const& probingScheme = {},
                      Storage storage = {},
                      aclrtStream stream = nullptr);

  /**
   * @brief 配置删除墓碑键，启用墓碑删除（Erase 写 erasedKey 而非空键）。
   *
   * 启用后 Erase 不会因删除而在探测链中留下空洞，从而支持并发删除的正确性，
   * 且 miss/不命中可在真空槽早停（避免退化到 O(capacity) 扫表）。
   *
   * @param erasedKey 删除墓碑键
   * @param stream ACL流
   *
   * @warning erasedKey 必须是与 emptyKey 及任何有效键都不同的保留值
   */
  void SetErasedKey(Key erasedKey, aclrtStream stream);

  /**
   * @brief 同步插入键值对到map中
   * 
   * @param values Device侧指向键值对数组的指针
   * @param valueNum 要插入的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @return 插入失败的键值对数量
   * 
   * @note 这是一个同步操作，会阻塞直到插入完成
   * 
   * @warning valueNum 参数必须与 values 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 values.size() 作为 valueNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see InsertAsync 用于异步插入操作
   */
  SizeType Insert(void *values, Extent valueNum, aclrtStream stream);

  /**
   * @brief 异步插入键值对到map中
   * 
   * @param values Device侧指向键值对数组的指针
   * @param valueNum 要插入的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see Insert 用于同步插入操作
   */
  void InsertAsync(void *values, Extent valueNum, aclrtStream stream);

  /**
   * @brief 同步插入或更新键值对到map中
   * 
   * @param values Device侧指向键值对数组的指针
   * @param valueNum 要插入或更新的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @return 插入或更新失败的键值对数量（仅当表满时才会失败）
   * 
   * @note 这是一个同步操作，会阻塞直到操作完成
   * @note 如果键已存在，则更新对应的值；如果键不存在，则插入新的键值对
   * 
   * @warning valueNum 参数必须与 values 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 values.size() 作为 valueNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see InsertOrAssignAsync 用于异步插入或更新操作
   */
  SizeType InsertOrAssign(void *values, Extent valueNum, aclrtStream stream);

  /**
   * @brief 同步插入或更新（计数重载）
   *
   * @param assignedNum 输出已存在 key 被 assign 的个数（不计入插入）
   * @return 插入或更新失败的键值对数量
   */
  SizeType InsertOrAssign(void *values, Extent valueNum, aclrtStream stream, SizeType &assignedNum);

  /**
   * @brief 异步插入或更新键值对到map中
   * 
   * @param values Device侧指向键值对数组的指针
   * @param valueNum 要插入或更新的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 如果键已存在，则更新对应的值；如果键不存在，则插入新的键值对
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see InsertOrAssign 用于同步插入或更新操作
   */
  void InsertOrAssignAsync(void *values, Extent valueNum, aclrtStream stream);

  /**
   * @brief 异步条件 assign-only：stencil 谓词为真且 key 已存在时更新 value，不存在不插入
   *
   * @tparam StencilT stencil数组的元素类型，与 values 一一对应
   * @tparam Predicate 仿函数类型，需提供 COLLECTION_HOST_DEVICE bool operator()(StencilT) const
   *
   * @param values Device侧指向键值对数组的指针
   * @param stencil Device侧指向stencil数组的指针
   * @param valueNum 键值对数量
   * @param stream ACL流
   */
  template <typename StencilT, typename Predicate>
  void AssignIfAsync(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream);


  /**
   * @brief 同步条件插入键值对到map中
   * 
   * @tparam StencilT stencil数组的元素类型。stencil数组与values数组一一对应，每个元素作为对应键值对的谓词判断输入，由仿函数根据stencil[i]的值决定是否插入values[i]
   * @tparam Predicate 仿函数类型，需提供 operator()(StencilT) const 重载，返回 bool；返回 true 表示执行插入，返回 false 表示跳过。仿函数需使用 COLLECTION_HOST_DEVICE 宏修饰，以确保在Host和Device侧均可调用
   * 
   * @param values Device侧指向键值对数组的指针
   * @param stencil Device侧指向stencil数组的指针，与values一一对应，用于谓词判断
   * @param valueNum 要插入的键值对数量，必须与values和stencil指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @return 插入失败的键值对数量（仅统计pred(stencil[i])为true且插入失败的元素）
   * 
   * @note 这是一个同步操作，会阻塞直到插入完成
   * @note 只有pred(stencil[i])为true时，才会尝试插入values[i]
   * 
   * @warning valueNum 参数必须与 values 和 stencil 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 values.size() 作为 valueNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * @warning stencil中的元素类型必须与StencilT一致
   * 
   * @see InsertIfAsync 用于异步条件插入操作
   */
  template <typename StencilT, typename Predicate>
  SizeType InsertIf(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream);

  /**
   * @brief 异步条件插入键值对到map中
   * 
   * @tparam StencilT stencil数组的元素类型。stencil数组与values数组一一对应，每个元素作为对应键值对的谓词判断输入，由仿函数根据stencil[i]的值决定是否插入values[i]
   * @tparam Predicate 仿函数类型，需提供 operator()(StencilT) const 重载，返回 bool；返回 true 表示执行插入，返回 false 表示跳过。仿函数需使用 COLLECTION_HOST_DEVICE 宏修饰，以确保在Host和Device侧均可调用
   * 
   * @param values Device侧指向键值对数组的指针
   * @param stencil Device侧指向stencil数组的指针，与values一一对应，用于谓词判断
   * @param valueNum 要插入的键值对数量，必须与values和stencil指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 只有pred(stencil[i])为true时，才会尝试插入values[i]
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see InsertIf 用于同步条件插入操作
   */
  template <typename StencilT, typename Predicate>
  void InsertIfAsync(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream);

  /**
   * @brief 同步插入并查找键值对
   * 
   * @param values Device侧指向键值对数组的指针
   * @param outputFind Device侧指向输出查找结果数组的指针，元素类型为Value。若键已存在则返回已存在的值，若键不存在则返回新插入的值，若键为空键或容量已满则返回空值（emptyValue）
   * @param outputInsert Device侧指向输出插入标志数组的指针，元素类型为unsigned char。非0表示新插入成功，0表示键已存在或插入失败
   * @param valueNum 要插入并查找的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到插入并查找完成
   * 
   * @warning valueNum 参数必须与 values 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 values.size() 作为 valueNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see InsertAndFindAsync 用于异步插入并查找操作
   */
  void InsertAndFind(void* values, void* outputFind, void* outputInsert, Extent valueNum, aclrtStream stream);

  /**
   * @brief 异步插入并查找键值对
   * 
   * @param values Device侧指向键值对数组的指针
   * @param outputFind Device侧指向输出查找结果数组的指针，元素类型为Value。若键已存在则返回已存在的值，若键不存在则返回新插入的值，若键为空键或容量已满则返回空值（emptyValue）
   * @param outputInsert Device侧指向输出插入标志数组的指针，元素类型为unsigned char。非0表示新插入成功，0表示键已存在或插入失败
   * @param valueNum 要插入并查找的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see InsertAndFind 用于同步插入并查找操作
   */
  void InsertAndFindAsync(void* values, void* outputFind, void* outputInsert, Extent valueNum, aclrtStream stream);

  /**
   * @brief 同步删除指定键的键值对
   * 
   * @param keys Device侧指向键数组的指针
   * @param keyNum 要删除的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @return 删除失败的键数量（即不存在的键的数量）
   * 
   * @note 这是一个同步操作，会阻塞直到删除完成
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see EraseAsync 用于异步删除操作
   */
  SizeType Erase(void *keys, Extent keyNum, aclrtStream stream);// void* 由于无法知道用户传递的到底是什么类型，有可能用户传入与要求输入的类型不一致，后续建议封装一个自己的vector作为输入

  /**
   * @brief 异步删除指定键的键值对
   * 
   * @param keys Device侧指向键数组的指针
   * @param keyNum 要删除的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see Erase 用于同步删除操作
   */
  void EraseAsync(void *keys, Extent keyNum, aclrtStream stream);

  /**
   * @brief 同步清空 static_map 中所有的元素
   * 
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到清空完成
   * 
   * @see ClearAsync 用于异步清空操作
   */
  void Clear(aclrtStream stream);

  /**
   * @brief 异步清空 static_map 中所有的元素
   * 
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see Clear 用于同步清空操作
   */
  void ClearAsync(aclrtStream stream) noexcept;

  /**
   * @brief 同步查找键对应的值
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到查找完成
   * @note 如果键不存在，返回空值（emptyValue）
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see FindAsync 用于异步查找操作
   */
  void Find(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // void* 获取用户传入的key，KeyNum表示传入要查找的key的个数

    /**
   * @brief 异步查找键对应的值
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 如果键不存在，返回空值（emptyValue）
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see Find 用于同步查找操作
   */
  void FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);

  /**
   * @brief 同步条件查找键对应的值
   * 
   * @tparam StencilT stencil数组的元素类型。stencil数组与keys数组一一对应，每个元素作为对应键的谓词判断输入，由仿函数根据stencil[i]的值决定是否查找keys[i]
   * @tparam Predicate 仿函数类型，需提供 operator()(StencilT) const 重载，返回 bool；返回 true 表示执行查找，返回 false 表示跳过。仿函数需使用 COLLECTION_HOST_DEVICE 宏修饰，以确保在Host和Device侧均可调用
   * 
   * @param keys Device侧指向键数组的指针
   * @param stencil Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys和stencil指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到查找完成
   * @note 只有pred(stencil[i])为true时，才会查找keys[i]；否则outputValues[i]写入空值（emptyValue）
   * @note 如果键不存在，返回空值（emptyValue）
   * 
   * @warning keyNum 参数必须与 keys 和 stencil 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * @warning stencil中的元素类型必须与StencilT一致
   * 
   * @see FindIfAsync 用于异步条件查找操作
   */
  template <typename StencilT, typename Predicate>
  void FindIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

  /**
   * @brief 异步条件查找键对应的值
   * 
   * @tparam StencilT stencil数组的元素类型。stencil数组与keys数组一一对应，每个元素作为对应键的谓词判断输入，由仿函数根据stencil[i]的值决定是否查找keys[i]
   * @tparam Predicate 仿函数类型，需提供 operator()(StencilT) const 重载，返回 bool；返回 true 表示执行查找，返回 false 表示跳过。仿函数需使用 COLLECTION_HOST_DEVICE 宏修饰，以确保在Host和Device侧均可调用
   * 
   * @param keys Device侧指向键数组的指针
   * @param stencil Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys和stencil指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 只有pred(stencil[i])为true时，才会查找keys[i]；否则outputValues[i]写入空值（emptyValue）
   * @note 如果键不存在，返回空值（emptyValue）
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * @warning stencil中的元素类型必须与StencilT一致
   * 
   * @see FindIf 用于同步条件查找操作
   */
  template <typename StencilT, typename Predicate>
  void FindIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

    /**
   * @brief 同步检查指定键是否存在
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到检查完成
   * @note 无返回值，检查结果通过 outputValues 输出（bool类型）
   * @note 输出值为 true 表示键存在，false 表示键不存在
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see ContainsAsync 用于异步检查操作
   */
  void Contains(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // outputValues中的元素为bool类型

    /**
   * @brief 异步检查指定键是否存在
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 无返回值，检查结果通过 outputValues 输出（bool类型）
   * @note 输出值为 true 表示键存在，false 表示键不存在
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see Contains 用于同步检查操作
   */
  void ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // outputValues中的元素为bool类型

  /**
   * @brief 同步条件检查指定键是否存在
   * 
   * @tparam StencilT stencil数组的元素类型。stencil数组与keys数组一一对应，每个元素作为对应键的谓词判断输入，由仿函数根据stencil[i]的值决定是否检查keys[i]
   * @tparam Predicate 仿函数类型，需提供 operator()(StencilT) const 重载，返回 bool；返回 true 表示执行检查，返回 false 表示跳过。仿函数需使用 COLLECTION_HOST_DEVICE 宏修饰，以确保在Host和Device侧均可调用
   * 
   * @param keys Device侧指向键数组的指针
   * @param stencil Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys和stencil指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到检查完成
   * @note 只有pred(stencil[i])为true时，才会检查keys[i]；否则outputValues[i]写入false
   * @note 输出值为 true 表示键存在，false 表示键不存在或谓词为false
   * 
   * @warning keyNum 参数必须与 keys 和 stencil 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * @warning stencil中的元素类型必须与StencilT一致
   * 
   * @see ContainsIfAsync 用于异步条件检查操作
   */
  template <typename StencilT, typename Predicate>
  void ContainsIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

  /**
   * @brief 异步条件检查指定键是否存在
   * 
   * @tparam StencilT stencil数组的元素类型。stencil数组与keys数组一一对应，每个元素作为对应键的谓词判断输入，由仿函数根据stencil[i]的值决定是否检查keys[i]
   * @tparam Predicate 仿函数类型，需提供 operator()(StencilT) const 重载，返回 bool；返回 true 表示执行检查，返回 false 表示跳过。仿函数需使用 COLLECTION_HOST_DEVICE 宏修饰，以确保在Host和Device侧均可调用
   * 
   * @param keys Device侧指向键数组的指针
   * @param stencil Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys和stencil指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 只有pred(stencil[i])为true时，才会检查keys[i]；否则outputValues[i]写入false
   * @note 输出值为 true 表示键存在，false 表示键不存在或谓词为false
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * @warning stencil中的元素类型必须与StencilT一致
   * 
   * @see ContainsIf 用于同步条件检查操作
   */
  template <typename StencilT, typename Predicate>
  void ContainsIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

  /**
   * @brief 同步遍历哈希表中与指定键匹配的槽位，对每个匹配的槽位执行回调函数
   * 
   * @tparam CallbackOp 仿函数类型，要求如下：
   *   - 提供 COLLECTION_SIMT_DEVICE void operator()(Pair<Key, T>) const 重载，接收匹配的槽位作为参数
   *   - 提供 COLLECTION_SIMT_DEVICE 构造函数接受 __gm__ uint8_t* 参数，用于接收 callbackArgs 指针并在内部 reinterpret_cast 为实际类型
   *   - operator() 中可使用 AscendC::Simt::AtomicAdd 等设备端原子操作访问 callbackArgs 指向的设备内存
   * 
   * @param keys Device侧指向键数组的指针
   * @param keyNum 要遍历的键数量，必须与keys指向的数组实际大小一致
   * @param callbackArgs Device侧指向用户自定义数据的指针，以 void* 类型擦除传入kernel，由 CallbackOp 构造函数 reinterpret_cast 为实际类型使用
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到遍历完成
   * @note 对于每个key，如果在哈希表中找到匹配的槽位，则调用回调函数；遇到空槽位则停止探测
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 传入的指针中数据类型需要和map中的相对应
   * @warning 回调函数中不应修改哈希表的状态，否则可能导致未定义行为
   * 
   * @see ForEachAsync 用于异步遍历操作
   */
  template <typename CallbackOp>
  void ForEach(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream);

  /**
   * @brief 异步遍历哈希表中与指定键匹配的槽位，对每个匹配的槽位执行回调函数
   * 
   * @tparam CallbackOp 仿函数类型，要求如下：
   *   - 提供 COLLECTION_SIMT_DEVICE void operator()(Pair<Key, T>) const 重载，接收匹配的槽位作为参数
   *   - 提供 COLLECTION_HOST_DEVICE 构造函数接受 __gm__ uint8_t* 参数，用于接收 callbackArgs 指针并在内部 reinterpret_cast 为实际类型
   *   - operator() 中可使用 AscendC::Simt::AtomicAdd 等设备端原子操作访问 callbackArgs 指向的设备内存
   * 
   * @param keys Device侧指向键数组的指针
   * @param keyNum 要遍历的键数量，必须与keys指向的数组实际大小一致
   * @param callbackArgs Device侧指向用户自定义数据的指针，以 void* 类型擦除传入kernel，由 CallbackOp 构造函数 reinterpret_cast 为实际类型使用
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 对于每个key，如果在哈希表中找到匹配的槽位，则调用回调函数；遇到空槽位则停止探测
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * @warning 回调函数中不应修改哈希表的状态，否则可能导致未定义行为
   * 
   * @see ForEach 用于同步遍历操作
   */
  template <typename CallbackOp>
  void ForEachAsync(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream);

  /**
   * @brief 同步统计指定键在map中存在的数量
   * 
   * @param keys Device侧指向键数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @return 存在的键数量
   * 
   * @note 这是一个同步操作，会阻塞直到统计完成
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   */
  SizeType Count(void *keys, Extent keyNum, aclrtStream stream);

  /**
   * @brief 获取map的实际容量
   * 
   * @return 实际容量
   * 
   * @note 实际容量会向上取整到BuketSize的倍数
   */
  constexpr auto Capacity() const noexcept;

  /**
   * @brief 获取当前使用的键比较器
   * 
   * @return 当前键比较器的实例
   * 
   * @note 该方法返回构造函数中的指定的键比较器
   */
  constexpr KeyEqual GetKeyEqual() const noexcept;

  /**
   * @brief 获取Device侧map内部数据的指针
   * 
   * @return 指向Device侧map内部数据的指针
   * 
   * @note 该方法返回Device侧map内部存储的原始指针
   * 
   * @warning 返回Device侧的内存指针，不能在host侧直接访问
   */
  ValueType* Data() const;

 private:
  std::unique_ptr<ImplType> impl_;
  Key emptyKey_;
  T emptyValue_;
};
}

#include "detail/static_map/static_map.inl"