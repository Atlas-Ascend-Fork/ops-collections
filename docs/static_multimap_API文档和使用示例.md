# StaticMultimap API 与使用示例

`aclco::StaticMultimap<Key, Value>` 是固定容量的 Device 多值映射容器。每个键值对独立占槽，重复键和完全相同的键值对均保留。Key 和 Value 支持 `int32_t` 与 `int64_t`，面向支持 SIMT 的 Ascend 950，使用 C++17 和 CANN 编译器构建。

## 生命周期与调用约定

```cpp
#include "static_multimap.h"

// 调用方先完成 ACL 初始化、设备选择与 stream 创建。
aclco::StaticMultimap<int32_t, int32_t> map(
    aclco::Extent<std::size_t>(1024), INT32_MIN, INT32_MIN, stream);
// devicePairs 指向已拷贝到 Device 的 aclco::Pair<int32_t, int32_t>[n]。
auto failed = map.Insert(devicePairs, aclco::Extent<std::size_t>(n), stream);
auto size = map.Size(stream);
// deviceKeys 指向 Device 的 int32_t[q]，deviceValues 指向 int32_t[q]。
map.Find(deviceKeys, deviceValues, aclco::Extent<std::size_t>(q), stream);
map.Clear(stream);
```

构造容量必须大于 0。`Capacity()` 返回实际容量，`Size(stream)` 返回全部元素数量，`Clear(stream)` 清空元素但保留容量。析构释放所属 Device 内存，因此必须在 ACL 设备重置及终结之前析构。禁止复制；移动转移所有权，移动后原对象的 Capacity 为 0，其他方法报告 `std::logic_error`。

输入、输出和 stencil 都指向连续 Device 内存；调用方保证实际类型、分配长度和生命周期正确。非零输入与输出不得覆盖。所有批量方法同步完成本次设备工作；不同 Host 线程或 ACL 流调用同一对象时，调用方必须串行化。默认 `stream=nullptr` 使用 ACL 默认流。

## 接口

下表的 `n` 和 `outputCapacity` 类型均为 `aclco::Extent<std::size_t>`，`stream` 均可省略。数据指针在公开接口中使用 `void*`；stencil 保留 `Stencil*` 类型。

| 方法 | 行为 |
| --- | --- |
| `Insert(pairs, n, stream)` | 插入 Pair 数组并返回失败数。容量不足时接纳输入索引前缀，已有元素不变。非零长度空输入返回 n。 |
| `InsertIf<Stencil, Predicate>(pairs, stencil, n, stream)` | 默认构造设备谓词，仅插入选中元素。容量不足时接纳选中元素的输入顺序前缀，返回选中但未插入的数量。 |
| `Contains(keys, output, n, stream)` | 输出 n 个 bool，命中为 true。 |
| `ContainsIf<Stencil, Predicate>(keys, stencil, output, n, stream)` | 谓词不成立时写 false，否则查询。 |
| `Find(keys, values, n, stream)` | 命中时写该键对应的最小 Value，未命中写 emptyValue。 |
| `FindIf<Stencil, Predicate>(keys, stencil, values, n, stream)` | 谓词不成立时写 emptyValue，否则按 Find 规则查询。 |
| `Count(keys, n, stream)` | 返回 uint64_t 标量总匹配数。重复查询重复计数。 |
| `Retrieve(keys, n, probeOut, valueOut, outputCapacity, stream)` | 输出全部匹配并返回数量。按查询索引排列，每个查询的 Value 升序排列；probeOut 写对应查询 Key。 |
| `RetrieveAll(keyOut, valueOut, outputCapacity, stream)` | 输出全部元素并返回数量，保留重复键值对；输出顺序不属于接口契约。 |

零长度调用允许空数据指针，检索零匹配也允许空输出。查询 emptyKey 的结果始终为未命中。

emptyKey 和 emptyValue 为保留值，不能出现在选中的插入元素中。插入在修改容器前检查全部选中输入；若存在保留值，抛出 `std::invalid_argument` 并保持原状态。未选中的元素不校验。非零长度 InsertIf 的空输入或空 stencil 抛出 `std::invalid_argument`；其他查询的非零长度空输入或空输出也报告参数异常。

Retrieve 和 RetrieveAll 在写出前核验匹配数量。容量不足抛出 `std::length_error`，不截断，也不改写输出。声明长度的字节数溢出、计数上界溢出报告 `std::overflow_error`。ACL 分配和执行失败报告含错误码的 `std::runtime_error`。发生设备执行错误后不承诺事务回滚，应在处理设备错误后重建对象。

## 实现与性能范围

容量不超过 `UINT32_MAX` 时，容器按接纳顺序稠密保存 Pair，哈希表的每个槽位保存 32 位 Key 或键摘要及 32 位元素索引。位于 `[0, capacity)` 的非负整数键先通过容量范围内的可逆置换分散到唯一起始槽；置换在最小的 2 的幂域上使用奇数乘数和 cycle walking，既避免起始槽碰撞，也避免连续键在同一 Cache Line 内争用原子操作。其他 I32 键使用整数乘法哈希，其他 I64 键使用 MurmurHash3 finalizer；乘法范围缩减将哈希映射到实际容量。有界线性探测保留所有重复 Pair。插入连续写入 Pair，以一次 64 位 CAS 发布槽位索引；插入线程不读取其他线程的 Pair，查询在插入核完成之后执行。I64 查询在摘要匹配后比较完整 Key，摘要碰撞不会当成 Key 相等。

该表示的静态 Device 存储为 `capacity * (sizeof(Pair<Key,Value>) + 8) + 4` 字节，不包括分配器开销；末尾 4 字节是插入调用复用的状态字。超过 32 位索引范围的容量使用直接键值槽位和 64 位探测下标，避免索引截断。Clear 清空槽位索引并重置元素数量，稠密区域后续按新输入覆盖，不保留历史有效元素。

Count 使用 uint64_t 局部计数和多层分块前缀和，每个块内由一个 warp 协作扫描，分块间通过后续核函数累加偏移。Retrieve 对每块256个查询的总计数执行前缀和，输出核使用warp扫描重算块内偏移，保留查询顺序和段内原地堆排序。RetrieveAll 只保证返回完整键值对多重集，不保证输出顺序。常规稠密模式按接纳位置直接导出，因此当前实现会保留跨批次的接纳顺序；这属于实现行为，调用方不能依赖。大容量直接槽位模式先压缩有效槽并排序，以保证任务要求的重复执行可复现。

Retrieve 在计数阶段为单个匹配直接保存 Value，为多个匹配保存首个匹配槽位。输出阶段直接写出单个匹配；多个匹配从首个槽位开始，并在写满该查询的匹配数量后停止。该辅助数组使用带有明确活动成员的 `RetrievalMatch<Value>` 联合体，计数为1时读写Value成员，否则读写槽位成员；不依赖size_t与Value之间的类型双关或小端序。仍保留每查询计数和匹配缓存，块级前缀和未消除这两个O(NumInputs)数组；调用期间容器不可并发修改。

Count 和 Retrieve 复用对象内的查询缓冲及 Prefix workspace，仅在首次使用或容量不足时分配。Clear 保留缓存，析构时释放；同一对象的调用仍须外部串行化。

接近满表、高冲突或高重复键会增加线性探测成本；单个查询返回大量匹配也会增加成本。稠密模式的 RetrieveAll 时延与输入排列无关，均为一次连续导出；直接槽位模式仍包含压缩和排序成本。校验、同步和工作空间分配均在对应接口内部，计时不能排除这些成本。功能测试通过不能代替官方 32 个大规模性能用例的逐例验收。

## 测试

功能测试位于 `tests/static_multimap`，性能测试位于 `tests/performance/static_multimap`，公共辅助函数位于 `tests/common/static_multimap_test_common.h`。测试保留任务附件的输入、断言和性能计时范围。

`contract_test.cpp` 覆盖条件插入前缀、错误路径不修改输出、64 位扫描、整数边界及资源移动。`bulk_sequence_test.cpp` 检查不同容量、重复次数和命中率下的混合操作。`direct_storage_test.cpp` 在小容量下强制使用直接存储，覆盖公共批量操作、满表、跨块查询、清空后重插入及移动；生产构造函数仍按容量选择存储模式。

按仓库 CMake 配置构建后，可执行：

```bash
ctest --test-dir <build-directory> -R '^collection_tests_static_multimap_' --output-on-failure -j1
```

性能测试使用仓库现有框架和官方附件规模；两个时间字段都来自同步调用的 Host 墙钟计时，不是独立测得的核函数执行时间。设备性能分析使用官方 msopprof，并与端到端耗时分别解释。

启用 `BUILD_PERFORMANCE` 后，可运行查询缓冲复用基准，检查结果一致性、显式设备分配/释放次数及 Host 耗时：

```bash
cmake --build <build-directory> --target multimap_query_scratch
<build-directory>/tests/multimap_query_scratch 1000 0 --expect-reuse
```
