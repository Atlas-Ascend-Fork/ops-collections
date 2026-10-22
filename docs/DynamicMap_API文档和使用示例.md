# DynamicMap 容器 API 文档与使用示例

> DynamicMap 是 ops-collections 中的 C++ 容器类 `aclco::DynamicMap`，是一张容量随插入自动增长的并发哈希表。
> 本库为 **C++ 类接口**，非 aclnn 算子。

## 一、类型定义

```cpp
#include "dynamic_map.h"

using Key   = uint32_t;
using Value = uint32_t;
using ProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<Key>>;
using KeyEqual      = aclco::EqualTo<Key>;

using MyDynamicMap = aclco::DynamicMap<Key, Value, aclco::Extent<std::size_t>,
                                       KeyEqual, ProbingScheme, aclco::Storage<1>>;
```

**类型说明：**

| 模板参数 | 说明 |
| --- | --- |
| `Key` | 键类型，支持 `uint16_t`、`uint32_t`、`uint64_t`、`float` |
| `Value` | 值类型，支持同上类型 |
| `Extent` | 容量/数量类型，默认 `aclco::Extent<std::size_t>` |
| `KeyEqual` | 键比较器，默认 `aclco::EqualTo<Key>` |
| `ProbingScheme` | 探测策略，默认线性探测 `aclco::LinearProbing<aclco::murmurhash3_32<Key>>` |
| `Storage` | 存储策略（桶大小），默认 `aclco::Storage<5>` |

---

## 二、接口说明

### 2.1 构造 / 析构

**函数签名：**

```cpp
DynamicMap(Extent initialCapacity, Key emptyKey, T emptyValue, Key erasedKey,
           KeyEqual const& pred = {}, ProbingScheme const& ps = {},
           Storage storage = {}, aclrtStream stream = nullptr);
~DynamicMap();   // 自动释放全部子表
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
| --- | --- | --- | --- |
| initialCapacity | Extent | 输入 | 初始槽数（活跃子表初始容量，载满后自动追加更大子表）|
| emptyKey | Key | 输入 | 空槽标记键（保留值，不可作为有效 key 插入）|
| emptyValue | T | 输入 | 空槽标记值 |
| erasedKey | Key | 输入 | 删除墓碑标记键（保留值，须 ≠ emptyKey，启用 Erase）|
| pred / ps / storage | — | 输入 | 比较器 / 探测策略 / 存储策略，均有默认值 |
| stream | aclrtStream | 输入 | ACL 流 |

**功能说明：** 创建容量随插入自动增长的并发哈希表。占用率超过 `maxLoadFactor`（默认 0.60）时，自动追加一个容量翻倍的新子表。

### 2.2 Insert - 插入

```cpp
SizeType Insert(void* values, Extent valueNum, aclrtStream stream);                   // 返回成功插入数（kExactDedup）
SizeType Insert(void* values, Extent valueNum, aclrtStream stream, InsertMode mode);  // 指定插入模式
SizeType InsertAsync(void* values, Extent valueNum, aclrtStream stream);              // 异步，返回乐观插入计数
```

| 参数 | 类型 | 输入/输出 | 说明 |
| --- | --- | --- | --- |
| values | void* | 输入 | Device 侧 `Pair<Key,Value>` 数组指针 |
| valueNum | Extent | 输入 | 插入数量，须与数组实际大小一致 |
| stream | aclrtStream | 输入 | ACL 流 |
| mode | InsertMode | 输入 | 插入模式，默认 `kExactDedup`（见下表） |
| 返回值 | SizeType | 输出 | 成功插入的元素个数（重复 key 不重复计入）|

**功能说明：** 插入一批键值对，已存在的 key 不覆盖。容量不足时自动增长。

**插入模式 `InsertMode`：**

| 模式 | 适用场景 | 原理 |
| --- | --- | --- |
| `kExactDedup`（默认） | 通用插入：输入可能含重复 key，或对已增长（多子表）的容器重插旧 key | 容器增长出多个子表后，逐块对所有旧子表做去重扫描（提取 key → Contains 命中判定 → 仅插入未命中的元素），保证"重插已存在 key 返回 0、`Size()` 精确"。 |
| `kAppendUnique` | 调用方保证**本批及历史所有 key 全局唯一**的大规模/多批次 bulk 插入（如分批 800k 灌入唯一键的高吞吐场景） | 跳过增长后的跨子表去重扫描，每批直接追加到单一活跃子表；采用乐观计数——先按批量大小累加 `Size()`，不回读实际插入失败数，`Size()` 仅返回 Host 的乐观计数，不执行同步或计数修正，从而消除每批的 host↔device 同步停顿。**仅当 key 确实全局唯一时计数才精确**；若误用于含重复 key 的输入，重复 key 不会被去重、`Size()` 会偏大。 |

> `InsertAsync` 同样采用乐观计数（不逐批回读失败数），适合唯一 key 的高吞吐异步插入；`Size()` 直接返回 Host 计数，不会在读取时回读失败数或修正重复输入导致的高估。对含重复 key 或需即时精确计数的场景，请用默认的同步 `Insert`（`kExactDedup`）。

### 2.3 InsertOrAssign - 插入或赋值

```cpp
SizeType InsertOrAssign(void* values, Extent valueNum, aclrtStream stream);
```

**功能说明：** key 不存在则插入；已存在则更新其 value。参数同 Insert。

### 2.4 Find - 查找

```cpp
void Find(void* keys, void* outputValues, Extent keyNum, aclrtStream stream);
```

| 参数 | 类型 | 输入/输出 | 说明 |
| --- | --- | --- | --- |
| keys | void* | 输入 | Device 侧待查 key 数组 |
| outputValues | void* | 输出 | Device 侧 value 输出数组；命中写入对应 value，未命中写入 emptyValue |
| keyNum | Extent | 输入 | 查询数量 |

### 2.5 Contains - 是否包含

```cpp
void Contains(void* keys, void* outputValues, Extent keyNum, aclrtStream stream);
```

| 参数 | 类型 | 输入/输出 | 说明 |
| --- | --- | --- | --- |
| outputValues | void* | 输出 | Device 侧 `uint8` 数组；存在为 1，否则为 0 |

### 2.6 Erase - 删除

```cpp
SizeType Erase(void* keys, Extent keyNum, aclrtStream stream);       // 返回成功删除数
void     EraseAsync(void* keys, Extent keyNum, aclrtStream stream);  // 异步
```

**功能说明：** 同步 `Erase` 删除一批 key（墓碑删除，不破坏探测链），回读成功删除数并更新 Host 计数。删除不存在的 key 无副作用。须在构造时提供有效 `erasedKey`。`EraseAsync` 只下发设备删除，不更新 Host 计数；即使随后同步流，`Size()` 也不会自动反映该次异步删除。需要精确元素数量时使用同步 `Insert`（`kExactDedup`）和 `Erase`，避免混入未经统计的异步修改。

### 2.7 Reserve - 预留容量

```cpp
void Reserve(SizeType n, aclrtStream stream);
```

**功能说明：** 预先增长容量以容纳 `n` 个元素，避免插入过程中多次增长。

### 2.8 Clear / 容量查询

```cpp
void     Clear(aclrtStream stream);   // 清空所有元素（容量不变）
SizeType Size() const noexcept;       // Host 计数，无同步/回读；异步修改后可能不精确
SizeType Capacity() const noexcept;   // 所有子表容量之和
size_t   NumSubmaps() const noexcept; // 当前子表数量
```

---

## 三、数据类型与约束

- key / value 支持 `uint16` / `uint32` / `uint64` / `float32`，单元素大小 ≤ 8 字节。
- `emptyKey`、`erasedKey` 为保留标记值，不可作为有效 key 插入，且二者必须不同。
- 输入数组须位于 Device 侧；`valueNum`/`keyNum` 须与数组实际长度一致。
- `uint16` 的 key 取值空间只有 65536 个，去重后能存下的元素数受值域限制。

---

## 四、使用示例（设备缓冲区由调用方提供）

调用前，需初始化 ACL、设置设备并创建 `stream`；在 Device 上分配并填充 `dPairs`（`n` 个 `aclco::Pair<Key, Value>`）和 `dKeys`（`n` 个 `Key`），并为 `dValues`（`n` 个 `Value`）及 `dFlags`（`n` 个 `uint8_t`）分配输出缓冲区。以下示例演示容器 API 的调用顺序；实际程序还应检查 ACL 返回值，并由调用方释放这些缓冲区、销毁流和结束 ACL。

```cpp
#include "dynamic_map.h"
#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <limits>

using Key = std::uint32_t;
using Value = std::uint32_t;
using DMap = aclco::DynamicMap<Key, Value, aclco::Extent<std::size_t>,
                               aclco::EqualTo<Key>,
                               aclco::LinearProbing<aclco::murmurhash3_32<Key>>,
                               aclco::Storage<1>>;

void RunDynamicMapExample(void* dPairs, void* dKeys, void* dValues, void* dFlags,
                          std::size_t n, aclrtStream stream)
{
    Key emptyKey = std::numeric_limits<Key>::max();
    Value emptyValue = std::numeric_limits<Value>::max();
    Key erasedKey = emptyKey - 1;

    // 构造：初始容量 40,000,000，自动增长
    DMap map(aclco::Extent<std::size_t>(40000000), emptyKey, emptyValue, erasedKey, {}, {}, {}, stream);

    DMap::SizeType inserted = map.Insert(dPairs, aclco::Extent<std::size_t>(n), stream);
    aclrtSynchronizeStream(stream);

    // 查找（dKeys → dValues）
    map.Find(dKeys, dValues, aclco::Extent<std::size_t>(n), stream);

    // 是否包含（dKeys → dFlags）
    map.Contains(dKeys, dFlags, aclco::Extent<std::size_t>(n), stream);

    DMap::SizeType erased = map.Erase(dKeys, aclco::Extent<std::size_t>(n), stream);
    aclrtSynchronizeStream(stream);
    (void)inserted;
    (void)erased;
}
```
