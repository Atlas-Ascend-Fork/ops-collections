# BloomFilter API 文档和使用示例

## 1. 概述

`aclco::BloomFilter` 是固定容量、设备内存驻留的 Sectorized Bloom Filter。
每个 key 只访问一个 256 bit block；默认策略使用 `XXHash_64`、8 个
`uint32_t` word 和 8 个指纹位。容器支持 `int32_t`、`uint32_t`、
`int64_t`、`uint64_t` 和 `float` key。

Bloom Filter 不会对已成功 Add 且满足流依赖的 key 产生假阴性，但查询未插入
key 时允许出现假阳性。

## 2. 头文件与默认类型

```cpp
#include "bloom_filter.h"

using Filter = aclco::BloomFilter<std::uint32_t>;
```

为兼容既有调用方，也可以使用以下两个策略名称：

```cpp
using DefaultPolicy =
  aclco::DefaultFilterPolicy<aclco::xxhash_64<std::uint32_t>,
                             std::uint32_t,
                             8>;
using ArrowPolicy = aclco::ArrowFilterPolicy<std::uint32_t>;
```

`DefaultFilterPolicy` 和 `ArrowFilterPolicy` 都是现有 `BloomFilterPolicy`
的同类型别名，不包含额外状态或独立运行路径，因此不会增加运行时开销。

默认一个 block 为 32 byte，因此由目标字节数计算 block 数：

```cpp
std::size_t const filterBytes = 32ull * 1024 * 1024;
std::size_t const numBlocks = filterBytes / 32;
```

## 3. 公开接口

### 3.1 构造、析构与所有权

```cpp
explicit BloomFilter(
    Extent numBlocks,
    Policy const& policy = Policy{},
    Allocator const& allocator = Allocator{},
    aclrtStream stream = nullptr);

explicit BloomFilter(
    Extent numBlocks,
    Policy const& policy,
    aclrtStream stream);
```

构造函数申请设备内存并在指定 stream 上清零，返回前同步该 stream。拷贝被删除，
移动构造和移动赋值转移所有权。析构只释放存储，不隐式同步尚未完成的异步任务；
销毁前必须由调用方保证相关 stream 已完成。

三参数重载使用默认构造的 allocator，并委托给完整构造函数；它只提供调用兼容性，
不增加设备操作、同步或额外状态。

### 3.2 Clear

```cpp
void Clear(aclrtStream stream = nullptr);
void ClearAsync(aclrtStream stream = nullptr);
```

`Clear` 在返回前同步；`ClearAsync` 只把异步清零排入 stream。

### 3.3 Add

```cpp
void Add(void* keys, Extent keyNum, aclrtStream stream = nullptr);
void AddAsync(void* keys, Extent keyNum, aclrtStream stream = nullptr);
```

`keys` 必须是含 `keyNum` 个 `Key` 的设备地址。`keyNum==0` 时允许 `keys==nullptr`
且不启动 kernel。默认 H8V1 容器对足够大的稠密批次使用两阶段有界聚合快路：

1. Route 阶段只计算一次 hash，通过每 block 的原子计数器取得唯一槽位并保存
   `lowerHash`；超过槽位容量的 key 立即使用四次 U64 `asc_atomic_or`；
2. Apply 阶段在同一 stream 中随后执行，每个活跃 block 由唯一线程聚合已保留槽位，
   对非空或有溢出的 block 通过 `asc_ldcg/asc_stcg` 完成 32 字节 OR 写回，并
   复位计数器；容器能证明为空且该 block 未溢出时省略旧值读取。

kernel 边界保证 Apply 能看到 Route 的槽位写入和溢出原子结果，因此热点、重复 key、
连续多批 Add 及非空过滤器都保持精确位并集，不需要全量二次 hash/repair。工作区在
首次适用 Add 时通过容器 allocator 懒分配并复用；会根据批量密度和可用显存选择容量，
最大包含 1 GiB 槽位和 256 MiB 计数器。`SizeBytes()` 仍只表示公开 BloomFilter
位数组，不包含该内部工作区。小批量、稀疏批量、工作区申请失败或自定义 allocator
不满足 32 字节 block 对齐时，安全回退为每 key 四次 U64 packed atomic；只满足
4 字节对齐时使用 U32 原子。所有路径生成完全相同的八个 U32 word 布局。

批次至少包含 `max(16384, ceil(numBlocks/2))` 个 key 时才尝试该快路，避免稀疏
批次为 Apply 支付全 block counter 扫描。首次适用的 `AddAsync` 会在 host 侧查询
可用 HBM 并调用 allocator，因此首次调用可能包含分配和 counter 初始化开销；稳态
性能用例在计时前预热。`Clear/ClearAsync` 只清公开位数组，不回收已分配工作区；
后续 Add 继续复用，move 转移其所有权，析构时与公开位数组分别释放。工作区容量由
首次成功分配时的批量密度决定，后续不会扩容；首次申请失败后该对象保持原子回退而
不反复重试。这些策略不影响正确性，但较小的初始容量可能让后续大批量更多地走
overflow 原子回退。

### 3.4 AddIf

```cpp
void AddIf(
    void* keys, void* stencil, Extent keyNum, aclrtStream stream = nullptr);
void AddIfAsync(
    void* keys, void* stencil, Extent keyNum, aclrtStream stream = nullptr);
```

`stencil` 是含 `keyNum` 个 `uint8_t` 的设备地址。仅当 `stencil[i] != 0`
时添加 `keys[i]`。同步接口返回前等待 stream；异步接口只下发独立条件 kernel。
该实现不向普通 Add kernel 增加 stencil 分支，因此不改变普通 Add 的指令路径和性能。

### 3.5 Contains

```cpp
void Contains(
    void* keys, ExtentType keyNum, void* outputValues, aclrtStream stream = nullptr) const;
void ContainsAsync(
    void* keys, ExtentType keyNum, void* outputValues, aclrtStream stream = nullptr) const;
void Contains(
    void* keys, void* outputValues, ExtentType keyNum, aclrtStream stream = nullptr) const;
void ContainsAsync(
    void* keys, void* outputValues, ExtentType keyNum, aclrtStream stream = nullptr) const;
```

`outputValues` 是含 `keyNum` 个 `uint8_t` 的设备地址，结果严格写为 `0` 或 `1`。
前两个重载采用与 cuCollections 范围参数对应的输入地址、元素数、输出地址顺序；
后两个重载兼容验收调用以及仓内其他集合容器采用的输入地址、输出地址、元素数顺序。
兼容重载只在 host 侧转发到前两个重载，不增加 kernel、同步或设备计算。

### 3.6 ContainsIf

```cpp
void ContainsIf(
    void* keys,
    void* stencil,
    void* outputValues,
    Extent keyNum,
    aclrtStream stream = nullptr) const;
void ContainsIfAsync(
    void* keys,
    void* stencil,
    void* outputValues,
    Extent keyNum,
    aclrtStream stream = nullptr) const;
```

`stencil` 和 `outputValues` 均按 `uint8_t` 数组解释。当 `stencil[i] != 0`
时查询 `keys[i]` 并写入 `0` 或 `1`；跳过的位置显式写入 `0`。条件查询使用
独立 kernel，不改变普通 Contains 的设备代码。

### 3.7 Merge 与 Intersect

```cpp
void Merge(BloomFilter const& other, aclrtStream stream = nullptr);
void MergeAsync(BloomFilter const& other, aclrtStream stream = nullptr);
void Intersect(BloomFilter const& other, aclrtStream stream = nullptr);
void IntersectAsync(BloomFilter const& other, aclrtStream stream = nullptr);
```

两者分别执行原地逐 word OR 和 AND，不修改 `other`。两个过滤器必须具有相同
block 数和 hash seed，否则在 kernel 下发前抛出 `std::invalid_argument`。

### 3.8 观察接口与设备引用

```cpp
WordType* Data() noexcept;
WordType const* Data() const noexcept;
Extent BlockExtent() const noexcept;
SizeType NumWords() const noexcept;
std::size_t SizeBytes() const noexcept;
Policy const& GetPolicy() const noexcept;
```

`BloomFilterRef` 是仅供 AICore 设备侧使用的非拥有型引用。自定义 kernel 应由
host 传入 `Data()`、block 数和 Policy 所需的标量状态，并在 kernel 内从
`__gm__ WordType*` 构造 `BloomFilterRef`；host 侧不构造或传递 Ref 对象。

调用可变重载 `WordType* Data()` 不会永久禁用有界聚合快路；Apply 会读取并 OR
已有 block 内容。由于容器无法再证明外部没有写入，可变 `Data()` 会永久关闭
“已知空时省略旧 block 读取”这一项优化。调用方仍必须遵守流顺序：外部 kernel
或拷贝对该地址的写入必须在 Add、Clear、Merge、Intersect 或 Contains 前通过
同 stream 或 event 建立依赖。

## 4. 完整示例

以下代码省略 ACL 初始化和错误处理，`deviceKeys`、`deviceQueries` 和
`deviceOutput` 均为设备地址：

```cpp
#include "bloom_filter.h"

using Key = std::uint32_t;
using Filter = aclco::BloomFilter<Key>;

aclrtStream stream = nullptr;
aclrtCreateStream(&stream);

std::size_t const numBlocks = (32ull * 1024 * 1024) / 32;
Filter filter(aclco::Extent<std::size_t>{numBlocks},
              Filter::PolicyType{},
              Filter::AllocatorType{},
              stream);

filter.Add(deviceKeys, aclco::Extent<std::size_t>{numKeys}, stream);
filter.Contains(deviceQueries,
                aclco::Extent<std::size_t>{numQueries},
                deviceOutput,
                stream);

Filter other(aclco::Extent<std::size_t>{numBlocks},
             Filter::PolicyType{},
             Filter::AllocatorType{},
             stream);
other.Add(deviceOtherKeys, aclco::Extent<std::size_t>{numOtherKeys}, stream);
filter.Merge(other, stream);

aclrtSynchronizeStream(stream);
aclrtDestroyStream(stream);
```

## 5. Policy 与限制

本期 host bulk kernel 支持：

- `Key`：`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`float`；
- `Hasher`：`aclco::xxhash_64<Key>`，可配置 64 位 seed；
- `Word`：`uint32_t`；
- `Extent::ValueType`：无符号整数类型；
- `WordsPerBlock`：非零 2 的幂；
- `PatternBits`：不大于 64，且必须被 `WordsPerBlock` 整除；
- Add/Contains 布局乘积必须整除 `WordsPerBlock`；
- `0 < numBlocks <= UINT32_MAX`，且 word 数、字节数不得溢出宿主类型。

默认 Add 布局为 H8V1，默认 Contains 布局为 H1V8。任务书 Contains 性能用例需显式
实例化 H8V1 Policy。布局用于描述策略语义和选择可用快路；内部 kernel 可以采用
位级等价的独立 lane 调度、有界 block 聚合和相邻 word 配对原子，不承诺 H8V1
必然使用协作 shuffle，也不改变位数组语义。

同一过滤器上的 Add、AddIf、Clear、Contains、ContainsIf、Merge 和 Intersect
不应无依赖地跨 stream 并发。通过 `Data()` 或 `BloomFilterRef` 启动的自定义写
kernel 也必须与批量 Add 建立顺序，不能同时混用重叠的普通写、U32/U64 原子写。
跨 stream 使用时，应通过 ACL event 建立先写后读/写顺序。

## 6. 构建与测试

```bash
# 功能测试
bash scripts/build.sh -b
bash scripts/build.sh -r --test-name bloom_filter

# 性能测试（32/256/2048 MiB，I32/I64）
bash scripts/build.sh -p
bash scripts/build.sh -rp
```

构建前需设置与 Ascend 950PR&950DT 系列产品环境匹配的 `ASCEND_HOME_PATH`，并使用仓库指定的 CANN
版本以及 `ccec` 或毕昇 ASC 编译器。
