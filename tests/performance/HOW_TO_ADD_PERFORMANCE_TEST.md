# 如何添加性能测试样例

本文档详细说明如何在 `tests/performance/static_map` 目录下添加新的性能测试样例。

---

## 目录

1. [测试框架概述](#测试框架概述)
2. [添加测试样例的步骤](#添加测试样例的步骤)
3. [完整示例](#完整示例)
4. [常用工具函数](#常用工具函数)
5. [测试参数说明](#测试参数说明)
6. [常见问题](#常见问题)

---

##  测试框架概述

性能测试框架基于以下核心组件：

### 核心组件

- **`TestResult`**: 测试结果结构体，包含CPU时间、设备时间和迭代次数
- **`TestStatistics`**: 测试统计结果，包含均值、标准差等统计信息
- **`PerformanceTestsRegister`**: 测试注册器，负责管理测试函数和参数
- **`PerformanceTestsRegisterManager`**: 测试管理器，负责运行所有注册的测试

### 自动统计特性

框架会自动执行以下操作：
- 多次迭代测试（最多50次）
- 计算CPU时间和设备时间的均值和标准差
- 当根据相对标准差（RSD）判断结果稳定时自动停止迭代
- 输出详细的测试统计结果

### 测试执行流程

1. 框架遍历测试参数组合
2. 对于每个参数组合，调用 `SETUP_FUNC`来设置测试环境
3. 调用 `FUNC_PTR`执行测试，测试函数从静态上下文中获取之前设置的数据
4. 重复执行多次以获得统计结果

---

##  添加测试样例的步骤

### 步骤 1: 创建测试文件

在 `tests/performance/static_map/` 目录下创建新的 `.cpp` 文件，命名格式为 `perf_<operation>.cpp`。

例如：`perf_count.cpp`

**重要**: 所有测试代码必须放在 `namespace aclco::test` 命名空间内，以避免函数命名冲突。

### 步骤 2: 定义测试上下文结构

定义一个模板结构体来保存测试上下文数据（包括ACL流、设备缓冲区、map对象等）：

```cpp
namespace aclco::test {

template <typename K, typename V, int BucketSize>
struct CountTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::optional<map_factory::StaticMapT<K, V, BucketSize, map_factory::DefaultProbing<K>, aclco::EqualTo<K>>> map;
    // 添加其他需要的成员变量...
};

}
```

### 步骤 3: 定义上下文获取函数

定义一个静态函数来获取上下文实例：

```cpp
namespace aclco::test {

template <typename K, typename V, int BucketSize>
CountTestContext<K, V, BucketSize>& GetContext() {
    static CountTestContext<K, V, BucketSize> ctx;
    return ctx;
}

}
```

### 步骤 4: 编写设置函数

编写设置函数，该函数接收参数并初始化测试环境：

```cpp
namespace aclco::test {

template <typename K, typename V, int BucketSize>
void SetupCountTest(int numKeys, double loadFactor, int seed, std::string keyDistribution)
{
    auto& ctx = GetContext<K, V, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;

    using Key = K;
    using Value = V;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key, Value>();
    auto capacity = static_cast<size_t>(numKeys / loadFactor);
    ctx.map = map_factory::MakeStaticMap<Key, Value, BS, map_factory::DefaultProbing<Key>>(capacity, sent, ctx.stream);
    
    // 准备测试数据并保存到上下文
    auto hostPairs = MakeExamples<Key, Value>(seed, numKeys, sent, keyDistribution);
    ctx.hostPairs = hostPairs;
    
    // 准备设备缓冲区
    ctx.dPairs = DeviceBuffer<aclco::Pair<K, V>>(hostPairs.size());
    ctx.dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), ctx.stream);

    // 执行前置操作（如插入数据）
    ctx.map->Clear(ctx.stream);
    ctx.map->Insert(static_cast<void*>(ctx.dPairs.Data()),
                    aclco::Extent<std::size_t>(hostPairs.size()), ctx.stream);
}

}
```

### 步骤 5: 编写测试函数

编写测试函数，该函数不带参数，从静态上下文中获取数据：

```cpp
namespace aclco::test {

template <typename K, typename V, int BucketSize>
TestResult TestCount()
{
    auto& ctx = GetContext<K, V, BucketSize>();

    // 计时逻辑
    auto start = std::chrono::high_resolution_clock::now();
    
    // 在这里执行要测试的操作，使用上下文中的数据
    size_t count = ctx.map->Count(ctx.stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 返回测试结果
    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

}
```

### 步骤 6: 注册测试用例

使用 `REGISTER_PERFORMANCE_TEST` 宏注册测试函数和设置函数：

```cpp
// 参数: 测试名称, 测试函数指针, 设置函数指针, 参数类型列表
REGISTER_PERFORMANCE_TEST(count_case_1, (TestCount<uint32_t,uint32_t,12>), (SetupCountTest<uint32_t,uint32_t,12>), int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(count_case_2, (TestCount<uint64_t,uint64_t,12>), (SetupCountTest<uint64_t,uint64_t,12>), int, double, int, std::string);
```

### 步骤 7: 注册测试参数

使用 `REGISTER_PERFORMANCE_ARGS` 宏注册测试参数：

```cpp
// 参数: 测试名称, 参数组名称, 参数值列表, 参数类型列表
REGISTER_PERFORMANCE_ARGS(count_case_1, "count_uint32",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {500, 0.5, 200, "uniform"},
        {200, 0.5, 200, "uniform"}
    }),
    int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(count_case_2, "count_uint64",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {500, 0.5, 200, "uniform"},
        {200, 0.5, 200, "uniform"}
    }),
    int, double, int, std::string);
```


##  完整示例

以下是一个完整的 `perf_clear.cpp` 示例：

```cpp
#include "../performance_test_framework.h"

namespace aclco::test {

template <typename K, typename V, int BucketSize>
struct CountTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::optional<map_factory::StaticMapT<K, V, BucketSize, map_factory::DefaultProbing<K>, aclco::EqualTo<K>>> map;
};

template <typename K, typename V, int BucketSize>
CountTestContext<K, V, BucketSize>& GetContext() {
    static CountTestContext<K, V, BucketSize> ctx;
    return ctx;
}

template <typename K, typename V, int BucketSize>
void SetupCountTest(int numKeys, double loadFactor, int seed, std::string keyDistribution)
{
    auto& ctx = GetContext<K, V, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;

    using Key = K;
    using Value = V;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key, Value>();
    auto capacity = static_cast<size_t>(numKeys / loadFactor);
    ctx.map = map_factory::MakeStaticMap<Key, Value, BS, map_factory::DefaultProbing<Key>>(capacity, sent, ctx.stream);
    
    auto hostPairs = MakeExamples<Key, Value>(seed, numKeys, sent, keyDistribution);
    auto dPairs = DeviceBuffer<aclco::Pair<K, V>>(hostPairs.size());
    dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), ctx.stream);

    ctx.map->Clear(ctx.stream);
    ctx.map->Insert(static_cast<void*>(dPairs.Data()),
                    aclco::Extent<std::size_t>(hostPairs.size()), ctx.stream);
}

template <typename K, typename V, int BucketSize>
TestResult TestCount()
{
    auto& ctx = GetContext<K, V, BucketSize>();

    auto start = std::chrono::high_resolution_clock::now();
    
    size_t count = ctx.map->Count(ctx.stream);
    
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(count_case_1, (TestCount<uint32_t,uint32_t,12>), (SetupCountTest<uint32_t,uint32_t,12>), int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(count_case_2, (TestCount<uint64_t,uint64_t,12>), (SetupCountTest<uint64_t,uint64_t,12>), int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(count_case_1, "count_uint32",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {500, 0.5, 200, "uniform"},
        {200, 0.5, 200, "uniform"}
    }),
    int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(count_case_2, "count_uint64",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {500, 0.5, 200, "uniform"},
        {200, 0.5, 200, "uniform"}
    }),
    int, double, int, std::string);

}
```

---

## 常用工具函数

**注意**: 以下代码示例假设已在 `namespace aclco::test` 命名空间内。

### ACL资源管理

```cpp
AclStreamGuard streamGuard;
auto stream = streamGuard.stream;
```

### 数据生成

```cpp
auto sent = MakeDefaultSentinels<Key, Value>();

auto hostPairs = MakeExamples<Key, Value>(seed, numKeys, sent, keyDistribution);
```

### 设备内存管理

```cpp
// 创建设备缓冲区
DeviceBuffer<aclco::Pair<Key, Value>> dPairs(hostPairs.size());

// 从主机复制到设备（异步）
dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

// 从设备复制到主机（异步）
dPairs.CopyToHostAsync(hostPairs.data(), hostPairs.size(), stream);

// 同步流
Sync(stream);
```

### Map创建

```cpp
// 创建StaticMap
auto map = map_factory::MakeStaticMap<Key, Value, BS, map_factory::DefaultProbing<Key>>(capacity, sent, stream);
```

## 测试参数说明

### 参数类型

| 参数 | 类型 | 说明 |
|------|------|------|
| `numKeys` | `int` | 键的数量 |
| `loadFactor` | `double` | 负载因子（0.0 - 1.0） |
| `seed` | `int` | 随机数种子 |
| `keyDistribution` | `std::string` | 键分布类型（"uniform", "normal"等） |

### 负载因子说明

- **低负载因子（0.3 - 0.5）**: 更多空闲空间，插入快但空间利用率低
- **中等负载因子（0.5 - 0.7）**: 平衡性能和空间利用率
- **高负载因子（0.7 - 0.9）**: 空间利用率高但可能增加冲突

### 键分布类型

- `"uniform"`: 均匀分布
- `"normal"`: 正态分布
- `"sequential"`: 顺序分布

---

## 常见问题

### Q1: 如何修改测试的迭代次数？

**A**: 在 `performance_test_framework.h` 中修改 `max_iters` 常量：

```cpp
const int max_iters = 50;  // 修改为你需要的次数
```

### Q2: 如何修改稳定性判断标准？

**A**: 在 `performance_test_framework.h` 中修改 `ShouldStopIteration` 函数中的阈值：

```cpp
if (relStd < 0.005) {  // 修改0.005为你需要的阈值
    stopReason = "Stable (<0.5% RSD)";
    return true;
}
```

### Q3: 如何添加多个参数组？

**A**: 可以多次调用 `REGISTER_PERFORMANCE_ARGS`：

```cpp
REGISTER_PERFORMANCE_ARGS(count_case_1, "small_data",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {100, 0.5, 200, "uniform"}
    }),
    int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(count_case_1, "large_data",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {10000, 0.7, 300, "normal"}
    }),
    int, double, int, std::string);
```

### Q4: 如何测试不同的BucketSize？

**A**: 注册多个测试用例：

```cpp
REGISTER_PERFORMANCE_TEST(count_case_8, (TestCount<uint64_t,uint64_t,8>), (SetupCountTest<uint64_t,uint64_t,8>), int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(count_case_12, (TestCount<uint64_t,uint64_t,12>), (SetupCountTest<uint64_t,uint64_t,12>), int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(count_case_16, (TestCount<uint64_t,uint64_t,16>), (SetupCountTest<uint64_t,uint64_t,16>), int, double, int, std::string);
```



