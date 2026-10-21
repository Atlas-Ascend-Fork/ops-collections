# ops-collections API文档和使用示例

本文档提供ops-collections库的详细API接口说明，包括构造函数、插入、查找、删除、清空等操作。

## 一、类型定义

### 1.1 StaticMap

```cpp
using Key = uint32_t;
using Value = uint32_t;
constexpr int BucketSize = 5;
using ProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<Key>>;
using KeyEqual = aclco::EqualTo<Key>;

using MyStaticMap = aclco::StaticMap<Key, Value, aclco::Extent<size_t>, KeyEqual, ProbingScheme>;
```

**类型说明：**

- `Key`：键类型，支持 `int32_t`、`uint32_t`、`float`，大小不超过8字节
- `Value`：值类型，支持同上类型，大小不超过8字节
- `BucketSize`：桶大小
- `ProbingScheme`：探测策略，默认为线性探测 `aclco::LinearProbing<aclco::murmurhash3_32<Key>>`
- `KeyEqual`：键比较器，默认为 `aclco::EqualTo<Key>`

### 1.2 StaticSet

```cpp
using SetKey = uint32_t;
constexpr int SetBucketSize = 5;
using SetProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<SetKey>>;
using SetKeyEqual = aclco::EqualTo<SetKey>;

using MyStaticSet = aclco::StaticSet<SetKey, aclco::Extent<size_t>, SetKeyEqual, SetProbingScheme>;
```

**类型说明：**

- `Key`：键类型，支持 `int32_t`、`uint32_t`、`float`，大小不超过8字节
- `BucketSize`：桶大小
- `ProbingScheme`：探测策略，默认为双重哈希探测 `aclco::DoubleHashing<aclco::xxhash_32<SetKey>>`
- `KeyEqual`：键比较器，默认为 `aclco::EqualTo<SetKey>`

## 二、容器类

### 2.1 StaticMap

静态哈希表容器，提供高效的键值对存储和查询功能。

### 2.2 StaticSet

静态哈希集合容器，提供高效的键存储和查询功能。与StaticMap不同，StaticSet仅存储键，不存储值。

## 三、核心算子API

### 3.1 构造函数

**函数签名：**

```cpp
// StaticMap
constexpr StaticMap(Extent capacity,
                    Key emptyKey,
                    T emptyValue,
                    KeyEqual const& pred = {},
                    ProbingScheme const& probingScheme = {},
                    Storage storage = {},
                    aclrtStream stream = nullptr);

// StaticSet
constexpr StaticSet(Extent capacity,
                    Key emptyKey,
                    KeyEqual const& pred = {},
                    ProbingScheme const& probingScheme = {},
                    Storage storage = {},
                    aclrtStream stream = nullptr);
```

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 |说明 |
|------|------|------|------|
| capacity | Extent | 输入 | map的容量 |
| emptyKey | Key | 输入 | 表示空键的标记值 |
| emptyValue | T | 输入 | 表示空值的标记值 |
| pred | KeyEqual const& | 输入 | 键比较器，默认为 `KeyEqual()` |
| probingScheme | ProbingScheme const& | 输入 | 探测策略，默认为 `ProbingScheme()` |
| storage | Storage | 输入 | 存储策略，默认为 `Storage<BucketSize>()` |
| stream | aclrtStream | 输入 | ACL流，默认为 `nullptr` |

**StaticSet：**

| 参数 | 类型 | 输入/输出 |说明 |
|------|------|------|------|
| capacity | Extent | 输入 | set的容量 |
| emptyKey | Key | 输入 | 表示空键的标记值 |
| pred | KeyEqual const& | 输入 | 键比较器，默认为 `KeyEqual()` |
| probingScheme | ProbingScheme const& | 输入 | 探测策略，默认为 `ProbingScheme()` |
| storage | Storage | 输入 | 存储策略，默认为 `Storage<BucketSize>()` |
| stream | aclrtStream | 输入 | ACL流，默认为 `nullptr` |

**功能说明：**
创建一个指定容量的容器。实际容量会向上取整到BucketSize的倍数。

**使用示例：**

```cpp
// 初始化ACL环境
aclInit(nullptr);
aclrtSetDevice(0);
aclrtStream stream;
aclrtCreateStream(&stream);

// 定义空键值对（根据类型自动选择合适的空值）
Key emptyKey = std::is_signed_v<Key> ? static_cast<Key>(-1) : std::numeric_limits<Key>::max();
Value emptyValue = std::is_signed_v<Value> ? static_cast<Value>(-1) : std::numeric_limits<Value>::max();

// 创建static_map，容量为100000
size_t capacity = 100000;
MyStaticMap map(capacity, emptyKey, emptyValue, KeyEqual(), ProbingScheme(), aclco::Storage<BucketSize>(), stream);

// StaticSet 构造
SetKey setEmptyKey = std::is_signed_v<SetKey> ? static_cast<SetKey>(-1) : std::numeric_limits<SetKey>::max();
size_t capacity = 100000;
MyStaticSet set(capacity, setEmptyKey, SetKeyEqual(), SetProbingScheme(), aclco::Storage<SetBucketSize>(), stream);
```

---

### 3.2 Insert - 插入

**函数签名：**

```cpp
// StaticMap：插入键值对
SizeType Insert(void *values, Extent valueNum, aclrtStream stream);
void InsertAsync(void *values, Extent valueNum, aclrtStream stream);

// StaticSet：插入键
SizeType Insert(void *keys, Extent keyNum, aclrtStream stream);
void InsertAsync(void *keys, Extent keyNum, aclrtStream stream);
```

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| values | void* | 输入 | Device侧指向键值对数组的指针 |
| valueNum | Extent | 输入 | 要插入的数量，**必须与指针指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| keyNum | Extent | 输入 | 要插入的数量，**必须与指针指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**

- `Insert`：返回插入失败的数量
- `InsertAsync`：无返回值

**功能说明：**

- `Insert`：同步插入，等待操作完成后返回
- `InsertAsync`：异步插入，需要调用 `aclrtSynchronizeStream` 等待完成

**注意事项：**

- 插入数量参数必须与指针指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `values.size()` / `keys.size()` 作为插入数量参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应

**使用示例：**

```cpp
// StaticMap：插入键值对
size_t insertCount = 10000;
std::vector<aclco::Pair<Key, Value>> hostPairs(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostPairs[i] = aclco::MakePair<Key, Value>(i, i * 2);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<aclco::Pair<Key, Value>> devicePairs(insertCount);
devicePairs.CopyFromHostAsync(hostPairs.data(), insertCount, stream);

// 同步插入操作
auto failedCount = map.Insert(static_cast<void*>(devicePairs.Data()),
                              aclco::Extent<size_t>(insertCount), stream);
// 输出插入失败个数
std::cout << "Insert failed count: " << failedCount << std::endl;

// 异步插入操作
map.InsertAsync(static_cast<void*>(devicePairs.Data()),
               aclco::Extent<size_t>(insertCount), stream);
aclrtSynchronizeStream(stream);
```

```cpp
// StaticSet：插入键
size_t insertCount = 10000;
std::vector<SetKey> hostKeys(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostKeys[i] = static_cast<SetKey>(i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<SetKey> deviceKeys(insertCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), insertCount, stream);

// 同步插入操作
auto failedCount = set.Insert(static_cast<void*>(deviceKeys.Data()),
                              aclco::Extent<size_t>(insertCount), stream);

// 输出插入失败个数
std::cout << "Insert failed count: " << failedCount << std::endl;

// 异步插入操作
set.InsertAsync(static_cast<void*>(deviceKeys.Data()),
               aclco::Extent<size_t>(insertCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.3 InsertIf - 条件插入

**函数签名：**

```cpp
// StaticMap：条件插入键值对
template <typename StencilT, typename Predicate>
SizeType InsertIf(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream);

template <typename StencilT, typename Predicate>
void InsertIfAsync(void *values, StencilT *stencil, Extent valueNum, aclrtStream stream);

// StaticSet：条件插入键
template <typename StencilT, typename Predicate>
SizeType InsertIf(void *keys, StencilT *stencil, Extent keyNum, aclrtStream stream);

template <typename StencilT, typename Predicate>
void InsertIfAsync(void *keys, StencilT *stencil, Extent keyNum, aclrtStream stream);
```

**模板参数说明：**

| 模板参数 | 说明 |
|------|------|
| StencilT | stencil数组的元素类型。stencil数组与输入数组一一对应，每个元素作为对应项的谓词判断输入，由仿函数根据stencil[i]的值决定是否插入 |
| Predicate | 仿函数类型，需提供 `operator()(StencilT) const` 重载，返回 `bool`；返回 `true` 表示执行插入，返回 `false` 表示跳过。仿函数需使用 `COLLECTION_HOST_DEVICE` 宏修饰，以确保在Host和Device侧均可调用 |

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| values | void* | 输入 | Device侧指向键值对数组的指针 |
| stencil | StencilT* | 输入 | Device侧指向stencil数组的指针，与values一一对应，用于谓词判断 |
| valueNum | Extent | 输入 | 要插入的数量，**必须与输入数组和stencil指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| stencil | StencilT* | 输入 | Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断 |
| keyNum | Extent | 输入 | 要插入的数量，**必须与输入数组和stencil指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**

- `InsertIf`：返回插入失败的元素数量（仅统计 `pred(stencil[i])` 为 `true` 且插入失败的元素）
- `InsertIfAsync`：无返回值

**功能说明：**

- `InsertIf`：同步条件插入，等待操作完成后返回
- `InsertIfAsync`：异步条件插入，需要调用 `aclrtSynchronizeStream` 等待完成
- 只有 `pred(stencil[i])` 为 `true` 时，才会尝试插入

**注意事项：**

- 插入数量参数必须与输入数组和stencil指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `values.size()` / `keys.size()` 作为插入数量参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应
- stencil中的元素类型必须与 `StencilT` 一致
- 仿函数需使用 `COLLECTION_HOST_DEVICE` 宏修饰，以确保在Host和Device侧均可调用

**使用示例：**

```cpp
// 定义仿函数：判断stencil值是否为奇数
struct IsOdd {
    COLLECTION_HOST_DEVICE bool operator()(uint32_t val) const noexcept
    {
        return val % 2 != 0;
    }
};

// StaticMap：条件插入键值对
size_t insertCount = 10000;
std::vector<aclco::Pair<Key, Value>> hostPairs(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostPairs[i] = aclco::MakePair<Key, Value>(i, i * 2);
}

// 准备stencil数组，与键值对一一对应
std::vector<uint32_t> hostStencil(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostStencil[i] = static_cast<uint32_t>(i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<aclco::Pair<Key, Value>> devicePairs(insertCount);
devicePairs.CopyFromHostAsync(hostPairs.data(), insertCount, stream);

aclco::DeviceBuffer<uint32_t> deviceStencil(insertCount);
deviceStencil.CopyFromHostAsync(hostStencil.data(), insertCount, stream);

// 同步条件插入操作：只有stencil为奇数时才插入对应的键值对
auto failedCount = map.InsertIf<uint32_t, IsOdd>(
    static_cast<void*>(devicePairs.Data()), deviceStencil.Data(),
    aclco::Extent<size_t>(insertCount), stream);
std::cout << "InsertIf failed count: " << failedCount << std::endl;

// 异步条件插入操作
map.InsertIfAsync<uint32_t, IsOdd>(
    static_cast<void*>(devicePairs.Data()), deviceStencil.Data(),
    aclco::Extent<size_t>(insertCount), stream);
aclrtSynchronizeStream(stream);
```

```cpp
// StaticSet：条件插入键
size_t insertCount = 10000;
std::vector<SetKey> hostKeys(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostKeys[i] = static_cast<SetKey>(i);
}

// 准备stencil数组，与键值对一一对应
std::vector<uint32_t> hostStencil(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostStencil[i] = static_cast<uint32_t>(i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<SetKey> deviceKeys(insertCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), insertCount, stream);

aclco::DeviceBuffer<uint32_t> deviceStencil(insertCount);
deviceStencil.CopyFromHostAsync(hostStencil.data(), insertCount, stream);

// 同步条件插入操作：只有stencil为奇数时才插入对应的键值对
auto failedCount = set.InsertIf<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    aclco::Extent<size_t>(insertCount), stream);
std::cout << "InsertIf failed count: " << failedCount << std::endl;

// 异步条件插入操作
set.InsertIfAsync<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    aclco::Extent<size_t>(insertCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.4 InsertOrAssign - 插入或更新

**函数签名：**

```cpp
// StaticMap：插入或更新键值对
SizeType InsertOrAssign(void *values, Extent valueNum, aclrtStream stream);
void InsertOrAssignAsync(void *values, Extent valueNum, aclrtStream stream);
```

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| values | void* | 输入 | Device侧指向键值对数组的指针 |
| valueNum | Extent | 输入 | 要插入或更新的键值对数量，**必须与values指向的数组实际大小一致（values.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

> **注意：** 此接口仅适用于 StaticMap，StaticSet 不支持此操作（Set中没有值的概念，无需"更新"语义，插入重复键时Insert本身即为幂等操作）

**返回值说明：**

- `InsertOrAssign`：返回插入或更新失败的键值对数量（仅当哈希表已满时才会失败）
- `InsertOrAssignAsync`：无返回值

**功能说明：**

- `InsertOrAssign`：同步插入或更新键值对到map中，等待操作完成后返回
- `InsertOrAssignAsync`：异步插入或更新键值对到map中，需要调用 `aclrtSynchronizeStream` 等待完成
- 如果键已存在，则更新对应的值；如果键不存在，则插入新的键值对
- 与 `Insert` 的区别：`Insert` 在键已存在时跳过（返回失败），`InsertOrAssign` 在键已存在时更新值

**注意事项：**

- `valueNum` 参数必须与 `values` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `values.size()` 作为 `valueNum` 参数，确保一致性
- 传入的指针中数据类型需要和map中的相对应
- 多线程并发对相同键执行 `InsertOrAssign` 时，最终写入的值是不确定的，但保证每个键有且仅有一个有效值

**使用示例：**

```cpp

// 准备初始键值对数据
size_t insertCount = 10000;
std::vector<aclco::Pair<Key, Value>> hostPairs(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostPairs[i] = aclco::MakePair<Key, Value>(i, i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<aclco::Pair<Key, Value>> devicePairs(insertCount);
devicePairs.CopyFromHostAsync(hostPairs.data(), insertCount, stream);

// 先插入初始键值对
auto failedCount = map.Insert(static_cast<void*>(devicePairs.Data()),
                              aclco::Extent<size_t>(insertCount), stream);
std::cout << "Insert failed count: " << failedCount << std::endl;

// 准备更新数据：相同键，新值
std::vector<aclco::Pair<Key, Value>> updatePairs(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    updatePairs[i] = aclco::MakePair<Key, Value>(i, i * 2);
}

// 分配设备内存并拷贝更新数据
aclco::DeviceBuffer<aclco::Pair<Key, Value>> deviceUpdatePairs(insertCount);
deviceUpdatePairs.CopyFromHostAsync(updatePairs.data(), insertCount, stream);

// 同步插入或更新操作：键已存在则更新值，键不存在则插入
auto updateFailedCount = map.InsertOrAssign(static_cast<void*>(deviceUpdatePairs.Data()),
                                            aclco::Extent<size_t>(insertCount), stream);
std::cout << "InsertOrAssign failed count: " << updateFailedCount << std::endl;

// 异步插入或更新操作
map.InsertOrAssignAsync(static_cast<void*>(deviceUpdatePairs.Data()),
                        aclco::Extent<size_t>(insertCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.5 InsertAndFind - 插入并查找

**函数签名：**

```cpp
// StaticMap：插入并查找键值对
void InsertAndFind(void *values, void *outputFind, void *outputInsert, Extent valueNum, aclrtStream stream);
void InsertAndFindAsync(void *values, void *outputFind, void *outputInsert, Extent valueNum, aclrtStream stream);

// StaticSet：插入并查找键
void InsertAndFind(void *keys, void *outputFind, void *outputInsert, Extent keyNum, aclrtStream stream);
void InsertAndFindAsync(void *keys, void *outputFind, void *outputInsert, Extent keyNum, aclrtStream stream);
```

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| values | void* | 输入 | Device侧指向键值对数组的指针 |
| outputFind | void* | 输出 | Device侧指向输出查找结果数组的指针，元素类型为Value。若键已存在则返回已存在的值，若键不存在则返回新插入的值，若键为空键或容量已满则返回空值（emptyValue） |
| outputInsert | void* | 输出 | Device侧指向输出插入标志数组的指针，元素类型为unsigned char。非0表示新插入成功，0表示键已存在或插入失败 |
| valueNum | Extent | 输入 | 要插入并查找的键值对数量，**必须与values指向的数组实际大小一致（values.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| outputFind | void* | 输出 | Device侧指向输出查找结果数组的指针，元素类型为Key。若键已存在则返回已存在的键，若键不存在则返回新插入的键，若键为空键或容量已满则返回空键（emptyKey） |
| outputInsert | void* | 输出 | Device侧指向输出插入标志数组的指针，元素类型为unsigned char。非0表示新插入成功，0表示键已存在或插入失败 |
| keyNum | Extent | 输入 | 要插入并查找的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**
无返回值，结果通过 `outputFind` 和 `outputInsert` 输出

**功能说明：**

- `InsertAndFind`：同步插入并查找，等待操作完成后返回
- `InsertAndFindAsync`：异步插入并查找，需要调用 `aclrtSynchronizeStream` 等待完成
- 对于每个键值对/键，先尝试插入；若键已存在则返回已存在的值/键且插入标志为0，若插入成功则返回新插入的值/键且插入标志非0
- 若键为空键或容量已满，`outputFind` 返回空值（emptyValue/emptyKey），`outputInsert` 为0

**注意事项：**

- `valueNum` / `keyNum` 参数必须与输入数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `values.size()` / `keys.size()` 作为数量参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应
- `outputFind` 的元素类型Map为Value、Set为Key，`outputInsert` 的元素类型为unsigned char

**使用示例：**

```cpp
// StaticMap：插入并查找键值对
size_t insertCount = 10000;
std::vector<aclco::Pair<Key, Value>> hostPairs(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostPairs[i] = aclco::MakePair<Key, Value>(i, i * 2);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<aclco::Pair<Key, Value>> devicePairs(insertCount);
devicePairs.CopyFromHostAsync(hostPairs.data(), insertCount, stream);

// 分配输出缓冲区
aclco::DeviceBuffer<Value> deviceOutputFind(insertCount);
aclco::DeviceBuffer<unsigned char> deviceOutputInsert(insertCount);

// 同步插入并查找操作
map.InsertAndFind(static_cast<void*>(devicePairs.Data()),
                  static_cast<void*>(deviceOutputFind.Data()),
                  static_cast<void*>(deviceOutputInsert.Data()),
                  aclco::Extent<size_t>(insertCount), stream);

// 将结果拷回主机
std::vector<Value> hostOutputFind(insertCount);
std::vector<unsigned char> hostOutputInsert(insertCount);
deviceOutputFind.CopyToHostAsync(hostOutputFind.data(), insertCount, stream);
deviceOutputInsert.CopyToHostAsync(hostOutputInsert.data(), insertCount, stream);
aclrtSynchronizeStream(stream);

// 打印结果
for (size_t i = 0; i < insertCount; ++i) {
    if (hostOutputInsert[i] != 0) {
        std::cout << "Key: " << hostPairs[i].first
                  << " inserted, value: " << hostOutputFind[i] << std::endl;
    } else {
        std::cout << "Key: " << hostPairs[i].first
                  << " already exists, value: " << hostOutputFind[i] << std::endl;
    }
}

// 异步插入并查找操作
map.InsertAndFindAsync(static_cast<void*>(devicePairs.Data()),
                       static_cast<void*>(deviceOutputFind.Data()),
                       static_cast<void*>(deviceOutputInsert.Data()),
                       aclco::Extent<size_t>(insertCount), stream);
aclrtSynchronizeStream(stream);
```

```cpp
// StaticSet：插入并查找键
size_t insertCount = 10000;
std::vector<SetKey> hostKeys(insertCount);
for (size_t i = 0; i < insertCount; ++i) {
    hostKeys[i] = static_cast<SetKey>(i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<SetKey> deviceKeys(insertCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), insertCount, stream);

// 分配输出缓冲区
aclco::DeviceBuffer<SetKey> deviceOutputFind(insertCount);
aclco::DeviceBuffer<unsigned char> deviceOutputInsert(insertCount);

// 同步插入并查找操作
set.InsertAndFind(static_cast<void*>(deviceKeys.Data()),
                  static_cast<void*>(deviceOutputFind.Data()),
                  static_cast<void*>(deviceOutputInsert.Data()),
                  aclco::Extent<size_t>(insertCount), stream);

// 将结果拷回主机
std::vector<SetKey> hostOutputFind(insertCount);
std::vector<unsigned char> hostOutputInsert(insertCount);
deviceOutputFind.CopyToHostAsync(hostOutputFind.data(), insertCount, stream);
deviceOutputInsert.CopyToHostAsync(hostOutputInsert.data(), insertCount, stream);
aclrtSynchronizeStream(stream);

// 打印结果
for (size_t i = 0; i < insertCount; ++i) {
    if (hostOutputInsert[i] != 0) {
        std::cout << "Key: " << hostKeys[i]
                  << " inserted" << std::endl;
    } else {
        std::cout << "Key: " << hostKeys[i]
                  << " already exists" << std::endl;
    }
}

// 异步插入并查找操作
set.InsertAndFindAsync(static_cast<void*>(deviceKeys.Data()),
                       static_cast<void*>(deviceOutputFind.Data()),
                       static_cast<void*>(deviceOutputInsert.Data()),
                       aclco::Extent<size_t>(insertCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.6 Find - 查找

**函数签名：**

```cpp
// StaticMap：查找键对应的值
void Find(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);
void FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);

// StaticSet：查找键
void Find(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);
void FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);
```

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| outputValues | void* | 输出 | Device侧指向输出值数组的指针 |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| outputValues | void* | 输出 | Device侧指向输出键数组的指针 |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**
无返回值，查找结果通过 `outputValues` 输出

**功能说明：**

- `Find`：同步查找，等待操作完成后返回
- `FindAsync`：异步查找，需要调用 `aclrtSynchronizeStream` 等待完成
- 如果键不存在，返回空值（emptyValue）

**注意事项：**

- `keyNum` 参数必须与 `keys` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `keys.size()` 作为 `keyNum` 参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应

**使用示例：**

```cpp
// StaticMap：查找键对应的值
size_t findCount = 1000;
std::vector<Key> hostKeys(findCount);
for (size_t i = 0; i < findCount; ++i) {
    hostKeys[i] = i * 10;
}

// 分配设备内存并拷贝键数据
aclco::DeviceBuffer<Key> deviceKeys(findCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), findCount, stream);

// 分配输出缓冲区
aclco::DeviceBuffer<Value> deviceValues(findCount);

// 同步查找操作
map.Find(static_cast<void*>(deviceKeys.Data()),
         static_cast<void*>(deviceValues.Data()),
         aclco::Extent<size_t>(findCount), stream);

// 将结果拷回主机
std::vector<Value> hostValues(findCount);
deviceValues.CopyToHostAsync(hostValues.data(), findCount, stream);
aclrtSynchronizeStream(stream);

// 打印查找结果
for (size_t i = 0; i < findCount; ++i) {
    if (hostValues[i] != emptyValue) {
        std::cout << "Key: " << hostKeys[i] << ", Value: " << hostValues[i] << std::endl;
    } else {
        std::cout << "Key: " << hostKeys[i] << " not found" << std::endl;
    }
}

// 异步查找操作
map.FindAsync(static_cast<void*>(deviceKeys.Data()),
             static_cast<void*>(deviceValues.Data()),
             aclco::Extent<size_t>(findCount), stream);
aclrtSynchronizeStream(stream);
```

```cpp
// StaticSet：查找键
size_t findCount = 1000;
std::vector<SetKey> hostKeys(findCount);
for (size_t i = 0; i < findCount; ++i) {
    hostKeys[i] = i * 10;
}

// 分配设备内存并拷贝键数据
aclco::DeviceBuffer<SetKey> deviceKeys(findCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), findCount, stream);

// 分配输出缓冲区
aclco::DeviceBuffer<SetKey> deviceOutputKeys(findCount);

// 同步查找操作
set.Find(static_cast<void*>(deviceKeys.Data()),
         static_cast<void*>(deviceOutputKeys.Data()),
         aclco::Extent<size_t>(findCount), stream);

// 异步查找操作
set.FindAsync(static_cast<void*>(deviceKeys.Data()),
             static_cast<void*>(deviceOutputKeys.Data()),
             aclco::Extent<size_t>(findCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.7 FindIf - 条件查找

**函数签名：**

```cpp
// StaticMap：条件查找键对应的值
template <typename StencilT, typename Predicate>
void FindIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

template <typename StencilT, typename Predicate>
void FindIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

// StaticSet：条件查找键
template <typename StencilT, typename Predicate>
void FindIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

template <typename StencilT, typename Predicate>
void FindIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);
```

**模板参数说明：**

| 模板参数 | 说明 |
|------|------|
| StencilT | stencil数组的元素类型。stencil数组与keys数组一一对应，每个元素作为对应键的谓词判断输入，由仿函数根据stencil[i]的值决定是否查找keys[i] |
| Predicate | 仿函数类型，需提供 `operator()(StencilT) const` 重载，返回 `bool`；返回 `true` 表示执行查找，返回 `false` 表示跳过。仿函数需使用 `COLLECTION_HOST_DEVICE` 宏修饰，以确保在Host和Device侧均可调用 |

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| stencil | StencilT* | 输入 | Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断 |
| outputValues | void* | 输出 | Device侧指向输出值数组的指针 |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys和stencil指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| stencil | StencilT* | 输入 | Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断 |
| outputValues | void* | 输出 | Device侧指向输出键数组的指针 |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys和stencil指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**
无返回值，查找结果通过 `outputValues` 输出

**功能说明：**

- `FindIf`：同步条件查找，等待操作完成后返回
- `FindIfAsync`：异步条件查找，需要调用 `aclrtSynchronizeStream` 等待完成
- 只有 `pred(stencil[i])` 为 `true` 时，才会查找 `keys[i]`
- 当 `pred(stencil[i])` 为 `false` 时，`outputValues[i]` 写入空值（emptyValue）
- 如果键不存在，返回空值（emptyValue）

**注意事项：**

- `keyNum` 参数必须与 `keys` 和 `stencil` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `keys.size()` 作为 `keyNum` 参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应
- stencil中的元素类型必须与 `StencilT` 一致
- 仿函数需使用 `COLLECTION_HOST_DEVICE` 宏修饰，以确保在Host和Device侧均可调用

**使用示例：**

```cpp
// 定义仿函数：判断stencil值是否为奇数
struct IsOdd {
    COLLECTION_HOST_DEVICE bool operator()(uint32_t val) const noexcept
    {
        return val % 2 != 0;
    }
};

// StaticMap：条件查找键对应的值
size_t findCount = 10000;
std::vector<Key> hostKeys(findCount);
for (size_t i = 0; i < findCount; ++i) {
    hostKeys[i] = i * 10;
}

// 准备stencil数组，与键一一对应
std::vector<uint32_t> hostStencil(findCount);
for (size_t i = 0; i < findCount; ++i) {
    hostStencil[i] = static_cast<uint32_t>(i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<Key> deviceKeys(findCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), findCount, stream);

aclco::DeviceBuffer<uint32_t> deviceStencil(findCount);
deviceStencil.CopyFromHostAsync(hostStencil.data(), findCount, stream);

// 分配输出缓冲区
aclco::DeviceBuffer<Value> deviceValues(findCount);

// 同步条件查找操作：只有stencil为奇数时才查找对应的键
map.FindIf<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    static_cast<void*>(deviceValues.Data()),
    aclco::Extent<size_t>(findCount), stream);

// 异步条件查找操作
map.FindIfAsync<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    static_cast<void*>(deviceValues.Data()),
    aclco::Extent<size_t>(findCount), stream);
aclrtSynchronizeStream(stream);
```

```cpp
// StaticSet：条件查找键
size_t findCount = 10000;
std::vector<SetKey> hostKeys(findCount);
for (size_t i = 0; i < findCount; ++i) {
    hostKeys[i] = i * 10;
}

// 准备stencil数组，与键一一对应
std::vector<uint32_t> hostStencil(findCount);
for (size_t i = 0; i < findCount; ++i) {
    hostStencil[i] = static_cast<uint32_t>(i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<SetKey> deviceKeys(findCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), findCount, stream);

aclco::DeviceBuffer<uint32_t> deviceStencil(findCount);
deviceStencil.CopyFromHostAsync(hostStencil.data(), findCount, stream);

// 分配输出缓冲区
aclco::DeviceBuffer<SetKey> deviceOutputKeys(findCount);

// 同步条件查找操作：只有stencil为奇数时才查找对应的键
set.FindIf<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    static_cast<void*>(deviceOutputKeys.Data()),
    aclco::Extent<size_t>(findCount), stream);

// 异步条件查找操作
set.FindIfAsync<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    static_cast<void*>(deviceOutputKeys.Data()),
    aclco::Extent<size_t>(findCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.8 Contains - 检查键是否存在

**函数签名：**

```cpp
// StaticMap / StaticSet：检查键是否存在
void Contains(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);
void ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);
```

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| outputValues | void* | 输出 | Device侧指向输出值数组的指针（bool类型） |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| outputValues | void* | 输出 | Device侧指向输出值数组的指针（bool类型） |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**
无返回值，检查结果通过 `outputValues` 输出（bool类型）

**功能说明：**

- `Contains`：同步检查键是否存在，等待操作完成后返回
- `ContainsAsync`：异步检查键是否存在，需要调用 `aclrtSynchronizeStream` 等待完成
- 输出值为 `true` 表示键存在，`false` 表示键不存在

**注意事项：**

- `keyNum` 参数必须与 `keys` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `keys.size()` 作为 `keyNum` 参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应

**使用示例：**

```cpp
// StaticMap / StaticSet 用法相同
size_t checkCount = 1000;
std::vector<Key> hostKeys(checkCount);
for (size_t i = 0; i < checkCount; ++i) {
    hostKeys[i] = i * 10;
}

// 分配设备内存并拷贝键数据
aclco::DeviceBuffer<Key> deviceKeys(checkCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), checkCount, stream);

// 分配输出缓冲区（bool类型）
aclco::DeviceBuffer<bool> deviceResults(checkCount);

// 同步检查操作
map.Contains(static_cast<void*>(deviceKeys.Data()),
            static_cast<void*>(deviceResults.Data()),
            aclco::Extent<size_t>(checkCount), stream);

// 将结果拷回主机
std::unique_ptr<bool[]> hostResults = std::make_unique<bool[]>(checkCount);
deviceResults.CopyToHostAsync(hostResults.get(), checkCount, stream);
aclrtSynchronizeStream(stream);

// 打印检查结果
for (size_t i = 0; i < checkCount; ++i) {
    std::cout << "Key: " << hostKeys[i]
              << ", Exists: " << (hostResults[i] ? "Yes" : "No") << std::endl;
}

// 异步检查操作
map.ContainsAsync(static_cast<void*>(deviceKeys.Data()),
                 static_cast<void*>(deviceResults.Data()),
                 aclco::Extent<size_t>(checkCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.9 ContainsIf - 条件检查键是否存在

**函数签名：**

```cpp
// StaticMap / StaticSet：条件检查键是否存在
template <typename StencilT, typename Predicate>
void ContainsIf(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);

template <typename StencilT, typename Predicate>
void ContainsIfAsync(void *keys, StencilT *stencil, void *outputValues, Extent keyNum, aclrtStream stream);
```

**模板参数说明：**

| 模板参数 | 说明 |
|------|------|
| StencilT | stencil数组的元素类型。stencil数组与keys数组一一对应，每个元素作为对应键的谓词判断输入，由仿函数根据stencil[i]的值决定是否检查keys[i] |
| Predicate | 仿函数类型，需提供 `operator()(StencilT) const` 重载，返回 `bool`；返回 `true` 表示执行检查，返回 `false` 表示跳过。仿函数需使用 `COLLECTION_HOST_DEVICE` 宏修饰，以确保在Host和Device侧均可调用 |

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| stencil | StencilT* | 输入 | Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断 |
| outputValues | void* | 输出 | Device侧指向输出值数组的指针（bool类型） |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys和stencil指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| stencil | StencilT* | 输入 | Device侧指向stencil数组的指针，与keys一一对应，用于谓词判断 |
| outputValues | void* | 输出 | Device侧指向输出值数组的指针（bool类型） |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys和stencil指向的数组实际大小一致** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**
无返回值，检查结果通过 `outputValues` 输出（bool类型）

**功能说明：**

- `ContainsIf`：同步条件检查键是否存在，等待操作完成后返回
- `ContainsIfAsync`：异步条件检查键是否存在，需要调用 `aclrtSynchronizeStream` 等待完成
- 只有 `pred(stencil[i])` 为 `true` 时，才会检查 `keys[i]`
- 当 `pred(stencil[i])` 为 `false` 时，`outputValues[i]` 写入 `false`
- 输出值为 `true` 表示键存在，`false` 表示键不存在或谓词为false

**注意事项：**

- `keyNum` 参数必须与 `keys` 和 `stencil` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `keys.size()` 作为 `keyNum` 参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应
- stencil中的元素类型必须与 `StencilT` 一致
- 仿函数需使用 `COLLECTION_HOST_DEVICE` 宏修饰，以确保在Host和Device侧均可调用

**使用示例：**

```cpp
// StaticMap / StaticSet 用法相同
// 定义仿函数：判断stencil值是否为奇数
struct IsOdd {
    COLLECTION_HOST_DEVICE bool operator()(uint32_t val) const noexcept
    {
        return val % 2 != 0;
    }
};

// 准备要检查的键
size_t checkCount = 10000;
std::vector<Key> hostKeys(checkCount);
for (size_t i = 0; i < checkCount; ++i) {
    hostKeys[i] = i * 10;
}

// 准备stencil数组，与键一一对应
std::vector<uint32_t> hostStencil(checkCount);
for (size_t i = 0; i < checkCount; ++i) {
    hostStencil[i] = static_cast<uint32_t>(i);
}

// 分配设备内存并拷贝数据
aclco::DeviceBuffer<Key> deviceKeys(checkCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), checkCount, stream);

aclco::DeviceBuffer<uint32_t> deviceStencil(checkCount);
deviceStencil.CopyFromHostAsync(hostStencil.data(), checkCount, stream);

// 分配输出缓冲区（bool类型）
aclco::DeviceBuffer<bool> deviceResults(checkCount);

// 同步条件检查操作：只有stencil为奇数时才检查对应的键
map.ContainsIf<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    static_cast<void*>(deviceResults.Data()),
    aclco::Extent<size_t>(checkCount), stream);

// 异步条件检查操作
map.ContainsIfAsync<uint32_t, IsOdd>(
    static_cast<void*>(deviceKeys.Data()), deviceStencil.Data(),
    static_cast<void*>(deviceResults.Data()),
    aclco::Extent<size_t>(checkCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.10 Erase - 删除

**函数签名：**

```cpp
// StaticMap / StaticSet：删除键
SizeType Erase(void *keys, Extent keyNum, aclrtStream stream);
void EraseAsync(void *keys, Extent keyNum, aclrtStream stream);
```

**参数说明：**

**StaticMap：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| keyNum | Extent | 输入 | 要删除的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**StaticSet：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| keyNum | Extent | 输入 | 要删除的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**

- `Erase`：返回删除失败的键数量
- `EraseAsync`：无返回值

**功能说明：**

- `Erase`：同步删除，等待操作完成后返回
- `EraseAsync`：异步删除，需要调用 `aclrtSynchronizeStream` 等待完成

**注意事项：**

- `keyNum` 参数必须与 `keys` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `keys.size()` 作为 `keyNum` 参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应

**使用示例：**

```cpp
// StaticMap / StaticSet 用法相同
// 准备要删除的键
size_t eraseCount = 500;
std::vector<Key> hostKeys(eraseCount);
for (size_t i = 0; i < eraseCount; ++i) {
    hostKeys[i] = i * 20;
}

// 分配设备内存并拷贝键数据
aclco::DeviceBuffer<Key> deviceKeys(eraseCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), eraseCount, stream);

// 同步删除操作
auto failedCount = map.Erase(static_cast<void*>(deviceKeys.Data()),
                              aclco::Extent<size_t>(eraseCount), stream);

// 输出删除失败个数
std::cout << "Erase failed count: " << failedCount << std::endl;

// 异步删除操作
map.EraseAsync(static_cast<void*>(deviceKeys.Data()),
               aclco::Extent<size_t>(eraseCount), stream);
aclrtSynchronizeStream(stream);
```

---

### 3.11 ForEach - 遍历匹配槽位并执行回调

**函数签名：**

```cpp
template <typename CallbackOp>
void ForEach(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream);

template <typename CallbackOp>
void ForEachAsync(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream);
```

**模板参数说明：**

| 模板参数 | 说明 |
|------|------|
| CallbackOp | 仿函数类型，要求如下：① **Map** 提供 `COLLECTION_SIMT_DEVICE void operator()(Pair<Key, T>) const` 重载，**Set** 提供 `COLLECTION_SIMT_DEVICE void operator()(Key) const` 重载，接收匹配的槽位作为参数；② 提供 `COLLECTION_SIMT_DEVICE` 构造函数接受 `__gm__ uint8_t*` 参数，用于接收 callbackArgs 指针并在内部 `reinterpret_cast` 为实际类型；③ operator() 中可使用 `AscendC::Simt::AtomicAdd` 等设备端原子操作访问 callbackArgs 指向的设备内存 |

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| keyNum | Extent | 输入 | 要遍历的键数量，**必须与keys指向的数组实际大小一致** |
| callbackArgs | void* | 输入 | Device侧指向用户自定义数据的指针，以 `void*` 类型擦除传入kernel，由 CallbackOp 构造函数 `reinterpret_cast` 为实际类型使用 |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**
无返回值，回调结果通过 callbackArgs 指向的设备内存输出

**功能说明：**

- `ForEach`：同步遍历哈希表中与指定键匹配的槽位，对每个匹配的槽位执行回调函数，等待操作完成后返回
- `ForEachAsync`：异步遍历哈希表中与指定键匹配的槽位，对每个匹配的槽位执行回调函数，需要调用 `aclrtSynchronizeStream` 等待完成
- 对于每个 key，沿探测序列查找匹配的槽位，找到匹配时调用 `callback(slot)`；遇到空槽位则停止探测
- callbackArgs 的设计原因：昇腾 NPU 的 SIMT 编程模型不支持将函数指针或带捕获的 lambda 传递到 device 侧，因此采用**模板仿函数 + callbackArgs 指针**的方式传递设备端状态

**注意事项：**

- `keyNum` 参数必须与 `keys` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 传入的指针中数据类型需要和容器中的相对应
- 回调函数中不应修改哈希表的状态，否则可能导致未定义行为
- callbackArgs 指向的设备内存需要在调用前正确初始化（如使用 `MemsetZero` 清零计数器）
- CallbackOp 的 operator() 仅在 Device 侧调用，不可使用 `COLLECTION_HOST_DEVICE` 修饰（否则其中调用的 `AtomicAdd` 等 device-only 函数会在 host 编译时报错）
- CallbackOp 的构造函数需要 `COLLECTION_SIMT_DEVICE` 修饰，因为仿函数在 kernel 内部构造，仅在 Device 侧执行

**使用示例：**

```cpp
// StaticMap：定义回调仿函数，统计偶数键且值为1的槽位数
template <typename Key, typename Value>
struct CountEvenKeyWithValueOne {
    __gm__ uint32_t *counter;

    CountEvenKeyWithValueOne() : counter{nullptr} {}
    COLLECTION_SIMT_DEVICE CountEvenKeyWithValueOne(__gm__ uint8_t *state)
        : counter{reinterpret_cast<__gm__ uint32_t*>(state)} {}

    COLLECTION_SIMT_DEVICE void operator()(aclco::Pair<Key, Value> slot) const noexcept
    {
        if (slot.first % 2 == 0 && slot.second == 1) {
            AscendC::Simt::AtomicAdd(counter, 1u);
        }
    }
};

// 准备探测键
size_t keyCount = 1000;
std::vector<Key> hostKeys(keyCount);
for (size_t i = 0; i < keyCount; ++i) {
    hostKeys[i] = static_cast<Key>(i);
}

// 分配设备内存并拷贝键数据
aclco::DeviceBuffer<Key> deviceKeys(keyCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), keyCount, stream);

// 分配并初始化回调参数（计数器）
aclco::DeviceBuffer<uint32_t> deviceCounter(1);
deviceCounter.MemsetZero(stream);

// 同步遍历操作
map.ForEach<CountEvenKeyWithValueOne<Key, Value>>(
    static_cast<void*>(deviceKeys.Data()),
    aclco::Extent<size_t>(keyCount),
    static_cast<void*>(deviceCounter.Data()),
    stream);

// 读取结果
auto result = deviceCounter.CopyToHost(stream);
std::cout << "Even keys with value 1 count: " << result[0] << std::endl;

// 异步遍历操作
map.ForEachAsync<CountEvenKeyWithValueOne<Key, Value>>(
    static_cast<void*>(deviceKeys.Data()),
    aclco::Extent<size_t>(keyCount),
    static_cast<void*>(deviceCounter.Data()),
    stream);
aclrtSynchronizeStream(stream);
```

```cpp
// StaticSet：定义回调仿函数，统计偶数键的槽位数
template <typename Key>
struct CountEvenKey {
    __gm__ uint32_t *counter;

    CountEvenKey() : counter{nullptr} {}
    COLLECTION_SIMT_DEVICE CountEvenKey(__gm__ uint8_t *state)
        : counter{reinterpret_cast<__gm__ uint32_t*>(state)} {}

    COLLECTION_SIMT_DEVICE void operator()(Key key) const noexcept
    {
        if (key % 2 == 0) {
            AscendC::Simt::AtomicAdd(counter, 1u);
        }
    }
};

// 准备探测键
size_t keyCount = 1000;
std::vector<SetKey> hostKeys(keyCount);
for (size_t i = 0; i < keyCount; ++i) {
    hostKeys[i] = static_cast<SetKey>(i);
}

// 分配设备内存并拷贝键数据
aclco::DeviceBuffer<SetKey> deviceKeys(keyCount);
deviceKeys.CopyFromHostAsync(hostKeys.data(), keyCount, stream);

// 分配并初始化回调参数（计数器）
aclco::DeviceBuffer<uint32_t> deviceCounter(1);
deviceCounter.MemsetZero(stream);

// 同步遍历操作
set.ForEach<CountEvenKey<SetKey>>(
    static_cast<void*>(deviceKeys.Data()),
    aclco::Extent<size_t>(keyCount),
    static_cast<void*>(deviceCounter.Data()),
    stream);

// 读取结果
auto result = deviceCounter.CopyToHost(stream);
std::cout << "Even key count: " << result[0] << std::endl;

// 异步遍历操作
set.ForEachAsync<CountEvenKey<SetKey>>(
    static_cast<void*>(deviceKeys.Data()),
    aclco::Extent<size_t>(keyCount),
    static_cast<void*>(deviceCounter.Data()),
    stream);
aclrtSynchronizeStream(stream);
```

---

### 3.12 Clear - 清空容器

**函数签名：**

```cpp
void Clear(aclrtStream stream);
void ClearAsync(aclrtStream stream) noexcept;
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| stream | aclrtStream | 输入 |ACL流 |

**返回值说明：**
无返回值

**功能说明：**

- `Clear`：同步清空容器，等待操作完成后返回
- `ClearAsync`：异步清空容器，需要调用 `aclrtSynchronizeStream` 等待完成

**使用示例：**

```cpp
// StaticMap / StaticSet 用法相同
// 同步清空操作
map.Clear(stream);

// 异步清空操作
map.ClearAsync(stream);
aclrtSynchronizeStream(stream);
```

---

### 3.13 Count - 统计键存在的数量

**函数签名：**

```cpp
SizeType Count(void *keys, Extent keyNum, aclrtStream stream);
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| keys | void* | 输入 | Device侧指向键数组的指针 |
| keyNum | Extent | 输入 | 要查找的键数量，**必须与keys指向的数组实际大小一致（keys.size()）** |
| stream | aclrtStream | 输入 | ACL流 |

**返回值说明：**
返回存在的键数量

**功能说明：**

- `Count`：同步统计指定键在容器中存在的数量，等待操作完成后返回

**注意事项：**

- `keyNum` 参数必须与 `keys` 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
- 建议使用 `keys.size()` 作为 `keyNum` 参数，确保一致性
- 传入的指针中数据类型需要和容器中的相对应

**使用示例：**

```cpp
// StaticMap / StaticSet 用法相同
// 准备要统计的键
size_t countKeyNum = 1000;
std::vector<Key> hostKeys(countKeyNum);
for (size_t i = 0; i < countKeyNum; ++i) {
    hostKeys[i] = i * 10;
}

// 分配设备内存并拷贝键数据
aclco::DeviceBuffer<Key> deviceKeys(countKeyNum);
deviceKeys.CopyFromHostAsync(hostKeys.data(), countKeyNum, stream);

// 同步统计操作
auto existCount = map.Count(static_cast<void*>(deviceKeys.Data()),
                            aclco::Extent<size_t>(countKeyNum), stream);
std::cout << "Exist key count: " << existCount << std::endl;
```

---

### 3.14 Capacity - 获取容量

**函数签名：**

```cpp
constexpr auto Capacity() const noexcept;
```

**参数说明：**
无参数

**返回值说明：**
返回容器的实际容量

**功能说明：**
获取容器的实际容量（向上取整到BucketSize的倍数）

**使用示例：**

```cpp
// StaticMap
size_t mapCapacity = map.Capacity();
std::cout << "Map capacity: " << mapCapacity << std::endl;

// StaticSet
size_t setCapacity = set.Capacity();
std::cout << "Set capacity: " << setCapacity << std::endl;
```

---

### 3.15 Data - 获取数据指针

**函数签名：**

```cpp
auto Data() const noexcept;
```

**参数说明：**
无参数

**返回值说明：**
返回Device侧容器内部数据的指针

**功能说明：**
获取Device侧容器内部数据的指针，用于直接访问底层存储

**使用示例：**

```cpp
// StaticMap
auto mapDataPtr = map.Data();

// StaticSet
auto setDataPtr = set.Data();
```

---

## 四、辅助组件API

### 4.1 Pair - 键值对

**函数签名：**

```cpp
template<typename F, typename S>
constexpr Pair<F, S> MakePair(F&& f, S&& s) noexcept;
```

**功能说明：**
创建键值对对象。接口参数为转发引用；显式指定 `F`、`S` 为值类型时，需传入对应的右值。以下示例用值转换临时量构造 `Pair<uint32_t, uint32_t>`。直接省略模板参数并传入左值会推导引用类型，不能将这种结果视为相同的值类型 Pair。

**使用示例：**

```cpp
uint32_t key = 1;
uint32_t value = 2;
auto pair = aclco::MakePair<uint32_t, uint32_t>(
    static_cast<uint32_t>(key), static_cast<uint32_t>(value));
```

### 4.2 Extent - 容量表示

**函数签名：**

```cpp
template<typename SizeType>
Extent(SizeType capacity);
```

**功能说明：**
创建容量表示对象

**使用示例：**

```cpp
auto extent = aclco::Extent<size_t>(10000);
```

### 4.3 Storage - 存储策略

**函数签名：**

```cpp
template<size_t BucketSize>
Storage();
```

**功能说明：**
创建存储策略对象，定义桶大小

**使用示例：**

```cpp
auto storage = aclco::Storage<5>();
```

### 4.4 LinearProbing - 线性探测策略

**函数签名：**

```cpp
template<typename Hash>
LinearProbing(Hash const& hash = {});
```

**功能说明：**
创建线性探测策略对象

**使用示例：**

```cpp
using ProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<Key>>;
```

### 4.5 DoubleHashing - 双重哈希探测策略

**函数签名：**

```cpp
template<typename Hash1, typename Hash2 = Hash1>
DoubleHashing(Hash1 const& hash1 = {}, Hash2 const& hash2 = {1});
```

**功能说明：**
创建双重哈希探测策略对象

**使用示例：**

```cpp
using ProbingScheme = aclco::DoubleHashing<aclco::xxhash_32<Key>>;
```

### 4.6 Hash Functions - 哈希函数

**支持的哈希函数：**

- `aclco::murmurhash3_32<Key>` - MurmurHash3 哈希函数
- `aclco::murmurhash3_fmix32<Key>` - MurmurHash3 Fmix 哈希函数（32位）
- `aclco::murmurhash3_fmix64<Key>` - MurmurHash3 Fmix 哈希函数（64位）
- `aclco::xxhash_32<Key>` - XXHash 哈希函数

**使用示例：**

```cpp
using HashFunc = aclco::murmurhash3_32<Key>;
using HashFunc = aclco::murmurhash3_fmix32<Key>;
using HashFunc = aclco::murmurhash3_fmix64<Key>;
using HashFunc = aclco::xxhash_32<Key>;
```

---

## 五、返回主文档

- **[返回README](../README.md)**
- **[查看开发指导](开发指导.md)**
