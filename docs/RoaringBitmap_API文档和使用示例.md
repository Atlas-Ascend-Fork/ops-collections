# RoaringBitmap API文档和使用示例

## 一、功能说明

`aclco::RoaringBitmap` 是面向 Ascend 950PR&950DT 系列产品的只读压缩位图容器，参考 `cuco::experimental::roaring_bitmap` 实现。容器在 host 侧解析 Roaring portable serialization，将原始字节和查询元数据复制到 device，并通过 AIV SIMT kernel 批量判断 key 是否存在。

支持：

- `uint32_t` 和 `uint64_t` key。
- 32-bit portable 格式中的 array、bitset、run container。
- 64-bit portable 格式中的高 32 位 bucket 和嵌套 32-bit bitmap。
- 同步 `Contains` 和异步 `ContainsAsync`。
- `Size`、`Empty`、`Data`、`SizeBytes`、`GetAllocator`、`Ref`。

不支持 frozen serialization，也不负责从 key 数组生成 serialized bitmap。

## 二、类型定义

```cpp
#include "roaring_bitmap.h"

using Bitmap32 = aclco::RoaringBitmap<uint32_t>;
using Bitmap64 = aclco::RoaringBitmap<uint64_t>;
```

完整模板参数：

```cpp
template <typename Key,
          typename Extent = aclco::Extent<size_t>,
          typename Allocator = aclco::DefaultAllocator<uint8_t>>
class RoaringBitmap;
```

`Key` 只能是 `uint32_t` 或 `uint64_t`。自定义 allocator 的 `ValueType` 必须是 `uint8_t`，并提供 `Allocate(size_t)` 和 `Deallocate(uint8_t*)`。

## 三、构造与析构

```cpp
explicit RoaringBitmap(void const* bitmap,
                       Allocator const& allocator = {},
                       aclrtStream stream = nullptr);

RoaringBitmap(void const* bitmap,
              size_t bitmapBytes,
              Allocator const& allocator = {},
              aclrtStream stream = nullptr);
```

| 参数 | 位置 | 说明 |
|------|------|------|
| `bitmap` | Host | serialized Roaring Bitmap 起始地址 |
| `bitmapBytes` | Host | 输入缓冲区长度，用于完整边界校验 |
| `allocator` | Host | device 字节存储分配器 |
| `stream` | Host | H2D copy 使用的 ACL stream |

仅传 `bitmap` 的重载与 cuCollections 接口对齐，解析器会从格式本身推导长度；调用方必须保证数据完整可信。文件、网络或其他外部输入应使用带 `bitmapBytes` 的重载。

构造函数在返回前同步指定 stream，因此返回后 host 输入缓冲区可以释放。容器不可复制，可以移动。析构前必须保证所有使用该容器的异步操作已经完成。

## 四、Contains

```cpp
void Contains(void const* keys,
              void* outputValues,
              Extent keyNum,
              aclrtStream stream) const;

void ContainsAsync(void const* keys,
                   void* outputValues,
                   Extent keyNum,
                   aclrtStream stream) const;
```

| 参数 | 位置 | 说明 |
|------|------|------|
| `keys` | Device | `Key[keyNum]` 查询数组 |
| `outputValues` | Device | `bool[keyNum]` 输出数组 |
| `keyNum` | Host | 查询数量 |
| `stream` | Host | kernel 执行 stream |

`Contains` 在返回前同步 stream；`ContainsAsync` 只提交 kernel。`keyNum == 0` 时允许 `keys` 和 `outputValues` 为 `nullptr`。

## 五、状态接口

```cpp
uint64_t Size() const noexcept;
bool Empty() const noexcept;
uint8_t const* Data() const noexcept;
uint64_t SizeBytes() const noexcept;
Allocator GetAllocator() const;
RoaringBitmapRef<Key> Ref() const noexcept;
```

- `Size()`：bitmap 中 key 的数量。
- `Empty()`：是否不包含 key。
- `Data()`：device 侧 serialized bytes 起始地址。
- `SizeBytes()`：serialized bytes 长度，不包含内部查询元数据。
- `Ref()`：非拥有型 device 查询视图，支持单 key `Contains`。

## 六、调用示例

```cpp
#include <acl/acl.h>

#include <cstdint>
#include <fstream>
#include <vector>

#include "roaring_bitmap.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    {
        std::ifstream file("bitmapwithruns.bin", std::ios::binary | std::ios::ate);
        size_t bitmapBytes = static_cast<size_t>(file.tellg());
        std::vector<uint8_t> serialized(bitmapBytes);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(serialized.data()), bitmapBytes);

        aclco::RoaringBitmap<uint32_t> bitmap(
            serialized.data(), serialized.size(), {}, stream);

        std::vector<uint32_t> hostKeys{0, 1000, 12345};
        uint32_t* deviceKeys = nullptr;
        bool* deviceOutput = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&deviceKeys),
                    hostKeys.size() * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(reinterpret_cast<void**>(&deviceOutput),
                    hostKeys.size() * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpyAsync(deviceKeys, hostKeys.size() * sizeof(uint32_t),
                         hostKeys.data(), hostKeys.size() * sizeof(uint32_t),
                         ACL_MEMCPY_HOST_TO_DEVICE, stream);

        bitmap.Contains(deviceKeys, deviceOutput,
                        aclco::Extent<size_t>{hostKeys.size()}, stream);

        std::vector<uint8_t> hostOutput(hostKeys.size());
        aclrtMemcpy(hostOutput.data(), hostOutput.size(),
                    deviceOutput, hostOutput.size(), ACL_MEMCPY_DEVICE_TO_HOST);

        aclrtFree(deviceOutput);
        aclrtFree(deviceKeys);
    } // Destroy the bitmap before tearing down the ACL runtime.

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}
```

## 七、功能测试

构建并运行自生成格式用例：

```sh
bash scripts/build.sh -b
bash scripts/build.sh -r --test-name roaring_bitmap
```

构建时会自动生成 RoaringFormatSpec 兼容性测试数据，并将数据目录编译到测试程序中。
运行前无需手工设置环境变量：

```sh
./build_cmake/ccec_build/tests/collection_tests_roaring_bitmap_contains_test
```

目录中需要：

- `bitmapwithoutruns.bin`
- `bitmapwithruns.bin`
- `portable_bitmap64.bin`

`bitmapwithruns.bin` 用于 U32 数据，`portable_bitmap64.bin` 用于 U64 数据。二进制测试数据不随
代码仓库提交；它们由构建过程生成到 `tests/testdata`，测试程序会自动加载。

## 八、性能测试

RoaringBitmap 性能用例按 DynamicMap、StaticMap、StaticSet 的统一方式接入
`tests/performance/*/perf_*.cpp`。启用 `BUILD_PERFORMANCE` 后，CMake 会生成
`roaring_bitmap_perf_create`、`roaring_bitmap_perf_destroy` 和 `roaring_bitmap_perf_contains`
三个可执行文件：

```sh
bash scripts/build.sh -p
./build/performance/roaring_bitmap/roaring_bitmap_perf_contains
```

性能用例读取 Roaring portable 官方数据文件，并覆盖构造、析构和 Contains。目录中需要：

- `bitmapwithruns.bin`
- `portable_bitmap64.bin`

性能结果按性能框架输出；构造、析构和 Contains 分别作为独立用例运行。二进制测试数据在
性能编译时自动生成并由可执行文件自动加载。
