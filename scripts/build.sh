#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# 集合测试框架构建和运行脚本
# 功能包括：编译模式配置、构建 Catch2、构建主项目、运行测试

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 默认配置
compile_mode="default"
need_debug="NONE"
ascend_home_path="${ASCEND_HOME_PATH:-}"
ccec_aicore_arch="${CCE_AICORE_ARCH:-dav-c310}"
bisheng_aicore_arch="${BISHENG_AICORE_ARCH:-dav-3510}"
build_dir="${ROOT_DIR}/build_cmake/ccec_build"
bin_dir="${build_dir}/tests"
catch2_src="${ROOT_DIR}/3rdparty/Catch2"
catch2_build="${ROOT_DIR}/build_cmake/_catch2_gxx_build"
catch2_install="${ROOT_DIR}/build_cmake/_catch2_gxx_install"
doxygen_build="${ROOT_DIR}/build_cmake/_doxygen_gxx_build"
doxygen_install="${ROOT_DIR}/build_cmake/_doxygen_gxx_install"
graphviz_build="${ROOT_DIR}/build_cmake/_graphviz_download"
graphviz_install="${ROOT_DIR}/build_cmake/_graphviz_install"
doxygen_markdown="${ROOT_DIR}/build_cmake/_doxygen_markdown"

# 清理构建目录
function clean_build() {
  echo "[INFO] 清理构建目录..."
  if [ -d "${ROOT_DIR}/build" ]; then
    rm -rf "${ROOT_DIR}/build"
  fi
  if [ -d "${ROOT_DIR}/build_cmake" ]; then
    rm -rf "${ROOT_DIR}/build_cmake"
  fi
  if [ -f "${ROOT_DIR}/kernel.o" ]; then
    rm -rf "${ROOT_DIR}/kernel.o"
  fi
  echo "[INFO] 清理完成。"
}

function show_help() {
  cat << EOF
集合测试框架构建和运行脚本
为华为昇腾平台的集合库提供一站式测试管理工具

用法: $0 [选项] [参数]

选项:
  -h, --help              显示此帮助信息
  -m, --mode MODE         设置编译模式: default 或 kernel (默认: default)
  -d, --debug             启用调试模式
  -c, --clean             清理构建目录
  -b, --build             构建整个项目（包含清理）
  -r, --run [PATTERN]     运行所有测试
  -a, --all               清理、构建并运行所有测试（完整流程）
  -p, --performance       构建性能测试
  -rp, --run-performance  运行性能测试
  -doc, --document        生成项目文档
  --ascend-home PATH      设置 ASCEND_HOME_PATH 路径
  --test-name NAME        指定要运行的测试可执行文件名或关键字
  --test-pattern PATTERN  指定 Catch2 测试标签过滤模式

示例:
  $0 -c                       # 清理构建目录
  $0 -b                       # 构建整个项目
  $0 -r                       # 运行所有测试
  $0 -a                       # 完整流程：清理、构建、运行测试

  $0 -b -d                    # 调试模式构建
  $0 -b -m kernel             # 使用 kernel 模式构建

  $0 -r --test-name static_map # 运行所有 static_map 相关测试
  $0 -r --test-pattern "[insert]" # 运行所有 insert 标签的测试

  $0 -p                       # 构建性能测试
  $0 -rp                      # 运行性能测试

  $0 -doc                     # 生成项目文档

  $0 -b --ascend-home /usr/local/Ascend/ascend-toolkit/latest
EOF
}

# 设置编译配置
function set_compile_config() {
  case "$1" in
    "default"|"kernel")
      compile_mode="$1"
      echo "[INFO] 设置编译模式为: $compile_mode"
      ;;
    "--debug")
      need_debug="--debug"
      echo "[INFO] 启用调试模式"
      ;;
    *)
      echo "错误: 参数 '$1' 无效。支持的参数: default, kernel, --debug"
      exit 1
      ;;
  esac
}

function resolve_cce_compiler() {
  local ascend_home_real
  ascend_home_real="$(readlink -f -- "${ascend_home_path}" 2>/dev/null || true)"
  if [ -z "${ascend_home_real}" ] || [ ! -d "${ascend_home_real}" ]; then
    echo "错误: ASCEND_HOME_PATH 不是有效的 CANN 安装目录: ${ascend_home_path}" >&2
    return 1
  fi

  local compiler_dirs=(
    "${ascend_home_path}/compiler/ccec_compiler/bin"
    "${ascend_home_path}/tools/ccec_compiler/bin"
    "${ascend_home_path}/toolkit/tools/ccec_compiler/bin"
  )
  # Atlas 950 toolkits expose Bisheng through ccec_compiler/bin. Prefer that
  # frontend when present, and keep ccec as the legacy CANN fallback.
  local compiler_names=(bisheng ccec)

  local candidates=()
  local compiler_dir compiler_name compiler compiler_real
  for compiler_name in "${compiler_names[@]}"; do
    for compiler_dir in "${compiler_dirs[@]}"; do
      compiler="${compiler_dir}/${compiler_name}"
      candidates+=("${compiler}")
      if [ -f "${compiler}" ] && [ -x "${compiler}" ]; then
        compiler_real="$(readlink -f -- "${compiler}" 2>/dev/null || true)"
        case "${compiler_real}" in
          "${ascend_home_real}"/*)
            ;;
          *)
            echo "错误: 编译器实际路径不属于 ASCEND_HOME_PATH。" >&2
            echo "  入口: ${compiler}" >&2
            echo "  实际: ${compiler_real:-<无法解析>}" >&2
            echo "  CANN: ${ascend_home_real}" >&2
            return 1
            ;;
        esac
        printf '%s\n' "${compiler}"
        return 0
      fi
    done
  done

  echo "错误: ASCEND_HOME_PATH 下未找到 Bisheng 或 ccec；已检查:" >&2
  printf '  - %s\n' "${candidates[@]}" >&2
  echo "请通过 --ascend-home 指定包含编译器、头文件和库的同一 CANN 安装目录。" >&2
  return 1
}

function ensure_compiler_cache_matches() {
  local target_build_dir="$1"
  local requested_compiler="$2"
  local requested_mode="$3"
  local cache_file="${target_build_dir}/CMakeCache.txt"
  if [ ! -f "${cache_file}" ]; then
    return 0
  fi

  local cached_compiler
  cached_compiler="$(
    sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "${cache_file}" | head -n 1
  )"
  if [ -z "${cached_compiler}" ]; then
    return 0
  fi

  local cached_mode
  case "$(basename -- "${cached_compiler}")" in
    bisheng|ccec)
      cached_mode="$(basename -- "${cached_compiler}")"
      ;;
    *)
      cached_mode="unknown"
      ;;
  esac

  local requested_real cached_real
  requested_real="$(
    readlink -f -- "${requested_compiler}" 2>/dev/null ||
      printf '%s' "${requested_compiler}"
  )"
  cached_real="$(
    readlink -f -- "${cached_compiler}" 2>/dev/null ||
      printf '%s' "${cached_compiler}"
  )"
  if [ "${requested_real}" != "${cached_real}" ] ||
     [ "${cached_mode}" != "${requested_mode}" ]; then
    echo "错误: 构建目录已缓存其他 C++ 编译器或编译模式。" >&2
    echo "  缓存: ${cached_compiler} (mode=${cached_mode})" >&2
    echo "  请求: ${requested_compiler} (mode=${requested_mode})" >&2
    echo "切换编译器前请先运行: bash scripts/build.sh -c" >&2
    return 1
  fi
}

# 编译测试框架
function compile_testframework() {
  echo "[INFO] 开始构建测试框架..."
  mkdir -p "${ROOT_DIR}/build"

  if [ "$compile_mode" == "kernel" ]; then
    echo "[INFO] kernel 模式，跳过普通构建。"
    return 0
  fi

  if [ "$need_debug" == "--debug" ]; then
    debug_flag="DEBUG"
  else
    debug_flag=""
  fi

  echo "[INFO] 构建完成。"
}

# 构建 Catch2
function build_catch2() {
  echo "[INFO] 构建 Catch2..."

  mkdir -p "${catch2_src}"
  if [ ! -d "${catch2_src}/.git" ]; then
    echo "[3rdparty] 克隆 Catch2 到 ${catch2_src}"
    git clone --depth 1 --branch v3.5.4 https://github.com/catchorg/Catch2.git "${catch2_src}"
  fi

  mkdir -p "${catch2_build}" "${catch2_install}"

  echo "[Catch2] 使用 g++ 构建/安装 -> ${catch2_install}"
  cmake -S "${catch2_src}" -B "${catch2_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_INSTALL_PREFIX="${catch2_install}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

  cmake --build "${catch2_build}" -j"$(nproc)"
  cmake --install "${catch2_build}"

  # 确定 Catch2 cmake 目录
  if [ -d "${catch2_install}/lib/cmake/Catch2" ]; then
    catch2_dir="${catch2_install}/lib/cmake/Catch2"
  elif [ -d "${catch2_install}/lib64/cmake/Catch2" ]; then
    catch2_dir="${catch2_install}/lib64/cmake/Catch2"
  else
    echo "错误: 在 ${catch2_install} 中找不到 Catch2 cmake 包目录"
    exit 1
  fi

  echo "[INFO] Catch2 构建完成。"
}

# 构建主项目
function build_main_project() {
  echo "[INFO] 构建主项目..."

  if [ -z "${ascend_home_path}" ]; then
    echo "错误: ASCEND_HOME_PATH 未设置。"
    echo "示例:"
    echo "  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest"
    echo "或使用: $0 --ascend-home /path/to/ascend"
    exit 1
  fi

  local cce_compiler_path
  cce_compiler_path="$(resolve_cce_compiler)"
  local cce_compiler_mode
  cce_compiler_mode="$(basename -- "${cce_compiler_path}")"
  local cce_compiler_real_path
  cce_compiler_real_path="$(
    readlink -f -- "${cce_compiler_path}" 2>/dev/null || printf '%s' "${cce_compiler_path}"
  )"
  ensure_compiler_cache_matches \
    "${build_dir}" "${cce_compiler_path}" "${cce_compiler_mode}"
  mkdir -p "${build_dir}"

  echo "[Main] 使用 CCE 编译器: ${cce_compiler_path}"
  echo "[Main] CCE 编译器实际路径: ${cce_compiler_real_path}"
  echo "[Main] 编译器模式: ${cce_compiler_mode}"
  if [ "${cce_compiler_mode}" = "bisheng" ]; then
    echo "[Main] 目标架构: ${bisheng_aicore_arch}"
  else
    echo "[Main] 目标架构: ${ccec_aicore_arch}"
  fi
  echo "[Main] 配置项目 (Catch2_DIR=${catch2_dir})"
  cmake -S "${ROOT_DIR}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${cce_compiler_path}" \
    -DCatch2_DIR="${catch2_dir}" \
    -DASCEND_HOME_PATH="${ascend_home_path}" \
    -DCCE_AICORE_ARCH="${ccec_aicore_arch}" \
    -DBISHENG_AICORE_ARCH="${bisheng_aicore_arch}" \
    -DBUILD_TESTS=ON \
    -DROARING_BITMAP_GENERATE_TESTDATA=ON

  echo "[Main] 使用 CCE 编译器构建"
  cmake --build "${build_dir}" -j"$(nproc)"

  echo "[INFO] 主项目构建完成。"
}

function build_performance_project() {
  echo "[INFO] 构建性能测试..."

  if [ -z "${ascend_home_path}" ]; then
    echo "错误: ASCEND_HOME_PATH 未设置。"
    echo "示例:"
    echo "  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest"
    echo "或使用: $0 --ascend-home /path/to/ascend"
    exit 1
  fi

  local cce_compiler_path
  cce_compiler_path="$(resolve_cce_compiler)"
  local cce_compiler_mode
  cce_compiler_mode="$(basename -- "${cce_compiler_path}")"
  local cce_compiler_real_path
  cce_compiler_real_path="$(
    readlink -f -- "${cce_compiler_path}" 2>/dev/null || printf '%s' "${cce_compiler_path}"
  )"
  perf_build_dir="${ROOT_DIR}/build/performance"
  ensure_compiler_cache_matches \
    "${perf_build_dir}" "${cce_compiler_path}" "${cce_compiler_mode}"
  mkdir -p "${perf_build_dir}"

  echo "[Performance] 使用 CCE 编译器: ${cce_compiler_path}"
  echo "[Performance] CCE 编译器实际路径: ${cce_compiler_real_path}"
  echo "[Performance] 编译器模式: ${cce_compiler_mode}"
  if [ "${cce_compiler_mode}" = "bisheng" ]; then
    echo "[Performance] 目标架构: ${bisheng_aicore_arch}"
  else
    echo "[Performance] 目标架构: ${ccec_aicore_arch}"
  fi
  echo "[Performance] 配置项目"
  cmake -S "${ROOT_DIR}" -B "${perf_build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${cce_compiler_path}" \
    -DASCEND_HOME_PATH="${ascend_home_path}" \
    -DCCE_AICORE_ARCH="${ccec_aicore_arch}" \
    -DBISHENG_AICORE_ARCH="${bisheng_aicore_arch}" \
    -DBUILD_PERFORMANCE=ON \
    -DBUILD_TESTS=OFF \
    -DROARING_BITMAP_GENERATE_TESTDATA=ON

  echo "[Performance] 使用 CCE 编译器构建"
  cmake --build "${perf_build_dir}" -j"$(nproc)"

  echo "[INFO] 性能测试构建完成，可执行文件在 build/performance/ 下。"
}

# 运行性能测试
function run_performance_test() {
  local perf_bin_dir="${ROOT_DIR}/build/performance"
  local test_count=0

  if [ ! -d "${perf_bin_dir}" ]; then
    echo "ERROR: ${perf_bin_dir} not found. 先运行构建: $0 -p"
    exit 1
  fi

  local bins=()
  while IFS= read -r -d '' file; do
    bins+=("$file")
  done < <(find "${perf_bin_dir}" -type f -name "*_perf_*" -print0)

  if [ "${#bins[@]}" -eq 0 ]; then
    echo "ERROR: no performance test binaries found in ${perf_bin_dir}"
    exit 1
  fi

  echo "[INFO] 找到 ${#bins[@]} 个性能测试文件"
  for b in "${bins[@]}"; do
    if [ ! -x "${b}" ]; then
      echo "ERROR: performance test binary is not executable: ${b}" >&2
      return 1
    fi
    local test_name="${b#${perf_bin_dir}/}"
    echo "==> RUN: ${test_name}"
    "${b}"
    test_count=$((test_count + 1))
  done
  echo ""
  echo "==> 完成 ${test_count} 个性能测试用例"
}


# 运行测试
function run_tests() {
  local SEL="${1:-}"
  local PATTERN="${2:-}"

  if [ ! -d "${bin_dir}" ]; then
    echo "ERROR: ${bin_dir} not found. 先运行构建: $0 build"
    exit 1
  fi

  shopt -s nullglob
  bins=()

  if [ -z "${SEL}" ]; then
    bins=( "${bin_dir}/collection_tests_"* )
  else
    if [ -x "${bin_dir}/${SEL}" ]; then
      bins=( "${bin_dir}/${SEL}" )
    else
      bins=( "${bin_dir}/collection_tests_"*"${SEL}"* )
    fi
  fi

  if [ "${#bins[@]}" -eq 0 ]; then
    echo "ERROR: no test binaries matched."
    echo "  BIN_DIR=${bin_dir}"
    echo "  SEL=${SEL}"
    echo "  可执行示例：collection_tests_static_map_insert_test"
    exit 1
  fi

  local passed=0
  local failed=0
  local skipped=0
  local failed_names=()
  local skipped_names=()

  for b in "${bins[@]}"; do
    if [ ! -x "${b}" ]; then
      echo "==> FAIL: $(basename "${b}") (not executable)"
      failed=$((failed + 1))
      failed_names+=("$(basename "${b}")")
      continue
    fi

    if [ -n "${PATTERN}" ]; then
      set +e
      test_output=$("${b}" "${PATTERN}" 2>&1)
      rc=$?
      set -e
      if echo "${test_output}" | grep -qE "No tests ran"; then
        echo "  [SKIP] $(basename "${b}") — no matching tests"
        skipped=$((skipped + 1))
        skipped_names+=("$(basename "${b}")")
        continue
      fi
      echo ""
      echo "==> RUN: $(basename "${b}") ${PATTERN}"
      printf '%s\n' "${test_output}"
      if [ "${rc}" -eq 0 ]; then
        echo "==> PASS: $(basename "${b}")"
        passed=$((passed + 1))
      else
        echo "==> FAIL: $(basename "${b}") (exit code: ${rc})"
        failed=$((failed + 1))
        failed_names+=("$(basename "${b}")")
      fi
    else
      echo ""
      echo "==> RUN: $(basename "${b}")"
      set +e
      "${b}"
      rc=$?
      set -e
      if [ "${rc}" -eq 0 ]; then
        echo "==> PASS: $(basename "${b}")"
        passed=$((passed + 1))
      else
        echo "==> FAIL: $(basename "${b}") (exit code: ${rc})"
        failed=$((failed + 1))
        failed_names+=("$(basename "${b}")")
      fi
    fi
  done

  echo ""
  echo "=============================================="
  echo "  测试汇总：共 $((passed + failed)) 个，通过 ${passed} 个，失败 ${failed} 个，跳过 ${skipped} 个"
  if [ "${failed}" -gt 0 ]; then
    echo "  失败列表："
    for fn in "${failed_names[@]}"; do
      echo "    - ${fn}"
    done
  fi
  if [ "${skipped}" -gt 0 ]; then
    echo "  跳过列表："
    for sn in "${skipped_names[@]}"; do
      echo "    - ${sn}"
    done
  fi
  echo "=============================================="

  if [ "${failed}" -gt 0 ]; then
    exit 1
  fi

  if [ "${passed}" -eq 0 ] && [ "${failed}" -eq 0 ] && [ "${skipped}" -gt 0 ]; then
    echo "ERROR: 所有测试都被跳过，未匹配到任何测试用例。请检查 --test-pattern 拼写。"
    exit 1
  fi
}

function setup_doxygen() {
  if command -v doxygen &> /dev/null; then
    local system_doxygen_version
    system_doxygen_version="$(doxygen --version)"
    if [ "${system_doxygen_version%% *}" = "1.9.8" ]; then
      return 0
    fi
  fi
  if [ -x "${doxygen_install}/bin/doxygen" ]; then
    local local_doxygen_version
    local_doxygen_version="$(${doxygen_install}/bin/doxygen --version)"
    if [ "${local_doxygen_version%% *}" = "1.9.8" ]; then
      export PATH="${doxygen_install}/bin:${PATH}"
      return 0
    fi
  fi
  if [ "$(uname -m)" != "x86_64" ]; then
    echo "[ERROR] 自动安装 Doxygen 仅支持 x86_64，请手动安装 Doxygen 并加入 PATH。"
    return 1
  fi

  mkdir -p "$doxygen_build"
  rm -rf "$doxygen_install"
  mkdir -p "$doxygen_install"
  local archive="${doxygen_build}/doxygen-1.9.8.linux.bin.tar.gz"
  local url="https://github.com/doxygen/doxygen/releases/download/Release_1_9_8/doxygen-1.9.8.linux.bin.tar.gz"
  echo "[INFO] 下载 Doxygen 1.9.8 到本地构建目录..."
  if command -v curl &> /dev/null; then
    curl -fL --retry 3 -o "$archive" "$url"
  elif command -v wget &> /dev/null; then
    wget -O "$archive" "$url"
  else
    echo "[ERROR] 未找到 curl 或 wget，无法下载 Doxygen。"
    return 1
  fi
  tar -xzf "$archive" --strip-components=1 -C "$doxygen_install"
  export PATH="${doxygen_install}/bin:${PATH}"
  command -v doxygen &> /dev/null
}

function setup_graphviz() {
  if command -v dot &> /dev/null && printf 'digraph G { a -> b }\n' | dot -Tpng -o /dev/null 2>/dev/null; then
    return 0
  fi
  if [ -x "${graphviz_install}/usr/bin/dot" ]; then
    export PATH="${graphviz_install}/usr/bin:${PATH}"
    export LD_LIBRARY_PATH="${graphviz_install}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    export GVBINDIR="${graphviz_install}/usr/lib/x86_64-linux-gnu/graphviz"
    dot -c &> /dev/null || true
    if printf 'digraph G { a -> b }\n' | dot -Tpng -o /dev/null 2>/dev/null; then
      return 0
    fi
  fi
  if [ "$(uname -m)" != "x86_64" ] || ! command -v apt-get &> /dev/null || ! command -v dpkg-deb &> /dev/null; then
    return 1
  fi

  mkdir -p "$graphviz_build" "$graphviz_install"
  echo "[INFO] 下载 Graphviz 到本地构建目录..."
  (
    cd "$graphviz_build"
    apt-get download graphviz libann0 libcdt5 libcgraph6 libgts-0.7-5t64 libgvc6 libgvpr2 liblab-gamut1 libpathplan4
  )
  local package_file
  for package_file in "$graphviz_build"/*.deb; do
    dpkg-deb -x "$package_file" "$graphviz_install"
  done
  export PATH="${graphviz_install}/usr/bin:${PATH}"
  export LD_LIBRARY_PATH="${graphviz_install}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export GVBINDIR="${graphviz_install}/usr/lib/x86_64-linux-gnu/graphviz"
  dot -c &> /dev/null
  printf 'digraph G { a -> b }\n' | dot -Tpng -o /dev/null 2>/dev/null
}

function stage_doxygen_markdown() {
  mkdir -p "$doxygen_markdown"
  sed -e 's|开发指导\.md|Development_Guide.md|g' -e 's|\.\./README\.md|index.html|g' \
    "${ROOT_DIR}/docs/API文档和使用示例.md" > "${doxygen_markdown}/API.md"
  cp "${ROOT_DIR}/docs/RoaringBitmap_API文档和使用示例.md" "${doxygen_markdown}/RoaringBitmap_API.md"
  cp "${ROOT_DIR}/docs/BloomFilter_API文档和使用示例.md" "${doxygen_markdown}/BloomFilter_API.md"
  cp "${ROOT_DIR}/docs/DynamicMap_API文档和使用示例.md" "${doxygen_markdown}/DynamicMap_API.md"
  sed -e 's|API文档和使用示例\.md|API.md|g' -e 's|\.\./README\.md|index.html|g' \
    "${ROOT_DIR}/docs/开发指导.md" > "${doxygen_markdown}/Development_Guide.md"
  sed \
    -e 's|docs/API文档和使用示例\.md|API.md|g' \
    -e 's|docs/RoaringBitmap_API文档和使用示例\.md|RoaringBitmap_API.md|g' \
    -e 's|docs/BloomFilter_API文档和使用示例\.md|BloomFilter_API.md|g' \
    -e 's|docs/DynamicMap_API文档和使用示例\.md|DynamicMap_API.md|g' \
    -e 's|docs/开发指导\.md|Development_Guide.md|g' \
    -e 's|docs/images/architecture\.png|architecture.png|g' \
    "${ROOT_DIR}/README.md" | sed -E \
      -e 's|\(API\.md#([^)]+)\)|(@ref autotoc_md\1)|g' \
      -e 's|\(RoaringBitmap_API\.md#([^)]+)\)|(@ref \1)|g' \
      > "${doxygen_markdown}/README.md"
}

function build_documentation() {
  echo "[INFO] 开始生成 Doxygen文档..."
  setup_doxygen

  if ! setup_graphviz; then
    echo "[ERROR] Graphviz 不可用，无法生成关系图。"
    return 1
  fi

  stage_doxygen_markdown
  rm -rf "${ROOT_DIR}/docs/html"
  cd "$ROOT_DIR"
  doxygen doxygen/Doxyfile
  echo "[INFO] Doxygen 文档生成成功，输出到 ${ROOT_DIR}/docs/html"
}

# 主函数
function main() {
  local action=""
  local test_name=""
  local test_pattern=""

  # 解析命令行参数
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        show_help
        exit 0
        ;;
      -m|--mode)
        if [[ -z "$2" ]] || [[ "$2" =~ ^- ]]; then
          echo "错误: --mode 需要一个参数"
          exit 1
        fi
        set_compile_config "$2"
        shift 2
        ;;
      -d|--debug)
        set_compile_config "--debug"
        shift
        ;;
      -b|--build)
        action="build"
        shift
        ;;
      -r|--run)
        action="run"
        shift
        if [[ $# -ge 1 ]] && [[ ! "$1" =~ ^- ]]; then
          if [[ "$1" =~ ^\[ ]]; then
            test_pattern="$1"
            shift
          else
            # 先当作 test_name
            test_name="$1"
            shift

            if [[ $# -ge 1 ]] && [[ ! "$1" =~ ^- ]] && [[ "$1" =~ ^\[ ]]; then
              test_pattern="$1"
              shift
            fi
          fi
        fi
        ;;
      -a|--all)
        action="all"
        shift
        ;;
      -c|--clean)
        action="clean"
        shift
        ;;
      --ascend-home)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          echo "错误: --ascend-home 需要一个参数"
          exit 1
        fi
        ascend_home_path="$2"
        export ASCEND_HOME_PATH="$ascend_home_path"
        shift 2
        ;;
      --test-name)
        if [[ -z "$2" ]]; then
          echo "错误: --test-name 需要一个参数"
          exit 1
        fi
        test_name="$2"
        shift 2
        ;;
      --test-pattern)
        if [[ -z "$2" ]]; then
          echo "错误: --test-pattern 需要一个参数"
          exit 1
        fi
        test_pattern="$2"
        shift 2
        ;;
      -p|--performance)
         action="performance"
         shift
         ;;
      -rp|--run-performance)
         action="run_performance"
         shift
         ;;
      -doc|--document)
         action="document"
         shift
         ;;
      *)
        # 处理未知参数
        echo "错误: 未知选项 '$1'"
        show_help
        exit 1
        ;;
    esac
  done

  # 如果没有指定动作，显示帮助
  if [ -z "$action" ]; then
    show_help
    exit 0
  fi

  case "$action" in
    "clean")
      clean_build
      ;;
    "build")
      clean_build
      compile_testframework
      build_catch2
      build_main_project
      ;;
    "run")
      run_tests "$test_name" "$test_pattern"
      ;;
    "all")
      clean_build
      compile_testframework
      build_catch2
      build_main_project
      run_tests "" ""
      ;;
    "performance")
      build_performance_project
      ;;
    "run_performance")
      run_performance_test
      ;;
    "document")
      build_documentation
      ;;
  esac
}

# 运行主函数
main "$@"
