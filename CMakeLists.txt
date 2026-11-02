# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

project(CUCOLLECTION LANGUAGES CXX)

option(BUILD_TESTS "Build Catch2-based tests" ON)
option(BUILD_PERFORMANCE "Build performance test" OFF)
option(ROARING_BITMAP_GENERATE_TESTDATA "Generate RoaringBitmap portable test data" ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CCE_AICORE_ARCH "dav-c310"
    CACHE STRING "CCE aicore arch for ccec (--cce-aicore-arch=...)")
set(BISHENG_AICORE_ARCH "dav-3510"
    CACHE STRING "NPU arch for Bisheng (--npu-arch=...)")

function(cucollection_enable_cce TARGET)
  if(CMAKE_CXX_COMPILER MATCHES "ccec$")
    target_compile_options(${TARGET} PRIVATE
      -xcce
      --cce-aicore-arch=${CCE_AICORE_ARCH}
      -Wno-macro-redefined
    )
  elseif(CMAKE_CXX_COMPILER MATCHES "bisheng$")
    target_compile_options(${TARGET} PRIVATE
      "SHELL:-x asc"
      --npu-arch=${BISHENG_AICORE_ARCH}
      -Wno-macro-redefined
    )
  endif()
endfunction()

if(NOT ASCEND_HOME_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
  endif()
endif()

add_library(COLLECTION INTERFACE)

target_include_directories(COLLECTION INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ASCEND_HOME_PATH)
  target_include_directories(COLLECTION INTERFACE
    ${ASCEND_HOME_PATH}/acllib/include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include
    ${ASCEND_HOME_PATH}/include/experiment/msp
    ${ASCEND_HOME_PATH}/include/prof
  )

  target_link_directories(COLLECTION INTERFACE
    ${ASCEND_HOME_PATH}/runtime/lib64
    ${ASCEND_HOME_PATH}/lib64
  )

  target_link_libraries(COLLECTION INTERFACE
    dl
    runtime
    ascendcl
    tiling_api
    platform
    unified_dlog
  )
endif()

if(BUILD_TESTS OR BUILD_PERFORMANCE)
  enable_testing()
endif()

if(BUILD_TESTS)

  if(NOT Catch2_DIR)
    message(FATAL_ERROR
      "Catch2_DIR is not set. Please build/install Catch2 with g++ first "
      "(use scripts/build.sh -b), then rerun CMake with "
      "-DCatch2_DIR=<catch2_install>/lib/cmake/Catch2."
    )
  endif()

  find_package(Catch2 3 CONFIG REQUIRED)
endif()

if(BUILD_TESTS OR BUILD_PERFORMANCE)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/roaring_testdata.cmake)
  add_subdirectory(tests)
endif()
