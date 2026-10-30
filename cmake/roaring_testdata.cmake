# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(ROARING_BITMAP_VERIFY_ONLY)
  if(NOT DEFINED ROARING_BITMAP_TEST_DATA_DIR)
    message(FATAL_ERROR "ROARING_BITMAP_TEST_DATA_DIR is not defined")
  endif()

  function(verify_roaring_file FILE_NAME EXPECTED_SIZE EXPECTED_SHA256)
    set(FILE_PATH "${ROARING_BITMAP_TEST_DATA_DIR}/${FILE_NAME}")
    if(NOT EXISTS "${FILE_PATH}")
      message(FATAL_ERROR "Missing generated RoaringBitmap file: ${FILE_PATH}")
    endif()

    file(SIZE "${FILE_PATH}" ACTUAL_SIZE)
    if(NOT ACTUAL_SIZE EQUAL EXPECTED_SIZE)
      message(FATAL_ERROR
        "Unexpected size for ${FILE_NAME}: expected ${EXPECTED_SIZE}, got ${ACTUAL_SIZE}")
    endif()

    file(SHA256 "${FILE_PATH}" ACTUAL_SHA256)
    if(NOT ACTUAL_SHA256 STREQUAL EXPECTED_SHA256)
      message(FATAL_ERROR
        "Unexpected SHA-256 for ${FILE_NAME}: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}")
    endif()
  endfunction()

  verify_roaring_file(
    "bitmapwithoutruns.bin"
    72616
    "d719ae2e0150a362ef7cf51c361527585891f01460b1a92bcfb6a7257282a442"
  )
  verify_roaring_file(
    "bitmapwithruns.bin"
    48056
    "1f1909bfdd354fa2f0694fe88b8076833ca5383ad9fc3f68f2709c84a2ab70e3"
  )
  verify_roaring_file(
    "portable_bitmap64.bin"
    16506
    "b5a553a759167f5f9ccb3fa21552d943b4c73235635b753376f4faf62067d178"
  )
  return()
endif()

if(NOT ROARING_BITMAP_GENERATE_TESTDATA)
  return()
endif()

set(CROARING_VERSION "v5.1.0")
set(CROARING_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/3rdparty/CRoaring-${CROARING_VERSION}"
    CACHE PATH "Shared directory containing the CRoaring test-data sources")
file(MAKE_DIRECTORY "${CROARING_SOURCE_DIR}")

function(ensure_croaring_file FILE_NAME EXPECTED_SHA256)
  set(FILE_PATH "${CROARING_SOURCE_DIR}/${FILE_NAME}")
  if(NOT EXISTS "${FILE_PATH}")
    set(FILE_URL
      "https://github.com/RoaringBitmap/CRoaring/releases/download/${CROARING_VERSION}/${FILE_NAME}")
    message(STATUS "Downloading CRoaring ${CROARING_VERSION}: ${FILE_NAME}")
    string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef DOWNLOAD_ID)
    set(DOWNLOAD_PATH "${FILE_PATH}.${DOWNLOAD_ID}.tmp")
    file(DOWNLOAD "${FILE_URL}" "${DOWNLOAD_PATH}"
      STATUS DOWNLOAD_STATUS
      TLS_VERIFY ON
    )
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_CODE)
    list(GET DOWNLOAD_STATUS 1 DOWNLOAD_MESSAGE)
    if(NOT DOWNLOAD_CODE EQUAL 0)
      file(REMOVE "${DOWNLOAD_PATH}")
      message(FATAL_ERROR "Failed to download ${FILE_URL}: ${DOWNLOAD_MESSAGE}")
    endif()
    file(SHA256 "${DOWNLOAD_PATH}" DOWNLOADED_SHA256)
    if(NOT DOWNLOADED_SHA256 STREQUAL EXPECTED_SHA256)
      file(REMOVE "${DOWNLOAD_PATH}")
      message(FATAL_ERROR
        "Unexpected SHA-256 for ${FILE_URL}: expected ${EXPECTED_SHA256}, got ${DOWNLOADED_SHA256}")
    endif()
    file(RENAME "${DOWNLOAD_PATH}" "${FILE_PATH}")
  endif()

  file(SHA256 "${FILE_PATH}" ACTUAL_SHA256)
  if(NOT ACTUAL_SHA256 STREQUAL EXPECTED_SHA256)
    message(FATAL_ERROR
      "Unexpected SHA-256 for ${FILE_PATH}: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}")
  endif()
endfunction()

ensure_croaring_file(
  "roaring.c"
  "8d06c57053611cee90342ebbbacee38effc0fe477105b05e604a3daa770e1fd4"
)
ensure_croaring_file(
  "roaring.h"
  "041cae6d0501b642936b4a5911aebc7c1945cd7d4d83cef11a3d09f134436532"
)
configure_file(
  "${CROARING_SOURCE_DIR}/roaring.c"
  "${CROARING_SOURCE_DIR}/roaring.cpp"
  COPYONLY
)

set(ROARING_BITMAP_TEST_DATA_DIR "${PROJECT_SOURCE_DIR}/tests/testdata")
set(ROARING_BITMAP_TEST_DATA_FILES
  "${ROARING_BITMAP_TEST_DATA_DIR}/bitmapwithoutruns.bin"
  "${ROARING_BITMAP_TEST_DATA_DIR}/bitmapwithruns.bin"
  "${ROARING_BITMAP_TEST_DATA_DIR}/portable_bitmap64.bin"
)

add_library(roaring_testdata_croaring STATIC
  "${CROARING_SOURCE_DIR}/roaring.cpp"
)
target_include_directories(roaring_testdata_croaring PUBLIC "${CROARING_SOURCE_DIR}")

add_executable(roaring_bitmap_testdata_generator
  "${PROJECT_SOURCE_DIR}/tests/utility/generate_roaring_testdata.cpp"
)
target_link_libraries(roaring_bitmap_testdata_generator PRIVATE roaring_testdata_croaring stdc++ m)

add_custom_command(
  OUTPUT ${ROARING_BITMAP_TEST_DATA_FILES}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ROARING_BITMAP_TEST_DATA_DIR}"
  COMMAND $<TARGET_FILE:roaring_bitmap_testdata_generator>
          --output-dir "${ROARING_BITMAP_TEST_DATA_DIR}"
  COMMAND ${CMAKE_COMMAND}
          -DROARING_BITMAP_VERIFY_ONLY=ON
          -DROARING_BITMAP_TEST_DATA_DIR=${ROARING_BITMAP_TEST_DATA_DIR}
          -P "${PROJECT_SOURCE_DIR}/cmake/roaring_testdata.cmake"
  DEPENDS roaring_bitmap_testdata_generator
  COMMENT "Generating RoaringBitmap portable test data"
  VERBATIM
)

add_custom_target(generate_roaring_testdata ALL
  COMMAND ${CMAKE_COMMAND}
          -DROARING_BITMAP_VERIFY_ONLY=ON
          -DROARING_BITMAP_TEST_DATA_DIR=${ROARING_BITMAP_TEST_DATA_DIR}
          -P "${PROJECT_SOURCE_DIR}/cmake/roaring_testdata.cmake"
  DEPENDS ${ROARING_BITMAP_TEST_DATA_FILES}
  VERBATIM
)
