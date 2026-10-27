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

#include <iostream>
#include <stdexcept>
#include <string>

#include <acl/acl.h>
#include "macros.h"

COLLECTION_HOST void CheckRet(uint32_t ret, const char* funcName) {
  if (ret != 0) {
    std::cout << funcName << " execute failed! ret:" << ret << "\n";
  }
}

COLLECTION_HOST void CheckRetOrThrow(uint32_t ret, const char* funcName) {
  if (ret != 0) {
    throw std::runtime_error(std::string(funcName) + " execute failed! ret:" + std::to_string(ret));
  }
}
