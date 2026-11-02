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

#include <acl/acl.h>

namespace aclco::test
{

inline int32_t GetDeviceId()
{
  const char* v = std::getenv("COLLECTION_TEST_DEVICE_ID");
  if (v == nullptr || v[0] == '\0') {
    return 0;
  }
  return std::atoi(v);
}

inline void CheckAcl(uint32_t ret, const char* what)
{
  if (ret != ACL_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed, ret=" + std::to_string(ret));
  }
}

// Catch2 invokes each generated case/section separately. Keep the runtime alive
// across those invocations: repeatedly resetting a device waits for TsdClose and
// can dominate or stall otherwise completed tests. Streams and allocations still
// retain their per-test lifetimes; the device is reset once at process exit.
struct AclProcessSession
{
  int32_t device;

  AclProcessSession() : device(GetDeviceId())
  {
    CheckAcl(aclInit(nullptr), "aclInit");
    CheckAcl(aclrtSetDevice(device), "aclrtSetDevice");
  }

  ~AclProcessSession()
  {
    (void)aclrtResetDevice(device);
    (void)aclFinalize();
  }

  AclProcessSession(AclProcessSession const&) = delete;
  AclProcessSession& operator=(AclProcessSession const&) = delete;
};

struct AclGlobalGuard
{
  AclGlobalGuard()
  {
    static AclProcessSession session;
  }

  ~AclGlobalGuard() = default;
  AclGlobalGuard(AclGlobalGuard const&) = delete;
  AclGlobalGuard& operator=(AclGlobalGuard const&) = delete;
};

struct AclStreamGuard
{
  aclrtStream stream{nullptr};

  AclStreamGuard()
  {
    CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
  }

  ~AclStreamGuard()
  {
    if (stream) {
      (void)aclrtSynchronizeStream(stream);
      (void)aclrtDestroyStream(stream);
      stream = nullptr;
    }
  }

  AclStreamGuard(const AclStreamGuard&) = delete;
  AclStreamGuard& operator=(const AclStreamGuard&) = delete;
};

inline void Sync(aclrtStream stream)
{
  CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
}

} // namespace aclco::test