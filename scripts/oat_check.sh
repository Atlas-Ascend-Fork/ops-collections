# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
exit_code=0
for f in "$@"; do
    case "$f" in
        *.c|*.cc|*.cpp|*.h|*.hpp|*.inl|*.py|*.sh|*.xml)
            head -n 15 "$f" | grep -q "Copyright" || { echo "OAT: missing copyright notice: $f"; exit_code=1; }
            head -n 15 "$f" | grep -q "CANN Open Software License Agreement" || { echo "OAT: missing CANN license header: $f"; exit_code=1; }
            ;;
    esac
done
exit $exit_code
