#pragma once

#include "object.hpp"
#include "code_module.hpp"
#include "value.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
namespace lmx::runtime {

// 运行时闭包对象
struct ClosureObj : Object {
    FuncObj*            func;
    uint32_t            cap_count;
    Value               caps[1];

    explicit ClosureObj(FuncObj* f, uint32_t cnt, const Value* capture_values) noexcept
        : Object(ObjectKind::Closure), func(f), cap_count(cnt) {
        if (cnt > 0 && capture_values) {
            for (uint32_t i = 0; i < cnt; ++i) {
                ::new (static_cast<void*>(&caps[i])) Value(capture_values[i]);
            }
        }
    }

    ~ClosureObj() noexcept {
        for (uint32_t i = 0; i < cap_count; ++i) {
            caps[i].~Value();
        }
    }

    static ClosureObj* make(FuncObj* f, uint32_t cnt, const Value* capture_values) noexcept {
        if (cnt == 0) {
            // 0 capture 仍构造一个 closure 对象，便于统一调用
            void* p = std::malloc(sizeof(ClosureObj));
            if (!p) return nullptr;
            return new (p) ClosureObj(f, 0, nullptr);
        }
        const size_t extra_bytes = (cnt - 1) * sizeof(Value);
        void* p = std::malloc(sizeof(ClosureObj) + extra_bytes);
        if (!p) return nullptr;
        return new (p) ClosureObj(f, cnt, capture_values);
    }

    // placement new 兼容 GC
    void* operator new(size_t /*size*/, void* p) noexcept { return p; }
    void operator delete(void* p) noexcept { std::free(p); }
};

} // namespace lmx::runtime
