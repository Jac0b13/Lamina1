
#pragma once
#include "mir.hpp"

namespace lmx::mir {
bool is_int_type(const Type *type) noexcept;
bool is_frac_type(const Type *type) noexcept;
bool is_float_type(const Type *type) noexcept;

class MirBuilder {
public:
    static MirModule from_ast_module(const std::shared_ptr<Module>& ast);
};

}
