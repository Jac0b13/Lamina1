
#include "type_checker.hpp"

#include <algorithm>
#include <functional>
#include <ranges>
#include <limits>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include "../ast/ast.hpp"
#include "../error.hpp"
#include "../mir/mir_builder.hpp"
using namespace lmx;
using namespace lmx::hir;


using namespace lmx::mir;
namespace lmx::hir{

static thread_local std::unordered_set<uint64_t> g_mu_head_ids;

static bool is_recursive_mu_head(TypeVariable* tv) noexcept {
    if (!tv) return false;
    uint64_t id = tv->id;
    if (g_mu_head_ids.count(id)) return true;
    if (!tv->binding) return false;

    std::unordered_set<const Type*> seen_structural;
    std::unordered_set<uint64_t> on_stack_tv;
    std::unordered_set<uint64_t> visited_tv;
    bool found_cycle = false;

    std::function<void(const std::shared_ptr<Type>&)> go;
    std::function<void(TypeVariable*)> go_tv;

    go_tv = [&](TypeVariable* x) {
        if (!x) return;
        uint64_t xid = x->id;
        if (found_cycle) return;
        if (on_stack_tv.count(xid)) { found_cycle = true; return; }
        if (visited_tv.count(xid)) return;
        on_stack_tv.insert(xid);
        visited_tv.insert(xid);
        if (x->binding) go(x->binding);
        on_stack_tv.erase(xid);
    };
    go = [&](const std::shared_ptr<Type>& x) {
        if (found_cycle) return;
        auto r = x;
        while (r && r->kind == TypeKind::TypeVariable) {
            auto* rv = static_cast<TypeVariable*>(r.get());
            go_tv(rv);
            if (found_cycle) return;
            if (!rv->binding) break;
            r = rv->binding;
        }
        if (!r) return;
        if (r->kind == TypeKind::TypeVariable) return;
        if (!seen_structural.insert(r.get()).second) return;
        switch (r->kind) {
        case TypeKind::Function: {
            auto f = std::static_pointer_cast<FunctionType>(r);
            for (const auto& p: f->params_ty) go(p);
            if (found_cycle) return;
            go(f->ret_ty);
            return;
        }
        case TypeKind::LambdaFunction: {
            auto f = std::static_pointer_cast<LambdaFunctionType>(r);
            for (const auto& p: f->params_ty) go(p);
            if (found_cycle) return;
            go(f->ret_ty);
            if (found_cycle) return;
            for (const auto& c: f->capture_tys) go(c);
            return;
        }
        case TypeKind::NativeFunction: {
            auto f = std::static_pointer_cast<NativeFunctionType>(r);
            for (const auto& p: f->params_ty) go(p);
            if (found_cycle) return;
            go(f->ret_ty);
            return;
        }
        case TypeKind::Array: go(std::static_pointer_cast<ArrayType>(r)->type); return;
        case TypeKind::Tuple: {
            auto f = std::static_pointer_cast<TupleType>(r);
            for (const auto& e: f->tys) { go(e); if (found_cycle) return; }
            return;
        }
        case TypeKind::Nullable: go(std::static_pointer_cast<NullableType>(r)->value_type); return;
        case TypeKind::Named: {
            auto f = std::static_pointer_cast<NamedType>(r);
            for (const auto& a: f->args) { go(a); if (found_cycle) return; }
            return;
        }
        default: return;
        }
    };

    go_tv(tv);
   
    if (found_cycle) {
        g_mu_head_ids.insert(id);
        for (uint64_t v: visited_tv) g_mu_head_ids.insert(v);
    }
    return found_cycle;
}

static void mark_mu_head(uint64_t id) noexcept {
    g_mu_head_ids.insert(id);
}

std::shared_ptr<Type> resolve_hm(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return type;
    if (type->kind == TypeKind::TypeVariable) {
        auto tv = std::static_pointer_cast<TypeVariable>(type);
        if (tv->binding) {
            if (is_recursive_mu_head(tv.get())) return type;
            auto repr = resolve_hm(tv->binding);
            if (!repr || repr->kind != TypeKind::TypeVariable ||
                !is_recursive_mu_head(std::static_pointer_cast<TypeVariable>(repr).get())) {
                tv->binding = repr;
            }
            return repr;
        }
        return type;
    }
    return type;
}
static std::pair<std::shared_ptr<Type>, std::vector<TypeVariable*>>
freeze_scheme_monotype(
    const std::shared_ptr<Type>& mono,
    const std::vector<TypeVariable*>& quantified_old
) noexcept {
    auto raw_resolve_tv = [](TypeVariable* tv) -> TypeVariable* {
        if (!tv) return nullptr;
        auto cur = tv;
        while (cur->binding && cur->binding->kind == TypeKind::TypeVariable) {
            auto next = static_cast<TypeVariable*>(cur->binding.get());
            if (next->binding && next->binding->kind == TypeKind::TypeVariable) {
                cur->binding = next->binding;
            }
            cur = next;
        }
        return cur;
    };

    auto is_recursive_bound_tv = [](TypeVariable* t) -> bool {
        return is_recursive_mu_head(t);
    };

    std::vector<TypeVariable*> qold_repr_order;
    qold_repr_order.reserve(quantified_old.size());
    std::unordered_set<TypeVariable*> qold_repr_set;
    qold_repr_set.reserve(quantified_old.size());
    for (auto* qold : quantified_old) {
        if (!qold) continue;
        auto* repr = raw_resolve_tv(qold);
        qold_repr_order.push_back(repr);
        qold_repr_set.insert(repr);
    }

    {
        std::unordered_set<const Type*> seen1;
        std::function<void(const std::shared_ptr<Type>&)> collect_raw;
        auto rec_tv_raw = [&](TypeVariable* tv) {
            if (!tv) return;
            if (!seen1.insert(tv).second) return;
            bool is_q = qold_repr_set.count(tv);
            bool is_rec = !is_q && is_recursive_bound_tv(tv);
            if (tv->binding) collect_raw(tv->binding);
        };
        collect_raw = [&](const std::shared_ptr<Type>& t) {
            if (!t) return;
            auto r = t;
            while (r && r->kind == TypeKind::TypeVariable) {
                auto* rv = static_cast<TypeVariable*>(r.get());
                rec_tv_raw(rv);
                if (!rv->binding) return;
                r = rv->binding;
            }
            if (!r) return;
            if (r->kind == TypeKind::TypeVariable) return;
            if (!seen1.insert(r.get()).second) return;
            switch (r->kind) {
            case TypeKind::Function: {
                auto f = std::static_pointer_cast<FunctionType>(r);
                for (const auto& p: f->params_ty) collect_raw(p);
                collect_raw(f->ret_ty);
                return;
            }
            case TypeKind::LambdaFunction: {
                auto f = std::static_pointer_cast<LambdaFunctionType>(r);
                for (const auto& p: f->params_ty) collect_raw(p);
                collect_raw(f->ret_ty);
                for (const auto& c: f->capture_tys) collect_raw(c);
                return;
            }
            case TypeKind::NativeFunction: {
                auto f = std::static_pointer_cast<NativeFunctionType>(r);
                for (const auto& p: f->params_ty) collect_raw(p);
                collect_raw(f->ret_ty);
                return;
            }
            case TypeKind::Array: collect_raw(std::static_pointer_cast<ArrayType>(r)->type); return;
            case TypeKind::Tuple: {
                auto f = std::static_pointer_cast<TupleType>(r);
                for (const auto& e: f->tys) collect_raw(e);
                return;
            }
            case TypeKind::Nullable: collect_raw(std::static_pointer_cast<NullableType>(r)->value_type); return;
            case TypeKind::Named: {
                auto f = std::static_pointer_cast<NamedType>(r);
                for (const auto& a: f->args) collect_raw(a);
                return;
            }
            default: return;
            }
        };
        collect_raw(mono);
    }

    std::unordered_map<TypeVariable*, std::shared_ptr<TypeVariable>> tv_map;
    std::vector<std::pair<std::shared_ptr<TypeVariable>, std::shared_ptr<Type>>> delayed_bindings;

    std::function<std::shared_ptr<Type>(const std::shared_ptr<Type>&)> copy;
    copy = [&](const std::shared_ptr<Type>& t) -> std::shared_ptr<Type> {
        if (!t) return t;
        auto repr = resolve_hm(t);
        if (repr->kind == TypeKind::TypeVariable) {
            auto* key = std::static_pointer_cast<TypeVariable>(repr).get();
            auto it = tv_map.find(key);
            if (it != tv_map.end()) {
                return std::static_pointer_cast<Type>(it->second);
            }
            bool is_q = (qold_repr_set.count(key) != 0);
            bool is_rec = !is_q && is_recursive_bound_tv(key);
            if (!is_q && !is_rec) {
                return repr;
            }
            auto ntv = type_pool.fresh_type_variable();
            tv_map.emplace(key, ntv);
            bool self_is_mu = is_rec || (is_q && is_recursive_bound_tv(key));
            if (self_is_mu) {
                mark_mu_head(ntv->id);
            }
            if (is_rec && key->binding) {
                delayed_bindings.emplace_back(ntv, key->binding);
            }
            return std::static_pointer_cast<Type>(ntv);
        }
        switch (repr->kind) {
        case TypeKind::Function: {
            auto f = std::static_pointer_cast<FunctionType>(repr);
            std::vector<std::shared_ptr<Type>> ps;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(copy(p));
            return type_pool.function(std::move(ps), copy(f->ret_ty));
        }
        case TypeKind::LambdaFunction: {
            auto f = std::static_pointer_cast<LambdaFunctionType>(repr);
            std::vector<std::shared_ptr<Type>> ps, cs;
            ps.reserve(f->params_ty.size());
            cs.reserve(f->capture_tys.size());
            for (const auto& p : f->params_ty) ps.push_back(copy(p));
            for (const auto& c : f->capture_tys) cs.push_back(copy(c));
            return type_pool.lambda_function(std::move(ps), copy(f->ret_ty), std::move(cs));
        }
        case TypeKind::NativeFunction: {
            auto f = std::static_pointer_cast<NativeFunctionType>(repr);
            std::vector<std::shared_ptr<Type>> ps;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(copy(p));
            return type_pool.native_function(std::move(ps), copy(f->ret_ty), f->name);
        }
        case TypeKind::Array:
            return type_pool.array(copy(std::static_pointer_cast<ArrayType>(repr)->type));
        case TypeKind::Tuple: {
            auto tp = std::static_pointer_cast<TupleType>(repr);
            std::vector<std::shared_ptr<Type>> vs;
            vs.reserve(tp->tys.size());
            for (const auto& e : tp->tys) vs.push_back(copy(e));
            return type_pool.tuple(std::move(vs));
        }
        case TypeKind::Nullable:
            return type_pool.nullable(copy(std::static_pointer_cast<NullableType>(repr)->value_type));
        case TypeKind::Named: {
            auto n = std::static_pointer_cast<NamedType>(repr);
            std::vector<std::shared_ptr<Type>> as;
            as.reserve(n->args.size());
            for (const auto& a : n->args) as.push_back(copy(a));
            return type_pool.named(n->name, std::move(as));
        }
        default:
            return repr;
        }
    };
    auto frozen_mono = copy(mono);
    (void)delayed_bindings; // placeholder for compile warning; intentionally not filling.

    std::vector<TypeVariable*> new_quantified;
    new_quantified.reserve(qold_repr_order.size());
    for (auto* repr_raw : qold_repr_order) {
        auto it = tv_map.find(repr_raw);
        if (it != tv_map.end()) {
            new_quantified.push_back(it->second.get());
        }
    }
    return { std::move(frozen_mono), std::move(new_quantified) };
}

static bool occurs_check_polar(
    const TypeVariable* target_tv,
    uint64_t target_id,
    int polarity, 
    const std::shared_ptr<Type>& in_type,
    std::unordered_set<const Type*>& seen_ptr 
) noexcept {
    auto resolved = resolve_hm(in_type);
    if (!resolved) return false;
    if (resolved->kind == TypeKind::TypeVariable) {
        auto* rv = static_cast<TypeVariable*>(resolved.get());
        if (rv->id == target_id) {
            return polarity != +1;
        }
        return false;
    }
    if (!seen_ptr.insert(resolved.get()).second) return false; // already visited
    auto flip = [](int p) -> int { return (p == 0) ? 0 : -p; };
    switch (resolved->kind) {
    case TypeKind::Function: {
        auto f = std::static_pointer_cast<FunctionType>(resolved);
        int param_pol = flip(polarity);
        for (const auto& p : f->params_ty) {
            if (occurs_check_polar(target_tv, target_id, param_pol, p, seen_ptr)) return true;
        }
        return occurs_check_polar(target_tv, target_id, polarity, f->ret_ty, seen_ptr);
    }
    case TypeKind::LambdaFunction: {
        auto f = std::static_pointer_cast<LambdaFunctionType>(resolved);
        int param_pol = flip(polarity);
        for (const auto& p : f->params_ty) {
            if (occurs_check_polar(target_tv, target_id, param_pol, p, seen_ptr)) return true;
        }
        if (occurs_check_polar(target_tv, target_id, polarity, f->ret_ty, seen_ptr)) return true;
        for (const auto& c : f->capture_tys) {
            if (occurs_check_polar(target_tv, target_id, polarity, c, seen_ptr)) return true;
        }
        return false;
    }
    case TypeKind::NativeFunction: {
        auto f = std::static_pointer_cast<NativeFunctionType>(resolved);
        int param_pol = flip(polarity);
        for (const auto& p : f->params_ty) {
            if (occurs_check_polar(target_tv, target_id, param_pol, p, seen_ptr)) return true;
        }
        return occurs_check_polar(target_tv, target_id, polarity, f->ret_ty, seen_ptr);
    }
    case TypeKind::Array:
        return occurs_check_polar(target_tv, target_id, polarity,
                                  std::static_pointer_cast<ArrayType>(resolved)->type, seen_ptr);
    case TypeKind::Tuple: {
        auto t = std::static_pointer_cast<TupleType>(resolved);
        for (const auto& e : t->tys) {
            if (occurs_check_polar(target_tv, target_id, polarity, e, seen_ptr)) return true;
        }
        return false;
    }
    case TypeKind::Nullable:
        return occurs_check_polar(target_tv, target_id, polarity,
                                  std::static_pointer_cast<NullableType>(resolved)->value_type, seen_ptr);
    case TypeKind::Named: {
        auto n = std::static_pointer_cast<NamedType>(resolved);
        for (const auto& a : n->args) {
            if (occurs_check_polar(target_tv, target_id, polarity, a, seen_ptr)) return true;
        }
        return false;
    }
    default:
        return false;
    }
}

bool occurs_check(const std::shared_ptr<Type>& var, const std::shared_ptr<Type>& in_type) noexcept {
    auto resolved_var = resolve_hm(var);
    auto resolved = resolve_hm(in_type);
    if (resolved_var.get() == resolved.get()) return true; 
    if (!resolved) return false;
    if (resolved_var->kind != TypeKind::TypeVariable) {
        std::unordered_set<const Type*> seen;
        std::function<bool(const std::shared_ptr<Type>&)> deep;
        deep = [&](const std::shared_ptr<Type>& t) -> bool {
            auto r = resolve_hm(t);
            if (!r) return false;
            if (r.get() == resolved_var.get()) return true;
            if (r->kind == TypeKind::TypeVariable) return false;
            if (!seen.insert(r.get()).second) return false;
            switch (r->kind) {
            case TypeKind::Function: {
                auto f = std::static_pointer_cast<FunctionType>(r);
                for (const auto& p : f->params_ty) if (deep(p)) return true;
                return deep(f->ret_ty);
            }
            case TypeKind::LambdaFunction: {
                auto f = std::static_pointer_cast<LambdaFunctionType>(r);
                for (const auto& p : f->params_ty) if (deep(p)) return true;
                if (deep(f->ret_ty)) return true;
                for (const auto& c : f->capture_tys) if (deep(c)) return true;
                return false;
            }
            case TypeKind::NativeFunction: {
                auto f = std::static_pointer_cast<NativeFunctionType>(r);
                for (const auto& p : f->params_ty) if (deep(p)) return true;
                return deep(f->ret_ty);
            }
            case TypeKind::Array: return deep(std::static_pointer_cast<ArrayType>(r)->type);
            case TypeKind::Tuple: {
                auto t = std::static_pointer_cast<TupleType>(r);
                for (const auto& e : t->tys) if (deep(e)) return true;
                return false;
            }
            case TypeKind::Nullable: return deep(std::static_pointer_cast<NullableType>(r)->value_type);
            case TypeKind::Named: {
                auto n = std::static_pointer_cast<NamedType>(r);
                for (const auto& a : n->args) if (deep(a)) return true;
                return false;
            }
            default: return false;
            }
        };
        return deep(resolved);
    }
    auto* target_tv = static_cast<TypeVariable*>(resolved_var.get());
    uint64_t target_id = target_tv->id;
    std::unordered_set<const Type*> seen_ptr;
    return occurs_check_polar(target_tv, target_id, +1, resolved, seen_ptr);
}

struct UnifyTrailEntry {
    TypeVariable* tv;
    std::shared_ptr<Type> old_binding;
};
using UnifyTrailLevel = std::vector<UnifyTrailEntry>;
inline std::vector<UnifyTrailLevel>& unify_trail_stack() noexcept {
    static thread_local std::vector<UnifyTrailLevel> st;
    return st;
}
inline void unify_trail_push() noexcept { unify_trail_stack().emplace_back(); }
inline void unify_record_write(TypeVariable* tv) noexcept {
    if (unify_trail_stack().empty()) return;
    unify_trail_stack().back().push_back(UnifyTrailEntry{tv, tv->binding});
}
inline void unify_trail_rollback() noexcept {
    auto& st = unify_trail_stack();
    if (st.empty()) return;
    auto level = std::move(st.back());
    st.pop_back();
    for (auto it = level.rbegin(); it != level.rend(); ++it) {
        it->tv->binding = std::move(it->old_binding);
    }
}
inline void unify_trail_commit() noexcept {
    auto& st = unify_trail_stack();
    if (st.size() <= 1) {
        st.pop_back(); // top-level: discard, writes permanent
        return;
    }
    auto current = std::move(st.back());
    st.pop_back();
    auto& parent = st.back();
    parent.insert(parent.end(),
                  std::make_move_iterator(current.begin()),
                  std::make_move_iterator(current.end()));
}

bool unify_hm(const std::shared_ptr<Type>& lhs, const std::shared_ptr<Type>& rhs) noexcept {
    if (!lhs || !rhs) { return false; }
    auto a = resolve_hm(lhs);
    auto b = resolve_hm(rhs);

    if (a->kind == TypeKind::Unknown && b->kind == TypeKind::Unknown) {
        return true;
    }
    if (a->kind == TypeKind::Unknown) {
        return true;
    }
    if (b->kind == TypeKind::Unknown) {
        return true;
    }

    if (a->kind == TypeKind::TypeVariable && b->kind == TypeKind::TypeVariable) {
        if (a.get() == b.get()) return true;
        auto ta = std::static_pointer_cast<TypeVariable>(a);
        unify_record_write(ta.get());
        ta->binding = b;
        return true;
    }
    if (a->kind == TypeKind::TypeVariable) {
        auto ta = std::static_pointer_cast<TypeVariable>(a);
        if (occurs_check(a, b)) {
             return false;
         }
        unify_record_write(ta.get());
        ta->binding = b;
        return true;
    }
    if (b->kind == TypeKind::TypeVariable) {
        auto tb = std::static_pointer_cast<TypeVariable>(b);
        // Debug1
        if (occurs_check(b, a)) { return false; }
        unify_record_write(tb.get());
        tb->binding = a;
        return true;
    }
    if (a->kind != b->kind) {
        if ((a->kind == TypeKind::LambdaFunction && b->kind == TypeKind::Function) ||
            (b->kind == TypeKind::LambdaFunction && a->kind == TypeKind::Function)) {
            unify_trail_push();
            std::shared_ptr<Type> fa_t = a, fb_t = b;
            auto get_params = [](std::shared_ptr<Type> t) -> const std::vector<std::shared_ptr<Type>>& {
                if (t->kind == TypeKind::Function)
                    return std::static_pointer_cast<FunctionType>(t)->params_ty;
                return std::static_pointer_cast<LambdaFunctionType>(t)->params_ty;
            };
            auto get_ret = [](std::shared_ptr<Type> t) -> const std::shared_ptr<Type>& {
                if (t->kind == TypeKind::Function)
                    return std::static_pointer_cast<FunctionType>(t)->ret_ty;
                return std::static_pointer_cast<LambdaFunctionType>(t)->ret_ty;
            };
            const auto& pa = get_params(fa_t);
            const auto& pb = get_params(fb_t);
            if (pa.size() != pb.size()) { unify_trail_rollback(); return false; }
            for (size_t i = 0; i < pa.size(); ++i) {
                if (!unify_hm(pa[i], pb[i])) { unify_trail_rollback(); return false; }
            }
            if (!unify_hm(get_ret(fa_t), get_ret(fb_t))) { unify_trail_rollback(); return false; }
            unify_trail_commit();
            return true;
        }

        auto is_ground_leaf = [](TypeKind k) {
            return k == TypeKind::Basic || k == TypeKind::String ||
                   k == TypeKind::None  || k == TypeKind::Module ||
                   k == TypeKind::AdtConstructor || k == TypeKind::Dimensioned;
        };
        auto ret_of = [](const std::shared_ptr<Type>& f_t,
                         const std::shared_ptr<Type>** out_rt) -> bool {
            if (f_t->kind == TypeKind::LambdaFunction) {
                auto f = std::static_pointer_cast<LambdaFunctionType>(f_t);
                if (f->params_ty.size() != 1) return false;
                *out_rt = &f->ret_ty;
                return true;
            }
            if (f_t->kind == TypeKind::Function) {
                auto f = std::static_pointer_cast<FunctionType>(f_t);
                if (f->params_ty.size() != 1) return false;
                *out_rt = &f->ret_ty;
                return true;
            }
            if (f_t->kind == TypeKind::NativeFunction) {
                auto f = std::static_pointer_cast<NativeFunctionType>(f_t);
                if (f->params_ty.size() != 1) return false;
                *out_rt = &f->ret_ty;
                return true;
            }
            return false;
        };
        auto try_etar_bidir = [&](const std::shared_ptr<Type>& A,
                                  const std::shared_ptr<Type>& B) -> bool {
          
            auto is_1param_callable = [&](const std::shared_ptr<Type>& t, const std::shared_ptr<Type>** out_rt) -> bool {
                auto cur = t;
                while (cur && cur->kind == TypeKind::TypeVariable) {
                    auto tv = std::static_pointer_cast<TypeVariable>(cur);
                    if (!tv->binding) break;
                    cur = tv->binding;
                }
                if (!cur) return false;
                if (cur->kind == TypeKind::LambdaFunction) {
                    auto f = std::static_pointer_cast<LambdaFunctionType>(cur);
                    if (f->params_ty.size() != 1) return false;
                    *out_rt = &f->ret_ty;
                    return true;
                }
                if (cur->kind == TypeKind::Function) {
                    auto f = std::static_pointer_cast<FunctionType>(cur);
                    if (f->params_ty.size() != 1) return false;
                    *out_rt = &f->ret_ty;
                    return true;
                }
                if (cur->kind == TypeKind::NativeFunction) {
                    auto f = std::static_pointer_cast<NativeFunctionType>(cur);
                    if (f->params_ty.size() != 1) return false;
                    *out_rt = &f->ret_ty;
                    return true;
                }
                return false;
            };
            constexpr int MAX_ETA_DEPTH = 24;

            std::vector<std::shared_ptr<Type>> a_nodes, b_nodes;
            a_nodes.push_back(A);
            b_nodes.push_back(B);
            int da = 0, db = 0;
            {
                auto c = A;
                const std::shared_ptr<Type>* rp;
                while (da < MAX_ETA_DEPTH && is_1param_callable(c, &rp)) {
                    c = *rp; ++da; a_nodes.push_back(c);
                }
            }
            {
                auto c = B;
                const std::shared_ptr<Type>* rp;
                while (db < MAX_ETA_DEPTH && is_1param_callable(c, &rp)) {
                    c = *rp; ++db; b_nodes.push_back(c);
                }
            }
            int min_d = std::min(da, db);
            int diff  = std::abs(da - db);

            auto with_perm_commit = [&](auto fn) -> bool {
                auto& st = unify_trail_stack();
                std::vector<UnifyTrailLevel> saved_levels;
                saved_levels.reserve(st.size());
                while (!st.empty()) { saved_levels.emplace_back(std::move(st.back())); st.pop_back(); }
                bool ok = fn();
                while (!saved_levels.empty()) { st.emplace_back(std::move(saved_levels.back())); saved_levels.pop_back(); }
                return ok;
            };
            auto uwalk = [](std::shared_ptr<Type> c) -> std::shared_ptr<Type> {
                while (c && c->kind == TypeKind::TypeVariable) {
                    auto tv = std::static_pointer_cast<TypeVariable>(c);
                    if (!tv->binding) break;
                    c = tv->binding;
                }
                return c;
            };

            if (diff > 0) {
                bool a_longer = (da > db);
                auto short_mth = a_longer ? b_nodes[min_d] : a_nodes[min_d];  
                auto long_pre  = a_longer ? a_nodes[min_d] : b_nodes[min_d];  
                auto long_term = a_longer ? a_nodes.back()  : b_nodes.back();  
                {
                    auto lt = uwalk(long_term);
                    if (lt && lt->kind == TypeKind::TypeVariable) {
                        auto tv = std::static_pointer_cast<TypeVariable>(lt);
                        if (min_d < 1) {
                            return false;
                        } else {
                            return with_perm_commit([&]() {                                
                                tv->binding = short_mth; return true;
                            });
                        }
                    }
                }

                if (min_d >= 1) {
                    return with_perm_commit([&]() {
                        return unify_hm(short_mth, long_term);
                    });
                }
            }
            if (diff == 0) {
                auto ta = uwalk(a_nodes.back());
                auto tb = uwalk(b_nodes.back());
                if (ta && ta->kind == TypeKind::TypeVariable) {
                    auto tv = std::static_pointer_cast<TypeVariable>(ta);
                    return with_perm_commit([&]() {
                        tv->binding = b_nodes.back(); return true;
                    });
                }
                if (tb && tb->kind == TypeKind::TypeVariable) {
                    auto tv = std::static_pointer_cast<TypeVariable>(tb);
                    return with_perm_commit([&]() {
                        tv->binding = a_nodes.back(); return true;
                    });
                }
                if (min_d >= 1) {
                    return with_perm_commit([&]() {
                        return unify_hm(a_nodes.back(), b_nodes.back());
                    });
                }
            }

            return false;
        };
       
        bool a_leaf = is_ground_leaf(a->kind);
        bool b_leaf = is_ground_leaf(b->kind);
        if (a_leaf || b_leaf) {
            if (try_etar_bidir(a, b)) return true;
        }
        return false;
    }
    switch (a->kind) {
    case TypeKind::Basic:
    case TypeKind::String:
    case TypeKind::None:
    case TypeKind::Module:
    case TypeKind::AdtConstructor:
    case TypeKind::Dimensioned:
        if (!a->equals(b.get())) { return false; }
        return true;
    case TypeKind::Function: {
        unify_trail_push();
        auto fa = std::static_pointer_cast<FunctionType>(a);
        auto fb = std::static_pointer_cast<FunctionType>(b);
        if (fa->params_ty.size() != fb->params_ty.size()) { unify_trail_rollback(); return false; }
        for (size_t i = 0; i < fa->params_ty.size(); ++i) {
            if (!unify_hm(fa->params_ty[i], fb->params_ty[i])) { unify_trail_rollback(); return false; }
        }
        if (!unify_hm(fa->ret_ty, fb->ret_ty)) { unify_trail_rollback(); return false; }
        unify_trail_commit();
        return true;
    }
    case TypeKind::LambdaFunction: {
        unify_trail_push();
        auto fa = std::static_pointer_cast<LambdaFunctionType>(a);
        auto fb = std::static_pointer_cast<LambdaFunctionType>(b);
        if (fa->params_ty.size() != fb->params_ty.size()) {unify_trail_rollback(); return false; }
        for (size_t i = 0; i < fa->params_ty.size(); ++i) {
            if (!unify_hm(fa->params_ty[i], fb->params_ty[i])) { unify_trail_rollback(); return false; }
        }
        if (!unify_hm(fa->ret_ty, fb->ret_ty)) { unify_trail_rollback(); return false; }
        unify_trail_commit();
        return true;
    }
    case TypeKind::NativeFunction: {
        unify_trail_push();
        auto fa = std::static_pointer_cast<NativeFunctionType>(a);
        auto fb = std::static_pointer_cast<NativeFunctionType>(b);
        if (fa->name != fb->name) {unify_trail_rollback(); return false; }
        if (fa->params_ty.size() != fb->params_ty.size()) {unify_trail_rollback(); return false; }
        for (size_t i = 0; i < fa->params_ty.size(); ++i) {
            if (!unify_hm(fa->params_ty[i], fb->params_ty[i])) { unify_trail_rollback(); return false; }
        }
        if (!unify_hm(fa->ret_ty, fb->ret_ty)) { unify_trail_rollback(); return false; }
        unify_trail_commit();
        return true;
    }
    case TypeKind::Array: {
        auto aa = std::static_pointer_cast<ArrayType>(a);
        auto ab = std::static_pointer_cast<ArrayType>(b);
        return unify_hm(aa->type, ab->type);
    }
    case TypeKind::Tuple: {
        unify_trail_push();
        auto ta = std::static_pointer_cast<TupleType>(a);
        auto tb = std::static_pointer_cast<TupleType>(b);
        if (ta->tys.size() != tb->tys.size()) { unify_trail_rollback(); return false; }
        for (size_t i = 0; i < ta->tys.size(); ++i) {
            if (!unify_hm(ta->tys[i], tb->tys[i])) { unify_trail_rollback(); return false; }
        }
        unify_trail_commit();
        return true;
    }
    case TypeKind::Nullable: {
        auto na = std::static_pointer_cast<NullableType>(a);
        auto nb = std::static_pointer_cast<NullableType>(b);
        return unify_hm(na->value_type, nb->value_type);
    }
    case TypeKind::Named: {
        unify_trail_push();
        auto na = std::static_pointer_cast<NamedType>(a);
        auto nb = std::static_pointer_cast<NamedType>(b);
        if (na->name != nb->name) { unify_trail_rollback(); return false; }
        if (na->args.size() != nb->args.size()) { unify_trail_rollback(); return false; }
        for (size_t i = 0; i < na->args.size(); ++i) {
            if (!unify_hm(na->args[i], nb->args[i])) { unify_trail_rollback(); return false; }
        }
        unify_trail_commit();
        return true;
    }
    case TypeKind::Unknown:
        return true;
    default:
        return a->equals(b.get());
    }
}

std::shared_ptr<Type> deep_resolve(const std::shared_ptr<Type>& type) noexcept {
    auto resolved = resolve_hm(type);
    if (!resolved) return resolved;
    switch (resolved->kind) {
    case TypeKind::Function: {
        auto f = std::static_pointer_cast<FunctionType>(resolved);
        std::vector<std::shared_ptr<Type>> params;
        params.reserve(f->params_ty.size());
        for (const auto& p : f->params_ty) params.push_back(deep_resolve(p));
        return type_pool.function(std::move(params), deep_resolve(f->ret_ty));
    }
    case TypeKind::LambdaFunction: {
        auto f = std::static_pointer_cast<LambdaFunctionType>(resolved);
        std::vector<std::shared_ptr<Type>> params, captures;
        params.reserve(f->params_ty.size());
        captures.reserve(f->capture_tys.size());
        for (const auto& p : f->params_ty) params.push_back(deep_resolve(p));
        for (const auto& c : f->capture_tys) captures.push_back(deep_resolve(c));
        return type_pool.lambda_function(std::move(params), deep_resolve(f->ret_ty), std::move(captures));
    }
    case TypeKind::NativeFunction: {
        auto f = std::static_pointer_cast<NativeFunctionType>(resolved);
        std::vector<std::shared_ptr<Type>> params;
        params.reserve(f->params_ty.size());
        for (const auto& p : f->params_ty) params.push_back(deep_resolve(p));
        return type_pool.native_function(std::move(params), deep_resolve(f->ret_ty), f->name);
    }
    case TypeKind::Array: {
        auto a = std::static_pointer_cast<ArrayType>(resolved);
        return type_pool.array(deep_resolve(a->type));
    }
    case TypeKind::Tuple: {
        auto t = std::static_pointer_cast<TupleType>(resolved);
        std::vector<std::shared_ptr<Type>> elements;
        elements.reserve(t->tys.size());
        for (const auto& e : t->tys) elements.push_back(deep_resolve(e));
        return type_pool.tuple(std::move(elements));
    }
    case TypeKind::Nullable: {
        auto n = std::static_pointer_cast<NullableType>(resolved);
        return type_pool.nullable(deep_resolve(n->value_type));
    }
    case TypeKind::Named: {
        auto n = std::static_pointer_cast<NamedType>(resolved);
        std::vector<std::shared_ptr<Type>> args;
        args.reserve(n->args.size());
        for (const auto& a : n->args) args.push_back(deep_resolve(a));
        return type_pool.named(n->name, std::move(args));
    }
    default:
        return resolved;
    }
}

std::shared_ptr<Type> replace_unknowns_with_tvars(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return type;
    auto r = resolve_hm(type);
    switch (r->kind) {
    case TypeKind::Unknown:
        return std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
    case TypeKind::Nullable: {
        auto n = std::static_pointer_cast<NullableType>(r);
        return type_pool.nullable(replace_unknowns_with_tvars(n->value_type));
    }
    case TypeKind::Array: {
        auto a = std::static_pointer_cast<ArrayType>(r);
        return type_pool.array(replace_unknowns_with_tvars(a->type));
    }
    case TypeKind::Tuple: {
        auto t = std::static_pointer_cast<TupleType>(r);
        std::vector<std::shared_ptr<Type>> vs;
        vs.reserve(t->tys.size());
        for (const auto& e : t->tys) vs.push_back(replace_unknowns_with_tvars(e));
        return type_pool.tuple(std::move(vs));
    }
    case TypeKind::Function: {
        auto f = std::static_pointer_cast<FunctionType>(r);
        std::vector<std::shared_ptr<Type>> ps;
        ps.reserve(f->params_ty.size());
        for (const auto& p : f->params_ty) ps.push_back(replace_unknowns_with_tvars(p));
        return type_pool.function(std::move(ps), replace_unknowns_with_tvars(f->ret_ty));
    }
    case TypeKind::LambdaFunction: {
        auto f = std::static_pointer_cast<LambdaFunctionType>(r);
        std::vector<std::shared_ptr<Type>> ps, cs;
        ps.reserve(f->params_ty.size());
        cs.reserve(f->capture_tys.size());
        for (const auto& p : f->params_ty) ps.push_back(replace_unknowns_with_tvars(p));
        for (const auto& c : f->capture_tys) cs.push_back(replace_unknowns_with_tvars(c));
        return type_pool.lambda_function(std::move(ps), replace_unknowns_with_tvars(f->ret_ty), std::move(cs));
    }
    case TypeKind::NativeFunction: {
        auto f = std::static_pointer_cast<NativeFunctionType>(r);
        std::vector<std::shared_ptr<Type>> ps;
        ps.reserve(f->params_ty.size());
        for (const auto& p : f->params_ty) ps.push_back(replace_unknowns_with_tvars(p));
        return type_pool.native_function(std::move(ps), replace_unknowns_with_tvars(f->ret_ty), f->name);
    }
    case TypeKind::Named: {
        auto n = std::static_pointer_cast<NamedType>(r);
        std::vector<std::shared_ptr<Type>> as;
        as.reserve(n->args.size());
        for (const auto& a : n->args) as.push_back(replace_unknowns_with_tvars(a));
        return type_pool.named(n->name, std::move(as));
    }
    default:
        return r;
    }
}

void collect_free_type_vars(
    const std::shared_ptr<Type>& type,
    std::unordered_set<TypeVariable*>& free_vars,
    std::unordered_set<TypeVariable*>& visited
) noexcept {
    auto resolved = resolve_hm(type);
    if (!resolved) return;
    if (resolved->kind == TypeKind::TypeVariable) {
        auto tv = std::static_pointer_cast<TypeVariable>(resolved).get();
        if (!visited.insert(tv).second) return;
        free_vars.insert(tv);
        return;
    }
    auto recurse_list = [&](const std::vector<std::shared_ptr<Type>>& list) {
        for (const auto& t : list) collect_free_type_vars(t, free_vars, visited);
    };
    switch (resolved->kind) {
    case TypeKind::Function: {
        auto f = std::static_pointer_cast<FunctionType>(resolved);
        recurse_list(f->params_ty);
        collect_free_type_vars(f->ret_ty, free_vars, visited);
        break;
    }
    case TypeKind::LambdaFunction: {
        auto f = std::static_pointer_cast<LambdaFunctionType>(resolved);
        recurse_list(f->params_ty);
        collect_free_type_vars(f->ret_ty, free_vars, visited);
        recurse_list(f->capture_tys);
        break;
    }
    case TypeKind::NativeFunction: {
        auto f = std::static_pointer_cast<NativeFunctionType>(resolved);
        recurse_list(f->params_ty);
        collect_free_type_vars(f->ret_ty, free_vars, visited);
        break;
    }
    case TypeKind::Array:
        collect_free_type_vars(std::static_pointer_cast<ArrayType>(resolved)->type, free_vars, visited);
        break;
    case TypeKind::Tuple:
        recurse_list(std::static_pointer_cast<TupleType>(resolved)->tys);
        break;
    case TypeKind::Nullable:
        collect_free_type_vars(std::static_pointer_cast<NullableType>(resolved)->value_type, free_vars, visited);
        break;
    case TypeKind::Named:
        recurse_list(std::static_pointer_cast<NamedType>(resolved)->args);
        break;
    default:
        break;
    }
}

std::shared_ptr<Type> instantiate_scheme(const TypeScheme& scheme) noexcept {
    auto const_resolve = [](const std::shared_ptr<Type>& t) -> std::shared_ptr<Type> {
        auto cur = t;
        while (cur && cur->kind == TypeKind::TypeVariable) {
            auto tv = std::static_pointer_cast<TypeVariable>(cur);
            if (is_recursive_mu_head(tv.get())) break; 
            if (!tv->binding) break;
            cur = tv->binding;
        }
        return cur;
    };

    std::unordered_map<TypeVariable*, std::shared_ptr<TypeVariable>> fresh_tvs;
    std::vector<std::pair<std::shared_ptr<TypeVariable>, std::shared_ptr<Type>>> delayed_fixup;
    fresh_tvs.reserve(scheme.quantified.size());
    delayed_fixup.reserve(scheme.quantified.size());
    for (auto* q : scheme.quantified) {
        auto fresh = type_pool.fresh_type_variable();
        fresh_tvs.emplace(q, fresh);
        if (q->binding) {
            delayed_fixup.emplace_back(fresh, q->binding);
        }
    }
    std::unordered_map<TypeVariable*, std::shared_ptr<Type>> subst;
    subst.reserve(fresh_tvs.size());
    for (auto& [k, v] : fresh_tvs) subst.emplace(k, std::static_pointer_cast<Type>(v));

    std::function<std::shared_ptr<Type>(const std::shared_ptr<Type>&)> copy;
    copy = [&](const std::shared_ptr<Type>& t) -> std::shared_ptr<Type> {
        if (t && t->kind == TypeKind::TypeVariable) {
            auto tv_raw = std::static_pointer_cast<TypeVariable>(t).get();
            auto it = subst.find(tv_raw);
            if (it != subst.end()) {
                return it->second;
            }
        }
        auto r = const_resolve(t);
        if (!r) return r;
        if (r->kind == TypeKind::TypeVariable) {
            auto tv = std::static_pointer_cast<TypeVariable>(r).get();
            auto it = subst.find(tv);
            if (it != subst.end()) {
                return it->second;
            }
            return r;
        }
        switch (r->kind) {
        case TypeKind::Function: {
            auto f = std::static_pointer_cast<FunctionType>(r);
            std::vector<std::shared_ptr<Type>> params;
            params.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) params.push_back(copy(p));
            return type_pool.function(std::move(params), copy(f->ret_ty));
        }
        case TypeKind::LambdaFunction: {
            auto f = std::static_pointer_cast<LambdaFunctionType>(r);
            std::vector<std::shared_ptr<Type>> params, captures;
            params.reserve(f->params_ty.size());
            captures.reserve(f->capture_tys.size());
            for (const auto& p : f->params_ty) params.push_back(copy(p));
            for (const auto& c : f->capture_tys) captures.push_back(copy(c));
            return type_pool.lambda_function(std::move(params), copy(f->ret_ty), std::move(captures));
        }
        case TypeKind::NativeFunction: {
            auto f = std::static_pointer_cast<NativeFunctionType>(r);
            std::vector<std::shared_ptr<Type>> params;
            params.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) params.push_back(copy(p));
            return type_pool.native_function(std::move(params), copy(f->ret_ty), f->name);
        }
        case TypeKind::Array:
            return type_pool.array(copy(std::static_pointer_cast<ArrayType>(r)->type));
        case TypeKind::Tuple: {
            auto t = std::static_pointer_cast<TupleType>(r);
            std::vector<std::shared_ptr<Type>> elements;
            elements.reserve(t->tys.size());
            for (const auto& e : t->tys) elements.push_back(copy(e));
            return type_pool.tuple(std::move(elements));
        }
        case TypeKind::Nullable:
            return type_pool.nullable(copy(std::static_pointer_cast<NullableType>(r)->value_type));
        case TypeKind::Named: {
            auto n = std::static_pointer_cast<NamedType>(r);
            std::vector<std::shared_ptr<Type>> args;
            args.reserve(n->args.size());
            for (const auto& a : n->args) args.push_back(copy(a));
            return type_pool.named(n->name, std::move(args));
        }
        default:
            return r;
        }
    };
    auto result = copy(scheme.monotype);
    (void)delayed_fixup;
    return result;
}

} // namespace lmx::hir

namespace {

bool is_basic_type(const std::shared_ptr<Type>& type, runtime::ValueKind kind) noexcept {
    return type && type->kind == TypeKind::Basic &&
           std::reinterpret_pointer_cast<BasicType>(type)->type == kind;
}

bool is_numeric_or_expr_type(const std::shared_ptr<Type>& type) noexcept {
    if (type && type->kind == TypeKind::Dimensioned) return true;
    if (!type || type->kind != TypeKind::Basic) return false;
    const auto kind = std::reinterpret_pointer_cast<BasicType>(type)->type;
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction ||
           kind == runtime::ValueKind::Real ||
           kind == runtime::ValueKind::Complex ||
           kind == runtime::ValueKind::Expr;
}

bool is_expr_type(const std::shared_ptr<Type>& type) noexcept {
    return is_basic_type(type, runtime::ValueKind::Expr);
}

bool is_expr_constructible(const std::shared_ptr<Type>& type) noexcept {
    if (type && type->kind == TypeKind::Dimensioned) return true;
    if (type && type->kind == TypeKind::Named) {
        return std::static_pointer_cast<NamedType>(type)->name == "interval";
    }
    if (!type || type->kind != TypeKind::Basic) return false;
    const auto kind = std::reinterpret_pointer_cast<BasicType>(type)->type;
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction ||
           kind == runtime::ValueKind::Real ||
           kind == runtime::ValueKind::Complex ||
           kind == runtime::ValueKind::Expr;
}

bool supports_basic_equality(const runtime::ValueKind kind) noexcept {
    switch (kind) {
    case runtime::ValueKind::Int:
    case runtime::ValueKind::Fraction:
    case runtime::ValueKind::Real:
    case runtime::ValueKind::Complex:
    case runtime::ValueKind::Vector:
    case runtime::ValueKind::Matrix:
    case runtime::ValueKind::Table:
    case runtime::ValueKind::Quantity:
    case runtime::ValueKind::Sparse:
    case runtime::ValueKind::Tensor:
    case runtime::ValueKind::Assumptions:
        return true;
    default:
        return false;
    }
}

bool is_int_or_fraction(const runtime::ValueKind kind) noexcept {
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction;
}

std::optional<runtime::ValueKind> basic_binary_result(
    const runtime::ValueKind operand,
    const BinaryNode::Op op) noexcept {
    switch (op) {
    case BinaryNode::Op::Add:
    case BinaryNode::Op::Sub:
    case BinaryNode::Op::Mul:
    case BinaryNode::Op::Mod:
    case BinaryNode::Op::Pow:
        if (is_int_or_fraction(operand) ||
            operand == runtime::ValueKind::Real) return operand;
        return std::nullopt;
    case BinaryNode::Op::Div:
        if (is_int_or_fraction(operand) ||
            operand == runtime::ValueKind::Real)
            return operand == runtime::ValueKind::Real
                ? runtime::ValueKind::Real : runtime::ValueKind::Fraction;
        return std::nullopt;
    case BinaryNode::Op::Eq:
    case BinaryNode::Op::Ne:
        if (supports_basic_equality(operand)) return runtime::ValueKind::Bool;
        return std::nullopt;
    case BinaryNode::Op::Gt:
    case BinaryNode::Op::Ge:
    case BinaryNode::Op::Lt:
    case BinaryNode::Op::Le:
        if (is_int_or_fraction(operand) ||
            operand == runtime::ValueKind::Real)
            return runtime::ValueKind::Bool;
        return std::nullopt;
    case BinaryNode::Op::And:
    case BinaryNode::Op::Or:
        if (operand == runtime::ValueKind::Bool)
            return runtime::ValueKind::Bool;
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

bool is_named_type(const std::shared_ptr<Type>& type, const std::string_view name) noexcept {
    return type && type->kind == TypeKind::Named &&
           std::static_pointer_cast<NamedType>(type)->name == name;
}

bool is_dimensioned_type(const std::shared_ptr<Type>& type) noexcept;

std::optional<std::pair<bool, bool>> interval_constructor_bounds(
    const ExprNode* expression) noexcept {
    if (!expression || expression->kind != ASTKind::DotExpr) return std::nullopt;
    const auto* dot = static_cast<const DotExprNode*>(expression);
    if (!dot->expr || dot->expr->kind != ASTKind::Identifier || !dot->rhs) {
        return std::nullopt;
    }
    const auto* module = static_cast<const IdentifierNode*>(dot->expr.get());
    if (module->id != "std") return std::nullopt;
    if (dot->rhs->id == "interval_closed") return std::pair{true, true};
    if (dot->rhs->id == "interval_open") return std::pair{false, false};
    if (dot->rhs->id == "interval_closed_open") return std::pair{true, false};
    if (dot->rhs->id == "interval_open_closed") return std::pair{false, true};
    return std::nullopt;
}

bool is_interval_ordered_type(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return false;
    if (type->kind == TypeKind::Dimensioned) return true;
    if (type->kind != TypeKind::Basic) return false;
    switch (std::static_pointer_cast<BasicType>(type)->type) {
    case runtime::ValueKind::Int:
    case runtime::ValueKind::Fraction:
    case runtime::ValueKind::Real:
        return true;
    default:
        return false;
    }
}

int numeric_rank(const std::shared_ptr<Type>& type) noexcept {
    if (!type || type->kind != TypeKind::Basic) return -1;
    switch (std::static_pointer_cast<BasicType>(type)->type) {
    case runtime::ValueKind::Int: return 0;
    case runtime::ValueKind::Fraction: return 1;
    case runtime::ValueKind::Real: return 2;
    default: return -1;
    }
}

std::shared_ptr<Type> unify_interval_bounds(const std::shared_ptr<Type>& lhs,
                                           const std::shared_ptr<Type>& rhs) noexcept {
    if (!lhs || !rhs) return nullptr;
    if (lhs->equals(rhs.get())) return lhs;
    const auto left_rank = numeric_rank(lhs);
    const auto right_rank = numeric_rank(rhs);
    if (left_rank >= 0 && right_rank >= 0) {
        const auto rank = std::max(left_rank, right_rank);
        const auto kind = rank == 0 ? runtime::ValueKind::Int
            : rank == 1 ? runtime::ValueKind::Fraction
                        : runtime::ValueKind::Real;
        return type_pool.basic(kind);
    }
    if (is_dimensioned_type(lhs) && is_dimensioned_type(rhs)) {
        const auto left = std::static_pointer_cast<DimensionedType>(lhs);
        const auto right = std::static_pointer_cast<DimensionedType>(rhs);
        if (left->unit.dimension == right->unit.dimension) return lhs;
    }
    return nullptr;
}

bool interval_member_assignable(const std::shared_ptr<Type>& expected,
                                const std::shared_ptr<Type>& actual) noexcept;

void mark_expr_promotion(const std::shared_ptr<ExprNode>& expression) {
    if (!expression || is_expr_type(expression->type)) return;
    expression->promoted_from_type = expression->type;
    expression->type = type_pool.basic(runtime::ValueKind::Expr);
}

bool is_dimensioned_type(const std::shared_ptr<Type>& type) noexcept {
    return type && type->kind == TypeKind::Dimensioned &&
           std::static_pointer_cast<DimensionedType>(type)->resolved;
}

std::optional<RationalScale> constant_numeric_value(const ExprNode* expression) noexcept {
    if (!expression) return std::nullopt;
    switch (expression->kind) {
    case ASTKind::Literal: {
        const auto* literal = static_cast<const LiteralNode*>(expression);
        if (literal->kind != LiteralNode::Kind::Integer &&
            literal->kind != LiteralNode::Kind::Float) return std::nullopt;
        return RationalScale::from_decimal(literal->val);
    }
    case ASTKind::Unary: {
        const auto* unary = static_cast<const UnaryNode*>(expression);
        if (unary->op != UnaryNode::Op::Neg) return std::nullopt;
        auto value = constant_numeric_value(unary->expr.get());
        if (!value) return std::nullopt;
        value->numerator = -value->numerator;
        return value;
    }
    case ASTKind::Binary: {
        const auto* binary = static_cast<const BinaryNode*>(expression);
        auto lhs = constant_numeric_value(binary->lhs.get());
        auto rhs = constant_numeric_value(binary->rhs.get());
        if (!lhs || !rhs) return std::nullopt;
        switch (binary->op) {
        case BinaryNode::Op::Mul: return lhs->multiplied_by(*rhs);
        case BinaryNode::Op::Div: return lhs->divided_by(*rhs);
        case BinaryNode::Op::Add: return lhs->added_to(*rhs);
        case BinaryNode::Op::Sub: return lhs->subtracted_by(*rhs);
        case BinaryNode::Op::Pow:
            if (rhs->denominator != 1 || rhs->numerator < -32 || rhs->numerator > 32)
                return std::nullopt;
            return lhs->raised_to(static_cast<int>(rhs->numerator));
        default: return std::nullopt;
        }
    }
    case ASTKind::UnitAnnotated: {
        const auto* unit = static_cast<const UnitAnnotatedExprNode*>(expression);
        auto value = constant_numeric_value(unit->value.get());
        return value ? value->multiplied_by(unit->resolved_unit.scale_to_base)
                     : std::nullopt;
    }
    default: return std::nullopt;
    }
}

std::optional<std::int64_t> signed_integer_literal(
    const ExprNode* expression) noexcept {
    if (!expression) return std::nullopt;
    if (expression->kind == ASTKind::Literal) {
        const auto* literal = static_cast<const LiteralNode*>(expression);
        if (literal->kind != LiteralNode::Kind::Integer) return std::nullopt;
        try {
            std::size_t used = 0;
            const auto value = std::stoll(literal->val, &used);
            return used == literal->val.size()
                ? std::optional<std::int64_t>(value) : std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
    if (expression->kind != ASTKind::Unary) return std::nullopt;
    const auto* unary = static_cast<const UnaryNode*>(expression);
    if (unary->op != UnaryNode::Op::Neg) return std::nullopt;
    const auto value = signed_integer_literal(unary->expr.get());
    if (!value || *value == std::numeric_limits<std::int64_t>::min())
        return std::nullopt;
    return -*value;
}

std::optional<UnitDefinition> combined_unit(const UnitDefinition& lhs,
                                            const UnitDefinition& rhs,
                                            const bool divide) {
    UnitDefinition result;
    result.dimension = divide ? lhs.dimension.divided_by(rhs.dimension)
                              : lhs.dimension.multiplied_by(rhs.dimension);
    const auto scale = divide ? lhs.scale_to_base.divided_by(rhs.scale_to_base)
                              : lhs.scale_to_base.multiplied_by(rhs.scale_to_base);
    if (!scale) return std::nullopt;
    result.scale_to_base = *scale;
    result.display_unit = lhs.display_unit + (divide ? "/" : "*") + rhs.display_unit;
    return result;
}

bool runtime_scale_representable(const RationalScale& scale) noexcept {
    return scale.numerator >= std::numeric_limits<std::int32_t>::min() &&
           scale.numerator <= std::numeric_limits<std::int32_t>::max() &&
           scale.denominator > 0 &&
           scale.denominator <= std::numeric_limits<std::int32_t>::max();
}

std::shared_ptr<Type> unify_types(const std::shared_ptr<Type>& lhs,
                                  const std::shared_ptr<Type>& rhs) noexcept;

std::shared_ptr<Type> literal_payload_type(const LiteralPayloadNode& node) noexcept {
    const bool has_expr = std::ranges::any_of(node.elements, [](const auto& element) {
        return is_expr_type(element->type);
    });
    if (node.payload_kind == LiteralPayloadNode::Kind::Interval) {
        if (has_expr) {
            const bool promotable = std::ranges::all_of(
                node.elements, [](const auto& element) {
                    return is_expr_constructible(element->type);
                });
            return promotable ? type_pool.basic(runtime::ValueKind::Expr)
                              : type_pool.unknown();
        }
        if (node.elements.size() != 2) return type_pool.unknown();
        auto element = unify_interval_bounds(node.elements[0]->type,
                                             node.elements[1]->type);
        return element ? type_pool.named("interval", {std::move(element)})
                       : type_pool.unknown();
    }

    if (node.elements.empty()) return type_pool.named("set", {type_pool.unknown()});
    auto element = node.elements.front()->type;
    for (std::size_t i = 1; i < node.elements.size(); ++i) {
        const auto& candidate = node.elements[i]->type;
        if (is_expr_type(element) || is_expr_type(candidate)) {
            if (!is_expr_constructible(element) ||
                !is_expr_constructible(candidate))
                return type_pool.unknown();
            element = type_pool.basic(runtime::ValueKind::Expr);
            continue;
        }
        auto unified = unify_types(element, candidate);
        if (!unified)
            unified = unify_interval_bounds(element, candidate);
        if (!unified) return type_pool.unknown();
        element = std::move(unified);
    }
    return type_pool.named("set", {std::move(element)});
}

using TypeBindings = std::unordered_map<std::string, std::shared_ptr<Type>>;

bool bind_adt_type(const std::shared_ptr<Type>& expected,
                   const std::shared_ptr<Type>& actual,
                   const std::unordered_set<std::string>& params,
                   TypeBindings& bindings) noexcept {
    if (!expected || !actual) return false;
    if (actual->kind == TypeKind::Unknown) return true;
    if (actual->kind == TypeKind::Never) return true;
    if (expected->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(expected);
        if (actual->kind == TypeKind::Basic &&
            std::static_pointer_cast<BasicType>(actual)->type == runtime::ValueKind::Null) return true;
        if (actual->kind == TypeKind::Nullable)
            return bind_adt_type(nullable->value_type,
                                 std::static_pointer_cast<NullableType>(actual)->value_type,
                                 params, bindings);
        return bind_adt_type(nullable->value_type, actual, params, bindings);
    }
    if (expected->kind == TypeKind::Named) {
        const auto named = std::static_pointer_cast<NamedType>(expected);
        if (params.contains(named->name) && named->args.empty()) {
            const auto it = bindings.find(named->name);
            if (it == bindings.end()) {
                bindings[named->name] = actual;
                return true;
            }
            return it->second->equals(actual.get());
        }
        if (actual->kind != TypeKind::Named) return false;
        const auto actual_named = std::static_pointer_cast<NamedType>(actual);
        if (named->name != actual_named->name || named->args.size() != actual_named->args.size()) return false;
        for (size_t i = 0; i < named->args.size(); ++i) {
            if (!bind_adt_type(named->args[i], actual_named->args[i], params, bindings)) return false;
        }
        return true;
    }
    return expected->equals(actual.get());
}

bool type_assignable(const std::shared_ptr<Type>& expected,
                     const std::shared_ptr<Type>& actual) noexcept {
    if (!expected || !actual) return false;
    const auto expected_resolved = resolve_hm(expected);
    const auto actual_resolved = resolve_hm(actual);
    if (actual_resolved->kind == TypeKind::Never) return true;
    if (expected_resolved->kind == TypeKind::TypeVariable ||
        actual_resolved->kind == TypeKind::TypeVariable) {
        return unify_hm(expected, actual);
    }
    if (numeric_rank(expected_resolved) >= 0 && numeric_rank(actual_resolved) >= 0) {
        return numeric_rank(expected_resolved) >= numeric_rank(actual_resolved);
    }
    if (expected_resolved->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(expected_resolved);
        if (actual_resolved->kind == TypeKind::Basic &&
            std::static_pointer_cast<BasicType>(actual_resolved)->type == runtime::ValueKind::Null) return true;
        if (actual_resolved->kind == TypeKind::Nullable)
            return type_assignable(nullable->value_type,
                                   std::static_pointer_cast<NullableType>(actual_resolved)->value_type);
        return type_assignable(nullable->value_type, actual_resolved);
    }
    if ((expected_resolved->kind == TypeKind::Function || expected_resolved->kind == TypeKind::LambdaFunction) &&
        (actual_resolved->kind == TypeKind::Function || actual_resolved->kind == TypeKind::LambdaFunction)) {
        if (expected_resolved->kind == TypeKind::Function && actual_resolved->kind == TypeKind::Function) {
            const auto expected_function = std::static_pointer_cast<FunctionType>(expected_resolved);
            const auto actual_function = std::static_pointer_cast<FunctionType>(actual_resolved);
            if (expected_function->params_ty.size() == actual_function->params_ty.size()) {
                bool params_ok = true;
                for (size_t i = 0; i < expected_function->params_ty.size(); ++i) {
                    if (!expected_function->params_ty[i]->equals(actual_function->params_ty[i].get())) {
                        params_ok = false;
                        break;
                    }
                }
                if (params_ok) {
                    return actual_function->ret_ty->kind == TypeKind::Never ||
                           expected_function->ret_ty->equals(actual_function->ret_ty.get());
                }
            }
        }
        return unify_hm(expected_resolved, actual_resolved);
    }
    if (expected_resolved->kind != TypeKind::Named || actual_resolved->kind != TypeKind::Named)
        return expected_resolved->equals(actual_resolved.get());
    const auto expected_named = std::static_pointer_cast<NamedType>(expected_resolved);
    const auto actual_named = std::static_pointer_cast<NamedType>(actual_resolved);
    if (expected_named->name != actual_named->name || expected_named->args.size() != actual_named->args.size()) return false;
    for (size_t i = 0; i < expected_named->args.size(); ++i) {
        if (actual_named->args[i]->kind == TypeKind::Unknown) continue;
        if (resolve_hm(actual_named->args[i])->kind == TypeKind::TypeVariable) continue;
        if (!type_assignable(expected_named->args[i], actual_named->args[i])) return false;
    }
    return true;
}

bool interval_member_assignable(const std::shared_ptr<Type>& expected,
                                const std::shared_ptr<Type>& actual) noexcept {
    if (numeric_rank(expected) >= 0 && numeric_rank(actual) >= 0) return true;
    if (is_dimensioned_type(expected) && is_dimensioned_type(actual)) {
        return std::static_pointer_cast<DimensionedType>(expected)->unit.dimension ==
               std::static_pointer_cast<DimensionedType>(actual)->unit.dimension;
    }
    return type_assignable(expected, actual);
}

std::shared_ptr<Type> unify_types(const std::shared_ptr<Type>& lhs,
                                  const std::shared_ptr<Type>& rhs) noexcept {
    if (!lhs || !rhs) return nullptr;
    auto l_resolved = resolve_hm(lhs);
    auto r_resolved = resolve_hm(rhs);
    if (lhs->kind == TypeKind::Never)
        return rhs->kind == TypeKind::Never ? lhs : rhs;
    if (rhs->kind == TypeKind::Never) return lhs;
    if (l_resolved->kind == TypeKind::Unknown) return rhs;
    if (r_resolved->kind == TypeKind::Unknown) return lhs;
    if (l_resolved->kind == TypeKind::TypeVariable || r_resolved->kind == TypeKind::TypeVariable) {
        if (unify_hm(lhs, rhs)) return deep_resolve(lhs);
        return nullptr;
    }
    if (numeric_rank(lhs) >= 0 && numeric_rank(rhs) >= 0)
        return unify_interval_bounds(lhs, rhs);
    const auto is_null = [](const std::shared_ptr<Type>& type) {
        if (!type)
            return false;
        return type->kind == TypeKind::Basic &&
               std::static_pointer_cast<BasicType>(type)->type == runtime::ValueKind::Null;
    };  
    if (lhs->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(lhs);
        if (is_null(rhs)) return lhs;
        auto value = rhs->kind == TypeKind::Nullable
            ? std::static_pointer_cast<NullableType>(rhs)->value_type : rhs;
        auto unified = unify_types(nullable->value_type, value);
        return unified ? type_pool.nullable(std::move(unified)) : nullptr;
    }
    if (rhs->kind == TypeKind::Nullable || is_null(lhs)) return unify_types(rhs, lhs);
    if (lhs->kind != TypeKind::Named || rhs->kind != TypeKind::Named) {
        if (unify_hm(lhs, rhs)) return deep_resolve(lhs);
        return lhs->equals(rhs.get()) ? lhs : nullptr;
    }
    const auto left = std::static_pointer_cast<NamedType>(lhs);
    const auto right = std::static_pointer_cast<NamedType>(rhs);
    if (left->name != right->name || left->args.size() != right->args.size()) return nullptr;
    std::vector<std::shared_ptr<Type>> args;
    args.reserve(left->args.size());
    for (size_t i = 0; i < left->args.size(); ++i) {
        auto unified = unify_types(left->args[i], right->args[i]);
        if (!unified) return nullptr;
        args.push_back(std::move(unified));
    }
    return type_pool.named(left->name, std::move(args));
}

bool contains_unknown_type(const std::shared_ptr<Type>& type) noexcept {
    auto resolved = resolve_hm(type);
    if (!resolved) return false;
    if (resolved->kind == TypeKind::Unknown) return true;
    if (resolved->kind == TypeKind::Nullable)
        return contains_unknown_type(std::static_pointer_cast<NullableType>(resolved)->value_type);
    if (resolved->kind == TypeKind::Tuple) {
        const auto tuple = std::static_pointer_cast<TupleType>(resolved);
        return std::any_of(tuple->tys.begin(), tuple->tys.end(), contains_unknown_type);
    }
    if (resolved->kind == TypeKind::Function) {
        const auto f = std::static_pointer_cast<FunctionType>(resolved);
        for (const auto& p : f->params_ty) if (contains_unknown_type(p)) return true;
        return contains_unknown_type(f->ret_ty);
    }
    if (resolved->kind == TypeKind::LambdaFunction) {
        const auto f = std::static_pointer_cast<LambdaFunctionType>(resolved);
        for (const auto& p : f->params_ty) if (contains_unknown_type(p)) return true;
        if (contains_unknown_type(f->ret_ty)) return true;
        for (const auto& c : f->capture_tys) if (contains_unknown_type(c)) return true;
        return false;
    }
    if (resolved->kind == TypeKind::Array)
        return contains_unknown_type(std::static_pointer_cast<ArrayType>(resolved)->type);
    if (resolved->kind != TypeKind::Named) return false;
    const auto named = std::static_pointer_cast<NamedType>(resolved);
    return std::any_of(named->args.begin(), named->args.end(), contains_unknown_type);
}

bool contains_adt_unknown_args(const std::shared_ptr<Type>& type) noexcept {
    auto resolved = resolve_hm(type);
    if (!resolved) return false;
    auto recurse = [&](const std::vector<std::shared_ptr<Type>>& list) {
        for (const auto& t : list) if (contains_adt_unknown_args(t)) return true;
        return false;
    };
    switch (resolved->kind) {
    case TypeKind::Nullable:
        return contains_adt_unknown_args(std::static_pointer_cast<NullableType>(resolved)->value_type);
    case TypeKind::Tuple:
        return recurse(std::static_pointer_cast<TupleType>(resolved)->tys);
    case TypeKind::Array:
        return contains_adt_unknown_args(std::static_pointer_cast<ArrayType>(resolved)->type);
    case TypeKind::Function: {
        auto f = std::static_pointer_cast<FunctionType>(resolved);
        return recurse(f->params_ty) || contains_adt_unknown_args(f->ret_ty);
    }
    case TypeKind::LambdaFunction: {
        auto f = std::static_pointer_cast<LambdaFunctionType>(resolved);
        return recurse(f->params_ty) || contains_adt_unknown_args(f->ret_ty) || recurse(f->capture_tys);
    }
    case TypeKind::NativeFunction: {
        auto f = std::static_pointer_cast<NativeFunctionType>(resolved);
        return recurse(f->params_ty) || contains_adt_unknown_args(f->ret_ty);
    }
    case TypeKind::Named: {
        auto n = std::static_pointer_cast<NamedType>(resolved);
        for (const auto& a : n->args) {
            auto ra = resolve_hm(a);
            if (ra && ra->kind == TypeKind::Unknown) return true;
            if (contains_adt_unknown_args(a)) return true;
        }
        return false;
    }
    default:
        return false;
    }
}

std::shared_ptr<Type> instantiate_adt_type(const std::shared_ptr<Type>& type,
                                           const TypeBindings& bindings) noexcept {
    if (!type) return type;
    if (type->kind == TypeKind::Nullable)
        return type_pool.nullable(instantiate_adt_type(
            std::static_pointer_cast<NullableType>(type)->value_type, bindings));
    if (type->kind != TypeKind::Named) return type;
    const auto named = std::static_pointer_cast<NamedType>(type);
    if (const auto it = bindings.find(named->name); it != bindings.end() && named->args.empty()) return it->second;
    std::vector<std::shared_ptr<Type>> args;
    args.reserve(named->args.size());
    for (const auto& arg : named->args) args.push_back(instantiate_adt_type(arg, bindings));
    return type_pool.named(named->name, std::move(args));
}

} // namespace

Scope::Scope(std::string name) noexcept : name(std::move(name)) {}

Scope::Scope(const ScopeType scope) noexcept : scope(scope) {}

std::optional<Scope::Var *> TypeCkContext::find_var(const std::string &name) noexcept {
    for (auto& i : scope_stack | std::views::reverse) {
        for (auto& j : i.vars) {
            if (j.name == name) return &j;
        }
    }
    return std::nullopt;
}
std::optional<Scope::Var *> TypeCkContext::find_global(const std::string &name) noexcept {
    for (auto& i : global_scope) {
        if (i.name == name) return &i;
    }
    return std::nullopt;
}

TypeDeclNode* TypeCkContext::find_module_adt(ModuleType* module, const std::string& name) noexcept {
    if (!module) return nullptr;
    for (const auto& declaration : module->adt_exports) {
        if (declaration->name == name || declaration->qualified_name == name) return declaration.get();
    }
    return nullptr;
}

std::pair<TypeDeclNode*, AdtConstructorDecl*> TypeCkContext::find_module_constructor(
    ModuleType* module, const std::string& name) noexcept {
    if (!module) return {nullptr, nullptr};
    for (const auto& declaration : module->adt_exports) {
        for (auto& constructor : declaration->constructors) {
            if (constructor.name == name) return {declaration.get(), &constructor};
        }
    }
    return {nullptr, nullptr};
}

std::shared_ptr<Type> TypeCkContext::resolve_type(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return type;
    if (type->kind == TypeKind::Dimensioned) {
        const auto dimensioned = std::static_pointer_cast<DimensionedType>(type);
        if (dimensioned->resolved) return type;
        const auto resolved = unit_system.resolve(dimensioned->syntax);
        if (!resolved) {
            throw_error(ErrorType::Analysis,
                        "UnitInvalid: unknown or invalid unit expression `" +
                            dimensioned->syntax.to_string() + "`", 0, 0);
            return type_pool.unknown();
        }
        return resolved->dimension.is_dimensionless()
            ? type_pool.basic(runtime::ValueKind::Fraction)
            : type_pool.dimensioned(*resolved);
    }
    if (type->kind == TypeKind::Named) {
        const auto named = std::static_pointer_cast<NamedType>(type);
        std::vector<std::shared_ptr<Type>> args;
        args.reserve(named->args.size());
        for (const auto& arg : named->args) args.push_back(resolve_type(arg));
        if (const auto it = adt_types.find(named->name); it != adt_types.end()) {
            if (args.size() != it->second->type_params.size()) {
                throw_error(ErrorType::Analysis, "type `" + named->name + "` expects " +
                            std::to_string(it->second->type_params.size()) + " argument(s)", 0, 0);
            }
            return type_pool.named(it->second->qualified_name, std::move(args));
        }
        if (named->name == "set") {
            if (args.size() != 1) {
                throw_error(ErrorType::Analysis,
                            "type `set` expects 1 argument(s)", 0, 0);
                return type_pool.named("set", std::move(args));
            }
            const bool unresolved_type_parameter =
                args.front()->kind == TypeKind::Named &&
                std::static_pointer_cast<NamedType>(args.front())->args.empty() &&
                !adt_types.contains(
                    std::static_pointer_cast<NamedType>(args.front())->name);
            if (args.front()->kind != TypeKind::Unknown &&
                !unresolved_type_parameter &&
                !is_equality_comparable(args.front())) {
                throw_error(ErrorType::Analysis,
                            "SetElementNotHashable", 0, 0);
            }
            return type_pool.named("set", std::move(args));
        }
        if (const auto dot = named->name.find('.'); dot != std::string::npos) {
            const auto module_name = named->name.substr(0, dot);
            const auto type_name = named->name.substr(dot + 1);
            if (const auto module_var = find_global(module_name);
                module_var.has_value() && (*module_var)->type->kind == TypeKind::Module) {
                auto module = std::static_pointer_cast<ModuleType>((*module_var)->type);
                if (auto* declaration = find_module_adt(module.get(), type_name)) {
                    if (args.size() != declaration->type_params.size()) {
                        throw_error(ErrorType::Analysis, "type `" + named->name + "` expects " +
                                    std::to_string(declaration->type_params.size()) + " argument(s)", 0, 0);
                    }
                    return type_pool.named(declaration->qualified_name, std::move(args));
                }
            }
        }
        return type_pool.named(named->name, std::move(args));
    }
    if (type->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(type);
        return type_pool.nullable(resolve_type(nullable->value_type));
    }
    if (type->kind == TypeKind::Array) {
        const auto array = std::static_pointer_cast<ArrayType>(type);
        return type_pool.array(resolve_type(array->type));
    }
    if (type->kind == TypeKind::Tuple) {
        const auto tuple = std::static_pointer_cast<TupleType>(type);
        std::vector<std::shared_ptr<Type>> elements;
        elements.reserve(tuple->tys.size());
        for (const auto& element : tuple->tys) elements.push_back(resolve_type(element));
        return type_pool.tuple(std::move(elements));
    }
    if (type->kind == TypeKind::Function) {
        const auto function = std::static_pointer_cast<FunctionType>(type);
        std::vector<std::shared_ptr<Type>> params;
        params.reserve(function->params_ty.size());
        for (const auto& param : function->params_ty) params.push_back(resolve_type(param));
        return type_pool.function(std::move(params), resolve_type(function->ret_ty));
    }
    if (type->kind == TypeKind::NativeFunction) {
        const auto function = std::static_pointer_cast<NativeFunctionType>(type);
        std::vector<std::shared_ptr<Type>> params;
        params.reserve(function->params_ty.size());
        for (const auto& param : function->params_ty) params.push_back(resolve_type(param));
        return type_pool.native_function(std::move(params), resolve_type(function->ret_ty), function->name);
    }
    return type;
}
bool TypeCkContext::is_equality_comparable(const std::shared_ptr<Type>& type) noexcept {
    std::unordered_set<std::string> visiting;
    std::function<bool(const std::shared_ptr<Type>&)> comparable;
    comparable = [&](const std::shared_ptr<Type>& current) {
        if (!current || current->kind == TypeKind::Unknown || current->kind == TypeKind::None ||
            current->kind == TypeKind::Function || current->kind == TypeKind::NativeFunction ||
            current->kind == TypeKind::Module || current->kind == TypeKind::AdtConstructor ||
            current->kind == TypeKind::Array) return false;
        if (current->kind == TypeKind::Dimensioned) return true;
        if (current->kind == TypeKind::Nullable)
            return comparable(std::static_pointer_cast<NullableType>(current)->value_type);
        if (current->kind == TypeKind::Tuple) {
            const auto tuple = std::static_pointer_cast<TupleType>(current);
            return std::all_of(tuple->tys.begin(), tuple->tys.end(), comparable);
        }
        if (current->kind == TypeKind::Basic) {
            return std::static_pointer_cast<BasicType>(current)->type != runtime::ValueKind::C_VaList;
        }
        if (current->kind == TypeKind::String) return true;
        if (current->kind != TypeKind::Named) return false;
        const auto named = std::static_pointer_cast<NamedType>(current);
        if (named->name == "set" || named->name == "interval") {
            return named->args.size() == 1 && comparable(named->args.front());
        }
        const auto declaration_it = adt_types.find(named->name);
        if (declaration_it == adt_types.end()) return false;
        auto* declaration = declaration_it->second;
        if (named->args.size() != declaration->type_params.size()) return false;
        const auto key = Type::to_string(current.get());
        if (!visiting.insert(key).second) return true;
        TypeBindings bindings;
        for (size_t i = 0; i < declaration->type_params.size(); ++i)
            bindings[declaration->type_params[i]] = named->args[i];
        for (const auto& constructor : declaration->constructors) {
            for (const auto& field : constructor.fields) {
                if (!comparable(instantiate_adt_type(field, bindings))) {
                    visiting.erase(key);
                    return false;
                }
            }
        }
        visiting.erase(key);
        return true;
    };
    return comparable(type);
}
TypeCkContext::TypeCkContext(ModuleResolver* module_resolver) noexcept
    : module_resolver(module_resolver) {
    scope_stack.emplace_back("@GLOBAL");
}

std::shared_ptr<Type> TypeCkContext::inference_type(ExprNode* type) noexcept {
    if (!type) return type_pool.unknown();
    switch (type->kind) {
    case ASTKind::Literal: {
        const auto node = reinterpret_cast<LiteralNode*>(type);
        switch (node->kind) {
        case LiteralNode::Kind::Integer: {
            return type_pool.basic(runtime::ValueKind::Int);
        }
        case LiteralNode::Kind::Float: {
            return type_pool.basic(runtime::ValueKind::Fraction);
        }
        case LiteralNode::Kind::String: {
            return type_pool.string();
        }
        case LiteralNode::Kind::Boolean: {
            return type_pool.basic(runtime::ValueKind::Bool);
        }
        case LiteralNode::Kind::Null: {
            return type_pool.basic(runtime::ValueKind::Null);
        }
        }
        break;
    }
    case ASTKind::Identifier: {
        const auto node = reinterpret_cast<IdentifierNode*>(type);
        if (node->id == "I") {
            return type_pool.basic(runtime::ValueKind::Expr);
        }
        if (node->type && node->type->kind != TypeKind::Unknown) return node->type;
        if (find_var(node->id).has_value()) {
            auto var = *find_var(node->id);
            if (var->scheme.has_value()) return instantiate_scheme(*var->scheme);
            return var->type;
        }
        if (find_global(node->id).has_value()) {
            auto var = *find_global(node->id);
            if (var->scheme.has_value()) return instantiate_scheme(*var->scheme);
            return var->type;
        }
        break;
    }
    case ASTKind::Unary: {
        const auto node = reinterpret_cast<UnaryNode*>(type);
        if (const auto t = inference_type(node->expr.get());
            t &&
            t->kind == TypeKind::Basic
            ) {
            if (const auto t2 = std::reinterpret_pointer_cast<BasicType>(t)->type;
                t2 == runtime::ValueKind::Int ||
                t2 == runtime::ValueKind::Fraction ||
                t2 == runtime::ValueKind::Expr) {

                return type_pool.basic(t2);
            }
        }
        break;
    }
    case ASTKind::Binary: {
        const auto node = reinterpret_cast<BinaryNode*>(type);
        auto left_ty = inference_type(node->lhs.get());
        const auto right_ty = inference_type(node->rhs.get());
        if (node->op == BinaryNode::Op::Bind) {
            return type_pool.named("Binding");
        }
        if (node->op == BinaryNode::Op::In ||
            node->op == BinaryNode::Op::NotIn ||
            node->op == BinaryNode::Op::Subset) {
            return is_expr_type(left_ty) || is_expr_type(right_ty)
                ? type_pool.basic(runtime::ValueKind::Expr)
                : type_pool.basic(runtime::ValueKind::Bool);
        }
        if (node->op == BinaryNode::Op::SetUnion ||
            node->op == BinaryNode::Op::SetIntersection ||
            node->op == BinaryNode::Op::SetSymmetricDifference ||
            (node->op == BinaryNode::Op::Sub &&
             (is_named_type(left_ty, "set") ||
              is_named_type(right_ty, "set"))))
            return left_ty;
        if (is_expr_type(left_ty) || is_expr_type(right_ty)) {
            return type_pool.basic(runtime::ValueKind::Expr);
        }
        if (left_ty->equals(right_ty.get())) return left_ty;
        break;
    }
    case ASTKind::LiteralPayload: {
        return literal_payload_type(*reinterpret_cast<LiteralPayloadNode*>(type));
    }
    case ASTKind::UnitAnnotated:
        return type->type ? type->type : type_pool.unknown();
    case ASTKind::MatchExpr: {
        return type->type ? type->type : type_pool.unknown();
    }
    case ASTKind::Block: {
        if (const auto node = reinterpret_cast<BlockExprNode*>(type);
            node->stmts.back()->kind == ASTKind::TailReturn)
        {
            const auto tail_ret = std::reinterpret_pointer_cast<TailReturnNode>(node->stmts.back());
            if (tail_ret->expr &&
                !Type::is_null_type(tail_ret->expr->type.get()) &&
                tail_ret->expr->type->kind != TypeKind::Unknown

                ) return tail_ret->expr->type;
            return inference_type(tail_ret->expr.get());
        } //否则就是Block没有返回值
        return type_pool.none();
        break;
    }
    case ASTKind::SuffixParen: {
        const auto node = reinterpret_cast<SuffixParenNode*>(type);
        const auto callee_ty_orig = node->expr && !Type::is_null_type(node->expr->type.get())
            ? node->expr->type
            : inference_type(node->expr.get());
        if (callee_ty_orig) {
            const auto callee_ty = resolve_hm(callee_ty_orig);
            if (callee_ty->kind == TypeKind::Function)
                return std::reinterpret_pointer_cast<FunctionType>(callee_ty)->ret_ty;
            if (callee_ty->kind == TypeKind::LambdaFunction)
                return std::reinterpret_pointer_cast<LambdaFunctionType>(callee_ty)->ret_ty;
            if (callee_ty->kind == TypeKind::NativeFunction)
                return std::reinterpret_pointer_cast<NativeFunctionType>(callee_ty)->ret_ty;
            if (callee_ty->kind == TypeKind::TypeVariable) {
                auto ret = type_pool.fresh_type_variable();
                std::vector<std::shared_ptr<Type>> params;
                const size_t n = node->suffix ? node->suffix->exprs.size() : 0;
                for (size_t i = 0; i < n; ++i)
                    params.push_back(std::static_pointer_cast<Type>(type_pool.fresh_type_variable()));
                auto lf = type_pool.lambda_function(std::move(params), std::static_pointer_cast<Type>(ret), {});
                unify_hm(callee_ty, lf);
                return std::static_pointer_cast<Type>(ret);
            }
        }
        return type_pool.unknown();
    }
    case ASTKind::SuffixBracket: {
        const auto node = reinterpret_cast<SuffixBracketNode*>(type);
        const auto left_t = inference_type(node->expr.get());
        if (left_t->kind == TypeKind::Array) {
            return std::reinterpret_pointer_cast<ArrayType>(left_t)->type;
        }
        break;
    }
    case ASTKind::IfExpr: {
        const auto node = reinterpret_cast<IfExprNode*>(type);

        return inference_type(node->then.get());
    }
    case ASTKind::AsExpr: {
        const auto node = reinterpret_cast<AsExprNode*>(type);
        return node->type ? node->type
                          : (node->cast_type ? node->cast_type : type_pool.unknown());
    }
    case ASTKind::DotExpr: {
        const auto node = reinterpret_cast<DotExprNode*>(type);
        const auto left_type = inference_type(node->expr.get());
        if (!left_type || left_type->kind != TypeKind::Module)
            return type_pool.unknown();
        const auto left = std::static_pointer_cast<ModuleType>(left_type);
        const auto found = left->find_var(node->rhs->id);
        return found ? (*found)->type : type_pool.unknown();
    }
    case ASTKind::NativeFuncCall: {
        const auto node = reinterpret_cast<NativeFuncCallExpr*>(type);
        const auto left_ty = std::reinterpret_pointer_cast<NativeFunctionType>(inference_type(node->expr.get()));
        return left_ty->ret_ty;
        break;
    }
    case ASTKind::ArrayLiteral: {
        const auto node = reinterpret_cast<ArrayLiteralNode*>(type);
        if (node->exprs.empty()) return type_pool.array(type_pool.unknown());
        const auto elem_ty = node->exprs[0]->type->kind == TypeKind::Unknown
            ? inference_type(node->exprs[0].get())
            : node->exprs[0]->type;
        for (size_t i = 1; i < node->exprs.size(); i++) {
            const auto& ety = node->exprs[i]->type->kind == TypeKind::Unknown
                ? inference_type(node->exprs[i].get())
                : node->exprs[i]->type;
            if (!elem_ty->equals(ety.get())) return type_pool.unknown();
        }
        if (Type::is_null_type(elem_ty.get())) return type_pool.unknown();
        return type_pool.array(elem_ty);
    }
    case ASTKind::TupleLiteral: {
        const auto node = reinterpret_cast<TupleLiteralNode*>(type);
        std::vector<std::shared_ptr<Type>> elements;
        elements.reserve(node->exprs.size());
        for (const auto& expression : node->exprs) {
            auto element = expression->type && expression->type->kind != TypeKind::Unknown
                ? expression->type : inference_type(expression.get());
            if (!element || Type::is_null_type(element.get())) return type_pool.unknown();
            elements.push_back(std::move(element));
        }
        return type_pool.tuple(std::move(elements));
    }
    case ASTKind::TupleGetExpr: {
        break;
    }
    case ASTKind::LambdaExpr: {
        const auto node = reinterpret_cast<LambdaExprNode*>(type);
        if (node->type && !Type::is_null_type(node->type.get()) && node->type->kind != TypeKind::Unknown) {
            return node->type;
        }
        return inference_type(node->body.get());
    }
    default: std::unreachable();
    }
    return type_pool.unknown();
}

static bool has_explicit_return(ExprNode *expr) noexcept {
    if (!expr) return false;
    if (expr->kind == ASTKind::Return) return true;
    if (expr->kind == ASTKind::Block) {
        auto *blk = static_cast<BlockExprNode*>(expr);
        for (const auto& s : blk->stmts) {
            if (!s) continue;
            if (s->kind == ASTKind::Return) return true;
            if (s->kind == ASTKind::ExprStmt) {
                auto *es = static_cast<ExprStmtNode*>(s.get());
                if (es->expr && has_explicit_return(es->expr.get())) return true;
            }
            if (s->kind == ASTKind::IfExpr) {
                auto *ie = reinterpret_cast<IfExprNode*>(s.get());
                if (has_explicit_return(static_cast<ExprNode*>(ie))) return true;
            }
        }
    }
    if (expr->kind == ASTKind::IfExpr) {
        auto *ie = static_cast<IfExprNode*>(expr);
        if (has_explicit_return(ie->then.get())) return true;
        if (ie->els && has_explicit_return(ie->els.get())) return true;
    }
    return false;
}


static void collect_referenced_identifiers(
    ExprNode *expr,
    std::unordered_set<std::string> &names
) noexcept {
    if (!expr) return;
    switch (expr->kind) {
    case ASTKind::Identifier: {
        const auto *id = static_cast<IdentifierNode*>(expr);
        names.insert(id->id);
        break;
    }
    case ASTKind::Binary: {
        const auto *b = static_cast<BinaryNode*>(expr);
        collect_referenced_identifiers(b->lhs.get(), names);
        collect_referenced_identifiers(b->rhs.get(), names);
        break;
    }
    case ASTKind::Unary: {
        const auto *u = static_cast<UnaryNode*>(expr);
        collect_referenced_identifiers(u->expr.get(), names);
        break;
    }
    case ASTKind::PipeExpr: {
        const auto *p = static_cast<PipeExprNode*>(expr);
        collect_referenced_identifiers(p->lhs.get(), names);
        collect_referenced_identifiers(p->rhs.get(), names);
        break;
    }
    case ASTKind::LambdaExpr: {
        const auto *l = static_cast<LambdaExprNode*>(expr);
        std::unordered_set<std::string> inner_params;
        if (l->params) {
            for (const auto& [pname, _] : l->params->stmts) {
                inner_params.insert(pname);
            }
        }
        std::unordered_set<std::string> inner_names;
        collect_referenced_identifiers(l->body.get(), inner_names);
        for (const auto& n : inner_names) {
            if (inner_params.count(n) == 0) {
                names.insert(n);
            }
        }
        break;
    }
    case ASTKind::IfExpr: {
        const auto *i = static_cast<IfExprNode*>(expr);
        collect_referenced_identifiers(i->cond.get(), names);
        collect_referenced_identifiers(i->then.get(), names);
        if (i->els) collect_referenced_identifiers(i->els.get(), names);
        break;
    }
    case ASTKind::Block: {
        const auto *blk = static_cast<BlockExprNode*>(expr);
        for (const auto& s : blk->stmts) {
            if (!s) continue;
            if (s->kind == ASTKind::Return) {
                const auto *ret = static_cast<ReturnNode*>(s.get());
                collect_referenced_identifiers(ret->expr.get(), names);
            } else if (s->kind == ASTKind::TailReturn) {
                const auto *ret = static_cast<TailReturnNode*>(s.get());
                collect_referenced_identifiers(ret->expr.get(), names);
            } else if (s->kind == ASTKind::ExprStmt) {
                const auto *es = static_cast<ExprStmtNode*>(s.get());
                collect_referenced_identifiers(es->expr.get(), names);
            } else if (s->kind == ASTKind::VarDecl) {
                // var 声明中的变量名是局部的，不捕获；但初始化达式中的引用需要收集
                const auto *vd = static_cast<VarDeclNode*>(s.get());
                collect_referenced_identifiers(vd->init_value.get(), names);
            } else if (s->kind == ASTKind::AssignStmt) {
                const auto *as = static_cast<AssignStmtNode*>(s.get());
                collect_referenced_identifiers(as->lhs.get(), names);
                collect_referenced_identifiers(as->rhs.get(), names);
            } else if (s->kind == ASTKind::LoopStmt) {
                const auto *lp = static_cast<LoopStmtNode*>(s.get());
                collect_referenced_identifiers(lp->expr.get(), names);
                for (const auto& bs : lp->body) {
                    if (bs && bs->kind == ASTKind::ExprStmt) {
                        collect_referenced_identifiers(
                            static_cast<ExprStmtNode*>(bs.get())->expr.get(), names);
                    }
                }
            } else if (s->kind == ASTKind::IfExpr) {
                collect_referenced_identifiers(reinterpret_cast<ExprNode*>(s.get()), names);
            }
        }
        break;
    }
    case ASTKind::SuffixParen: {
        const auto *sp = static_cast<SuffixParenNode*>(expr);
        collect_referenced_identifiers(sp->expr.get(), names);
        if (sp->suffix) {
            for (const auto& a : sp->suffix->exprs) {
                collect_referenced_identifiers(a.get(), names);
            }
        }
        break;
    }
    case ASTKind::SuffixBracket: {
        const auto *sb = static_cast<SuffixBracketNode*>(expr);
        collect_referenced_identifiers(sb->expr.get(), names);
        collect_referenced_identifiers(sb->suffix.get(), names);
        break;
    }
    case ASTKind::AsExpr: {
        const auto *ae = static_cast<AsExprNode*>(expr);
        collect_referenced_identifiers(ae->expr.get(), names);
        break;
    }
    case ASTKind::DotExpr: {
        // DotExpr 的 rhs 是字段名而非变量，只收集 lhs
        const auto *de = static_cast<DotExprNode*>(expr);
        collect_referenced_identifiers(de->expr.get(), names);
        break;
    }
    default:
        break;
    }
}

static void collect_param_constraints(
    ExprNode *expr,
    const std::unordered_set<std::string> &param_names,
    std::unordered_map<std::string, std::vector<std::shared_ptr<Type>>> &candidates
) noexcept {
    if (!expr) return;
    switch (expr->kind) {
    case ASTKind::Binary: {
        auto *b = static_cast<BinaryNode*>(expr);
        collect_param_constraints(b->lhs.get(), param_names, candidates);
        collect_param_constraints(b->rhs.get(), param_names, candidates);
        auto add_constraint_from_other = [&](ExprNode *which, ExprNode *other) {
            if (which->kind == ASTKind::Identifier) {
                const auto *id = static_cast<IdentifierNode*>(which);
                if (param_names.count(id->id) > 0 && other->type &&
                    other->type->kind != TypeKind::Unknown &&
                    other->type->kind != TypeKind::None) {
                    candidates[id->id].push_back(other->type);
                }
            }
        };
        add_constraint_from_other(b->lhs.get(), b->rhs.get());
        add_constraint_from_other(b->rhs.get(), b->lhs.get());
        break;
    }
    case ASTKind::Unary: {
        auto *u = static_cast<UnaryNode*>(expr);
        collect_param_constraints(u->expr.get(), param_names, candidates);
        break;
    }
    case ASTKind::PipeExpr: {
        auto *p = static_cast<PipeExprNode*>(expr);
        if (!p->lhs || !p->rhs) break;
        collect_param_constraints(p->lhs.get(), param_names, candidates);
        collect_param_constraints(p->rhs.get(), param_names, candidates);
        if (p->rhs->kind == ASTKind::LambdaExpr && p->lhs->type &&
            p->lhs->type->kind != TypeKind::Unknown &&
            p->lhs->type->kind != TypeKind::None) {
            auto *lambda = static_cast<LambdaExprNode*>(p->rhs.get());
            if (lambda->params && !lambda->params->stmts.empty()) {
                const auto &first = lambda->params->stmts.front();
                candidates[first.first].push_back(p->lhs->type);
            }
        }
        break;
    }
    case ASTKind::LambdaExpr: {
        auto *l = static_cast<LambdaExprNode*>(expr);
        collect_param_constraints(l->body.get(), param_names, candidates);
        break;
    }
    case ASTKind::IfExpr: {
        auto *i = static_cast<IfExprNode*>(expr);
        collect_param_constraints(i->cond.get(), param_names, candidates);
        collect_param_constraints(i->then.get(), param_names, candidates);
        if (i->els) collect_param_constraints(i->els.get(), param_names, candidates);
        break;
    }
    case ASTKind::Block: {
        auto *blk = static_cast<BlockExprNode*>(expr);
        for (auto &s : blk->stmts) {
            if (!s) continue;
            if (s->kind == ASTKind::Return) {
                auto *ret = static_cast<ReturnNode*>(s.get());
                collect_param_constraints(ret->expr.get(), param_names, candidates);
            } else if (s->kind == ASTKind::TailReturn) {
                auto *ret = static_cast<TailReturnNode*>(s.get());
                collect_param_constraints(ret->expr.get(), param_names, candidates);
            } else if (s->kind == ASTKind::ExprStmt) {
                auto *es = static_cast<ExprStmtNode*>(s.get());
                collect_param_constraints(es->expr.get(), param_names, candidates);
            } else if (s->kind == ASTKind::VarDecl) {
                auto *vd = static_cast<VarDeclNode*>(s.get());
                collect_param_constraints(vd->init_value.get(), param_names, candidates);
            } else if (s->kind == ASTKind::AssignStmt) {
                auto *as = static_cast<AssignStmtNode*>(s.get());
                collect_param_constraints(as->lhs.get(), param_names, candidates);
                collect_param_constraints(as->rhs.get(), param_names, candidates);
                if (as->lhs->kind == ASTKind::Identifier) {
                    const auto *lid = static_cast<IdentifierNode*>(as->lhs.get());
                    if (param_names.count(lid->id) > 0 && as->rhs->type &&
                        as->rhs->type->kind != TypeKind::Unknown &&
                        as->rhs->type->kind != TypeKind::None) {
                        candidates[lid->id].push_back(as->rhs->type);
                    }
                }
            } else if (s->kind == ASTKind::LoopStmt) {
                auto *lp = static_cast<LoopStmtNode*>(s.get());
                collect_param_constraints(lp->expr.get(), param_names, candidates);
                for (auto &bs : lp->body) {
                    if (bs->kind == ASTKind::ExprStmt) {
                        collect_param_constraints(static_cast<ExprStmtNode*>(bs.get())->expr.get(),
                                                  param_names, candidates);
                    }
                }
            } else if (s->kind == ASTKind::IfExpr) {
                auto *if_stmt_s = reinterpret_cast<IfExprNode*>(s.get());
                collect_param_constraints(static_cast<ExprNode*>(if_stmt_s), param_names, candidates);
            }
        }
        break;
    }
    case ASTKind::SuffixParen: {
        auto *sp = static_cast<SuffixParenNode*>(expr);
        collect_param_constraints(sp->expr.get(), param_names, candidates);
        if (sp->suffix) {
            for (auto &a : sp->suffix->exprs) collect_param_constraints(a.get(), param_names, candidates);
        }
        const auto callee_ty = sp->expr && !Type::is_null_type(sp->expr->type.get())
            ? sp->expr->type : nullptr;
        const std::vector<std::shared_ptr<Type>> *formal_params = nullptr;
        if (callee_ty && callee_ty->kind == TypeKind::Function)
            formal_params = &static_cast<FunctionType*>(callee_ty.get())->params_ty;
        else if (callee_ty && callee_ty->kind == TypeKind::LambdaFunction)
            formal_params = &static_cast<LambdaFunctionType*>(callee_ty.get())->params_ty;
        if (formal_params && sp->suffix) {
            const size_t N = std::min(formal_params->size(), sp->suffix->exprs.size());
            for (size_t i = 0; i < N; i++) {
                auto &arg = sp->suffix->exprs[i];
                auto &ft = (*formal_params)[i];
                if (!ft || ft->kind == TypeKind::Unknown || ft->kind == TypeKind::None) continue;
                if (arg->kind == ASTKind::Identifier) {
                    const auto *aid = static_cast<IdentifierNode*>(arg.get());
                    if (param_names.count(aid->id) > 0) candidates[aid->id].push_back(ft);
                }
            }
        }
        break;
    }
    default:
        break;
    }
}

static void narrow_lambda_param_types(LambdaExprNode *lambda, size_t line, size_t col) noexcept {
    std::unordered_set<std::string> param_names;
    for (auto &[name, _] : lambda->params->stmts) param_names.insert(name);

    std::unordered_map<std::string, std::vector<std::shared_ptr<Type>>> candidates;
    for (auto &[name, _] : lambda->params->stmts) candidates[name] = {};

    collect_param_constraints(lambda->body.get(), param_names, candidates);

    for (auto &[pname, pty] : lambda->params->stmts) {
        auto it = candidates.find(pname);
        if (it == candidates.end()) continue;
        auto &cands = it->second;
        std::vector<std::shared_ptr<Type>> uniq;
        for (auto &c : cands) {
            bool dup = false;
            for (auto &u : uniq) if (u->equals(c.get())) { dup = true; break; }
            if (!dup) uniq.push_back(c);
        }
        if (uniq.empty()) continue;
      
        auto inferred = uniq.front();
        if (Type::is_null_type(pty.get()) || pty->kind == TypeKind::Unknown) {
            pty = inferred;
        } else if (!pty->equals(inferred.get())) {
            throw_error(ErrorType::Analysis,
                "lambda param `" + pname + "` declared type (" + Type::to_string(pty.get()) +
                ") mismatches inferred type (" + Type::to_string(inferred.get()) + ")",
                line, col);
            return;
        }
    }
}

static std::shared_ptr<ExprNode> make_lambda_dummy_return(size_t line, size_t col) noexcept {
    return std::make_shared<LiteralNode>(line, col, "0", LiteralNode::Kind::Integer);
}

static std::shared_ptr<StmtNode> sugar_loop_count(const std::shared_ptr<LoopStmtNode>& stmt) noexcept {
    std::string name = "@loop_cnt_id";
    auto var_cnt = std::make_shared<VarDeclNode>(0, 0, name, type_pool.basic(runtime::ValueKind::Int), true);
    var_cnt->init_value = std::move(stmt->expr);


    const auto lhs = std::make_shared<IdentifierNode>(0, 0, name);
    const auto rhs = std::make_shared<LiteralNode>(0, 0, "0", LiteralNode::Kind::Integer);
    auto break_cond = std::make_shared<BinaryNode>(0, 0, lhs, BinaryNode::Op::Eq, rhs);

    auto break_stmt_block = std::make_shared<BlockExprNode>(0, 0, decltype(BlockExprNode::stmts){std::make_shared<BreakStmtNode>(0, 0)});

    auto break_if = std::make_shared<IfExprNode>(0, 0, break_cond, break_stmt_block, nullptr);

    stmt->body.insert(stmt->body.begin(), std::make_shared<ExprStmtNode>(0, 0, break_if));

    const auto one = std::make_shared<LiteralNode>(0, 0, "1", LiteralNode::Kind::Integer);
    const auto dec_cnt = std::make_shared<AssignStmtNode>(0, 0, lhs, std::make_shared<BinaryNode>(0, 0, lhs, BinaryNode::Op::Sub, one));
    stmt->body.insert(stmt->body.end(), dec_cnt);
    decltype(BlockExprNode::stmts) block{var_cnt, stmt};
    auto result = std::make_shared<ExprStmtNode>(
        stmt->line, stmt->col, std::make_shared<BlockExprNode>(stmt->line, stmt->col, std::move(block)));
    return result;
}

std::vector<Scope::Var> TypeCkContext::check_module(const std::shared_ptr<Module> &mod) noexcept {
    const auto save_cur_module = cur_module;
    cur_module = mod;
    mod->adt_exports.clear();
    mod->unit_exports.clear();

    if (!mod->native_funcs.empty() && mod->lib_name.empty()) {
        throw_error(ErrorType::Analysis, "module not `static` declare dynamic library, cannot declare native function", 0 , 0);

    }
    static const auto builtin_adts = [] {
        std::vector<std::unique_ptr<TypeDeclNode>> declarations;
        declarations.push_back(std::make_unique<TypeDeclNode>(0, 0, "Option",
            std::vector<std::string>{"T"},
            std::vector<AdtConstructorDecl>{
                {"Some", {type_pool.named("T")}},
                {"None", {}}
            }));
        declarations.push_back(std::make_unique<TypeDeclNode>(0, 0, "Result",
            std::vector<std::string>{"T", "E"},
            std::vector<AdtConstructorDecl>{
                {"Ok", {type_pool.named("T")}},
                {"Err", {type_pool.named("E")}}
            }));
        declarations.push_back(std::make_unique<TypeDeclNode>(0, 0, "Binding",
            std::vector<std::string>{"K", "V"},
            std::vector<AdtConstructorDecl>{{"Binding", {
                type_pool.named("K"), type_pool.named("V")
            }}}));
        return declarations;
    }();
    for (const auto& declaration_ptr : builtin_adts) {
        auto* declaration = declaration_ptr.get();
        adt_types[declaration->name] = declaration;
        for (auto& constructor : declaration->constructors) {
            adt_constructors[constructor.name] = {declaration, &constructor};
            new_global_var(constructor.name, type_pool.adt_constructor(
                declaration->qualified_name, constructor.name, declaration->type_params, constructor.fields));
        }
    }
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::ImportStmt) check_stmt(node);
    }
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::UnitDecl) check_stmt(node);
    }
    for (const auto& node : mod->decls) {
        if (node->kind != ASTKind::TypeDecl) continue;
        auto* declaration = reinterpret_cast<TypeDeclNode*>(node.get());
        if (adt_types.contains(declaration->name)) {
            throw_error(ErrorType::Analysis, "duplicate ADT `" + declaration->name + "`", declaration->line, declaration->col);
            continue;
        }
        declaration->qualified_name = mod->name + "::" + declaration->name;
        adt_types[declaration->name] = declaration;
        adt_types[declaration->qualified_name] = declaration;
        mod->adt_exports.push_back(std::static_pointer_cast<TypeDeclNode>(node));
        for (auto& constructor : declaration->constructors) {
            if (adt_constructors.contains(constructor.name)) {
                throw_error(ErrorType::Analysis, "duplicate constructor `" + constructor.name + "`", declaration->line, declaration->col);
                continue;
            }
            adt_constructors[constructor.name] = {declaration, &constructor};
            new_global_var(constructor.name, type_pool.adt_constructor(
                declaration->qualified_name, constructor.name, declaration->type_params, constructor.fields));
        }
    }
    for (const auto& node : mod->decls) {
        if (node->kind != ASTKind::TypeDecl) continue;
        auto* declaration = reinterpret_cast<TypeDeclNode*>(node.get());
        std::unordered_set<std::string> parameters;
        for (const auto& parameter : declaration->type_params) {
            if (!parameters.insert(parameter).second)
                throw_error(ErrorType::Analysis, "duplicate type parameter `" + parameter + "`", declaration->line, declaration->col);
        }
        std::function<void(const std::shared_ptr<Type>&)> validate_type;
        validate_type = [&](const std::shared_ptr<Type>& type) {
            if (!type) return;
            if (type->kind == TypeKind::Nullable) {
                validate_type(std::static_pointer_cast<NullableType>(type)->value_type);
                return;
            }
            if (type->kind != TypeKind::Named) return;
            const auto named = std::static_pointer_cast<NamedType>(type);
            if (parameters.contains(named->name)) {
                if (!named->args.empty())
                    throw_error(ErrorType::Analysis, "type parameter `" + named->name + "` cannot have arguments", declaration->line, declaration->col);
                return;
            }
            if (named->name == "set" || named->name == "interval") {
                if (named->args.size() != 1) {
                    throw_error(ErrorType::Analysis, "type `" + named->name + "` expects 1 argument(s)",
                                declaration->line, declaration->col);
                    return;
                }
                validate_type(named->args.front());
                return;
            }
            const auto referenced = adt_types.find(named->name);
            if (referenced == adt_types.end()) {
                throw_error(ErrorType::Analysis, "unknown field type `" + named->name + "`", declaration->line, declaration->col);
                return;
            }
            if (named->args.size() != referenced->second->type_params.size()) {
                throw_error(ErrorType::Analysis, "type `" + named->name + "` expects " +
                            std::to_string(referenced->second->type_params.size()) + " argument(s)",
                            declaration->line, declaration->col);
                return;
            }
            for (const auto& argument : named->args) validate_type(argument);
        };
        for (auto& constructor : declaration->constructors) {
            for (auto& field : constructor.fields) {
                field = resolve_type(field);
                validate_type(field);
            }
            if (const auto constructor_it = adt_constructors.find(constructor.name);
                constructor_it != adt_constructors.end() && constructor_it->second.first == declaration) {
                constructor_it->second = {declaration, &constructor};
                if (const auto global = find_global(constructor.name); global.has_value()) {
                    (*global)->type = type_pool.adt_constructor(
                        declaration->qualified_name, constructor.name,
                        declaration->type_params, constructor.fields);
                }
            }
        }
    }
    std::unordered_set<std::string> local_adt_names;
    std::unordered_set<std::string> exported_adt_identities;
    for (const auto& declaration : mod->adt_exports) {
        local_adt_names.insert(declaration->name);
        exported_adt_identities.insert(declaration->qualified_name);
    }
    for (const auto& imported_module : mod->imports | std::views::values) {
        for (const auto& declaration : imported_module->adt_exports) {
            if (local_adt_names.contains(declaration->name) ||
                !exported_adt_identities.insert(
                    declaration->qualified_name).second)
                continue;
            mod->adt_exports.push_back(declaration);
        }
    }
    mod->function_slots.clear();
    std::unordered_map<std::string, std::vector<std::pair<size_t, FuncImplNode*>>>
        regular_function_groups;
    for (size_t declaration_order = 0; declaration_order < mod->decls.size();
         ++declaration_order) {
        const auto& declaration = mod->decls[declaration_order];
        if (declaration->kind != ASTKind::FuncImpl) continue;
        auto* function = static_cast<FuncImplNode*>(declaration.get());
        if (function->func_id == "raise") {
            throw_error(ErrorType::Analysis,
                        "cannot redefine builtin `raise`",
                        function->line, function->col);
        }
        for (auto& [name, type] : function->params->stmts) type = resolve_type(type);
        function->return_type = resolve_type(function->return_type);
        regular_function_groups[function->func_id].emplace_back(
            declaration_order, function);
    }
    for (const auto& [name, declarations] : regular_function_groups) {
        for (size_t i = 0; i < declarations.size(); ++i) {
            const auto* lhs = declarations[i].second;
            for (size_t j = i + 1; j < declarations.size(); ++j) {
                const auto* rhs = declarations[j].second;
                if (lhs->params->stmts.size() != rhs->params->stmts.size()) continue;
                bool duplicate = true;
                for (size_t parameter = 0; parameter < lhs->params->stmts.size();
                     ++parameter) {
                    if (!lhs->params->stmts[parameter].second->equals(
                            rhs->params->stmts[parameter].second.get())) {
                        duplicate = false;
                        break;
                    }
                }
                if (duplicate) {
                    throw_error(ErrorType::Analysis,
                                "duplicate function parameter signature `" + name + "`",
                                rhs->line, rhs->col);
                }
            }
        }
    }
    for (size_t declaration_order = 0; declaration_order < mod->decls.size();
         ++declaration_order) {
        const auto& declaration = mod->decls[declaration_order];
        if (declaration->kind != ASTKind::FuncImpl) continue;
        auto* function = static_cast<FuncImplNode*>(declaration.get());
        const auto& group = regular_function_groups[function->func_id];
        function->compiled_symbol = group.size() == 1
            ? function->func_id
            : function->func_id + "\x1f" + std::to_string(declaration_order);
        mod->function_slots.push_back(function->compiled_symbol);
    }
    mod->builtin_functions.clear();
    const auto add_raise_builtin = [&](std::string symbol,
                                       std::shared_ptr<Type> parameter_type,
                                       const bool is_export) {
        SyntheticBuiltinSpec builtin{
            SyntheticBuiltinKind::Raise,
            "raise",
            std::move(symbol),
            "value",
            std::move(parameter_type),
            type_pool.never(),
            is_export,
        };
        auto type = type_pool.function(
            {builtin.parameter_type}, builtin.return_type);
        new_global_var(builtin.source_name, std::move(type), false,
                       builtin.compiled_symbol, builtin.is_export);
        mod->function_slots.push_back(builtin.compiled_symbol);
        mod->builtin_functions.push_back(std::move(builtin));
    };
    add_raise_builtin("@builtin.raise.text", type_pool.string(), false);
    auto normalized_module_path = mod->name;
    std::ranges::replace(normalized_module_path, '\\', '/');
    if (normalized_module_path.ends_with(
            "modules/std/mathematics_error/module.lm")) {
        add_raise_builtin(
            "@builtin.raise.mathematics_error",
            resolve_type(type_pool.named("MathError")),
            true);
    }
    for (const auto& n : mod->native_funcs) {
        for (auto& [name, type] : n->params->stmts) type = resolve_type(type);
        n->return_type = resolve_type(n->return_type);
        new_global_var(n->func_id, n->make_type());
    }
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::ImportStmt || node->kind == ASTKind::UnitDecl) continue;
        check_stmt(node);
    }

    cur_module = save_cur_module;

    std::vector<Scope::Var> result;

    // Module exports include imported modules so packages can expose
    // hierarchical APIs such as std.cas and std.math.
    for (const auto& v : get_global()) {
        const auto ty = resolve_hm(v.type);
        if (!ty) continue;
        if (v.is_export &&
            (ty->kind == TypeKind::Function ||
             ty->kind == TypeKind::NativeFunction ||
             ty->kind == TypeKind::Module)) {
            result.push_back(v);
        }
    }
    return result;
}

static void
curry_freshen_intersection_tvs(
    TypePool& type_pool,
    std::vector<std::shared_ptr<Type>>& remaining_params,
    std::vector<std::shared_ptr<Type>>& remaining_captures,
    std::shared_ptr<Type>& ret_ty
) noexcept {
    std::function<void(const std::shared_ptr<Type>&, std::unordered_set<TypeVariable*>&)>
        collect_bare;
    collect_bare = [&](const std::shared_ptr<Type>& t, std::unordered_set<TypeVariable*>& out) {
        if (!t) return;
        auto r = t;
        while (r && r->kind == TypeKind::TypeVariable) {
            auto* tv = static_cast<TypeVariable*>(r.get());
            if (is_recursive_mu_head(tv)) { out.insert(tv); return; } 
            if (!tv->binding) { out.insert(tv); return; }
            r = tv->binding;
        }
        switch (r->kind) {
        case TypeKind::Function: {
            auto f = std::static_pointer_cast<FunctionType>(r);
            for (const auto& p : f->params_ty) collect_bare(p, out);
            collect_bare(f->ret_ty, out);
            return;
        }
        case TypeKind::LambdaFunction: {
            auto f = std::static_pointer_cast<LambdaFunctionType>(r);
            for (const auto& p : f->params_ty) collect_bare(p, out);
            for (const auto& c : f->capture_tys) collect_bare(c, out);
            collect_bare(f->ret_ty, out);
            return;
        }
        case TypeKind::NativeFunction: {
            auto f = std::static_pointer_cast<NativeFunctionType>(r);
            for (const auto& p : f->params_ty) collect_bare(p, out);
            collect_bare(f->ret_ty, out);
            return;
        }
        case TypeKind::Array:
            collect_bare(std::static_pointer_cast<ArrayType>(r)->type, out); return;
        case TypeKind::Tuple: {
            auto f = std::static_pointer_cast<TupleType>(r);
            for (const auto& e : f->tys) collect_bare(e, out);
            return;
        }
        case TypeKind::Nullable:
            collect_bare(std::static_pointer_cast<NullableType>(r)->value_type, out); return;
        case TypeKind::Named: {
            auto f = std::static_pointer_cast<NamedType>(r);
            for (const auto& a : f->args) collect_bare(a, out);
            return;
        }
        default: return;
        }
    };

    std::unordered_set<TypeVariable*> in_remaining, in_ret;
    for (const auto& p : remaining_params) collect_bare(p, in_remaining);
    for (const auto& c : remaining_captures) collect_bare(c, in_remaining);
    collect_bare(ret_ty, in_ret);

    std::unordered_map<TypeVariable*, std::shared_ptr<TypeVariable>> rename;
    for (auto* tv : in_remaining) {
        rename.emplace(tv, type_pool.fresh_type_variable());
    }
    for (auto* tv : in_ret) {
        rename.emplace(tv, type_pool.fresh_type_variable());
    }
    if (rename.empty()) return;  // nothing to rename.
    std::function<std::shared_ptr<Type>(const std::shared_ptr<Type>&)> apply;
    apply = [&](const std::shared_ptr<Type>& x) -> std::shared_ptr<Type> {
        if (!x) return nullptr;
        if (x->kind == TypeKind::TypeVariable) {
            std::shared_ptr<Type> r = x;
            while (r && r->kind == TypeKind::TypeVariable) {
                auto* tv = static_cast<TypeVariable*>(r.get());
                if (is_recursive_mu_head(tv)) break;  // μ-opaque: stop, don't unroll cycle
                if (!tv->binding) break;
                r = tv->binding;
            }
            if (r && r->kind == TypeKind::TypeVariable) {
                auto* root_tv = static_cast<TypeVariable*>(r.get());
                auto it = rename.find(root_tv);
                if (it != rename.end()) {
                    return std::static_pointer_cast<Type>(it->second);
                }
                return x;
            }
            return apply(r);
        }
        switch (x->kind) {
        case TypeKind::Function: {
            auto f = std::static_pointer_cast<FunctionType>(x);
            std::vector<std::shared_ptr<Type>> ps;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(apply(p));
            auto rt = apply(f->ret_ty);
            return type_pool.function(std::move(ps), std::move(rt));
        }
        case TypeKind::LambdaFunction: {
            auto f = std::static_pointer_cast<LambdaFunctionType>(x);
            std::vector<std::shared_ptr<Type>> ps, cs;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(apply(p));
            cs.reserve(f->capture_tys.size());
            for (const auto& c : f->capture_tys) cs.push_back(apply(c));
            auto rt = apply(f->ret_ty);
            return type_pool.lambda_function(std::move(ps), std::move(rt), std::move(cs));
        }
        case TypeKind::NativeFunction: {
            auto f = std::static_pointer_cast<NativeFunctionType>(x);
            std::vector<std::shared_ptr<Type>> ps;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(apply(p));
            auto rt = apply(f->ret_ty);
            return type_pool.native_function(std::move(ps), std::move(rt), f->name);
        }
        case TypeKind::Array:
            return type_pool.array(apply(std::static_pointer_cast<ArrayType>(x)->type));
        case TypeKind::Tuple: {
            auto f = std::static_pointer_cast<TupleType>(x);
            std::vector<std::shared_ptr<Type>> es;
            es.reserve(f->tys.size());
            for (const auto& e : f->tys) es.push_back(apply(e));
            return type_pool.tuple(std::move(es));
        }
        case TypeKind::Nullable:
            return type_pool.nullable(
                apply(std::static_pointer_cast<NullableType>(x)->value_type));
        case TypeKind::Named: {
            auto f = std::static_pointer_cast<NamedType>(x);
            std::vector<std::shared_ptr<Type>> as;
            as.reserve(f->args.size());
            for (const auto& a : f->args) as.push_back(apply(a));
            return type_pool.named(f->name, std::move(as));
        }
        default: return x;
        }
    };

    for (auto& p : remaining_params) p = apply(p);
    for (auto& c : remaining_captures) c = apply(c);
    ret_ty = apply(ret_ty);
}

static void shallow_freshen_tvs(
    TypePool& type_pool,
    std::vector<std::shared_ptr<Type>>& params,
    std::vector<std::shared_ptr<Type>>& captures,
    std::shared_ptr<Type>& ret_ty
) noexcept {
    std::unordered_map<TypeVariable*, std::shared_ptr<TypeVariable>> rename;

    std::function<void(const std::shared_ptr<Type>&)> collect;
    collect = [&](const std::shared_ptr<Type>& t) {
        if (!t) return;
        if (t->kind == TypeKind::TypeVariable) {
            auto* tv = static_cast<TypeVariable*>(t.get());
            if (!rename.count(tv)) {
                rename.emplace(tv, type_pool.fresh_type_variable());
            }
            return;
        }
        switch (t->kind) {
        case TypeKind::Function: {
            auto f = std::static_pointer_cast<FunctionType>(t);
            for (const auto& p : f->params_ty) collect(p);
            collect(f->ret_ty);
            return;
        }
        case TypeKind::LambdaFunction: {
            auto f = std::static_pointer_cast<LambdaFunctionType>(t);
            for (const auto& p : f->params_ty) collect(p);
            for (const auto& c : f->capture_tys) collect(c);
            collect(f->ret_ty);
            return;
        }
        case TypeKind::NativeFunction: {
            auto f = std::static_pointer_cast<NativeFunctionType>(t);
            for (const auto& p : f->params_ty) collect(p);
            collect(f->ret_ty);
            return;
        }
        case TypeKind::Array:
            collect(std::static_pointer_cast<ArrayType>(t)->type); return;
        case TypeKind::Tuple: {
            auto f = std::static_pointer_cast<TupleType>(t);
            for (const auto& e : f->tys) collect(e);
            return;
        }
        case TypeKind::Nullable:
            collect(std::static_pointer_cast<NullableType>(t)->value_type); return;
        case TypeKind::Named: {
            auto f = std::static_pointer_cast<NamedType>(t);
            for (const auto& a : f->args) collect(a);
            return;
        }
        default: return;
        }
    };

    for (const auto& p : params) collect(p);
    for (const auto& c : captures) collect(c);
    collect(ret_ty);

    if (rename.empty()) return;

    std::function<std::shared_ptr<Type>(const std::shared_ptr<Type>&)> apply;
    apply = [&](const std::shared_ptr<Type>& x) -> std::shared_ptr<Type> {
        if (!x) return nullptr;
        if (x->kind == TypeKind::TypeVariable) {
            auto* tv = static_cast<TypeVariable*>(x.get());
            auto it = rename.find(tv);
            if (it != rename.end()) {
                return std::static_pointer_cast<Type>(it->second);
            }
            return x;
        }
        switch (x->kind) {
        case TypeKind::Function: {
            auto f = std::static_pointer_cast<FunctionType>(x);
            std::vector<std::shared_ptr<Type>> ps;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(apply(p));
            auto rt = apply(f->ret_ty);
            return type_pool.function(std::move(ps), std::move(rt));
        }
        case TypeKind::LambdaFunction: {
            auto f = std::static_pointer_cast<LambdaFunctionType>(x);
            std::vector<std::shared_ptr<Type>> ps, cs;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(apply(p));
            cs.reserve(f->capture_tys.size());
            for (const auto& c : f->capture_tys) cs.push_back(apply(c));
            auto rt = apply(f->ret_ty);
            return type_pool.lambda_function(std::move(ps), std::move(rt), std::move(cs));
        }
        case TypeKind::NativeFunction: {
            auto f = std::static_pointer_cast<NativeFunctionType>(x);
            std::vector<std::shared_ptr<Type>> ps;
            ps.reserve(f->params_ty.size());
            for (const auto& p : f->params_ty) ps.push_back(apply(p));
            auto rt = apply(f->ret_ty);
            return type_pool.native_function(std::move(ps), std::move(rt), f->name);
        }
        case TypeKind::Array:
            return type_pool.array(apply(std::static_pointer_cast<ArrayType>(x)->type));
        case TypeKind::Tuple: {
            auto f = std::static_pointer_cast<TupleType>(x);
            std::vector<std::shared_ptr<Type>> es;
            es.reserve(f->tys.size());
            for (const auto& e : f->tys) es.push_back(apply(e));
            return type_pool.tuple(std::move(es));
        }
        case TypeKind::Nullable:
            return type_pool.nullable(
                apply(std::static_pointer_cast<NullableType>(x)->value_type));
        case TypeKind::Named: {
            auto f = std::static_pointer_cast<NamedType>(x);
            std::vector<std::shared_ptr<Type>> as;
            as.reserve(f->args.size());
            for (const auto& a : f->args) as.push_back(apply(a));
            return type_pool.named(f->name, std::move(as));
        }
        default: return x;
        }
    };

    for (auto& p : params) p = apply(p);
    for (auto& c : captures) c = apply(c);
    ret_ty = apply(ret_ty);
}

void TypeCkContext::check_expr(std::shared_ptr<ExprNode>& expr) noexcept {
    if (!expr) return;
    switch (expr->kind) {
    case ASTKind::Literal: {
        expr->type = inference_type(expr.get());
        break;
    }
    case ASTKind::UnitAnnotated: {
        auto* node = static_cast<UnitAnnotatedExprNode*>(expr.get());
        check_expr(node->value);
        if (!is_basic_type(node->value->type, runtime::ValueKind::Int) &&
            !is_basic_type(node->value->type, runtime::ValueKind::Fraction)) {
            throw_error(ErrorType::Analysis,
                        "UnitTypeMismatch: units can only annotate numeric literals",
                        node->line, node->col);
            break;
        }
        const auto resolved = unit_system.resolve(node->unit_syntax);
        if (!resolved) {
            throw_error(ErrorType::Analysis,
                        "UnitInvalid: unknown or invalid unit expression `" +
                            node->unit_syntax.to_string() + "`",
                        node->line, node->col);
            break;
        }
        node->resolved_unit = *resolved;
        node->type = resolved->dimension.is_dimensionless()
            ? type_pool.basic(runtime::ValueKind::Fraction)
            : type_pool.dimensioned(*resolved);
        break;
    }
    case ASTKind::Identifier: {
        auto* node = reinterpret_cast<IdentifierNode*>(expr.get());
        if (node->id == "I") {
            node->type = type_pool.basic(runtime::ValueKind::Expr);
            break;
        }
        auto handle_adt_constructor = [&](const std::shared_ptr<Type>& ctor_type) -> bool {
            if (!ctor_type || ctor_type->kind != TypeKind::AdtConstructor) return false;
            const auto constructor = std::static_pointer_cast<AdtConstructorType>(ctor_type);
            if (!constructor->fields.empty()) return false;
            node->is_zero_adt_constructor = true;
            node->adt_type_name = constructor->type_name;
            std::vector<std::shared_ptr<Type>> args(constructor->type_params.size(), type_pool.unknown());
            node->type = type_pool.named(constructor->type_name, std::move(args));
            return true;
        };
        if (const auto re = find_var(node->id); re.has_value()) {
            if (handle_adt_constructor((*re)->type)) break;
            if ((*re)->scheme.has_value()) {
                node->type = instantiate_scheme(*(*re)->scheme);
            } else {
                node->type = (*re)->type;
            }
            break;
        }
        Scope::Var* resolved = nullptr;
        size_t regular_function_count = 0;
        for (auto& global : global_scope) {
            if (global.name != node->id) continue;
            if (!resolved) resolved = &global;
            if (global.type->kind == TypeKind::Function)
                ++regular_function_count;
        }
        if (regular_function_count > 1) {
            throw_error(ErrorType::Analysis, "ambiguous overloaded function",
                        node->line, node->col);
            break;
        }
        if (resolved) {
            if (resolved->type->kind == TypeKind::AdtConstructor) {
                const auto constructor =
                    std::static_pointer_cast<AdtConstructorType>(resolved->type);
                if (!constructor->fields.empty()) {
                    if (resolved->scheme.has_value()) {
                        node->type = instantiate_scheme(*resolved->scheme);
                    } else {
                        node->type = resolved->type;
                    }
                    break;
                }
                node->is_zero_adt_constructor = true;
                node->adt_type_name = constructor->type_name;
                std::vector<std::shared_ptr<Type>> args(
                    constructor->type_params.size(), type_pool.unknown());
                node->type = type_pool.named(constructor->type_name,
                                             std::move(args));
                break;
            }
            if (resolved->scheme.has_value()) {
                node->type = instantiate_scheme(*resolved->scheme);
            } else {
                node->type = resolved->type;
            }
            node->compiled_symbol = resolved->symbol;
            break;
        }
        throw_error(ErrorType::Analysis, "undefined var `" + node->id + "`", node->line, node->col);
        break;
    }
    case ASTKind::Unary: {
        auto* node = reinterpret_cast<UnaryNode*>(expr.get());
        check_expr(node->expr);
        const auto type = node->expr->type;
        if (Type::is_null_type(type.get()) ||
            type->kind == TypeKind::Unknown ||
            type->kind == TypeKind::None) {
            node->type = type_pool.unknown();
            break;
        }
        const auto resolved_operand = resolve_hm(type);
        if (resolved_operand->kind == TypeKind::TypeVariable) {
            node->type = type_pool.unknown();
            break;
        }
        if (type->kind == TypeKind::Dimensioned) {
            if (node->op != UnaryNode::Op::Neg) {
                throw_error(ErrorType::Analysis, "unary `not` requires bool",
                            expr->line, expr->col);
                break;
            }
            node->type = type;
            break;
        }
        if (type->kind != TypeKind::Basic) {
            throw_error(ErrorType::Analysis, "unary cannot applied to this type", expr->line, expr->col);
            break;
        }
        const auto t2 = std::reinterpret_pointer_cast<BasicType>(type);
        if (node->op == UnaryNode::Op::Neg) {
            if (
            t2->type != runtime::ValueKind::Int &&
            t2->type != runtime::ValueKind::Fraction &&
            t2->type != runtime::ValueKind::Real &&
            t2->type != runtime::ValueKind::Expr) {
                throw_error(ErrorType::Analysis, "unary`-` cannot applied to this type", expr->line, expr->col);
                break;
            }
        } else if (node->op == UnaryNode::Op::Not) {
            if (t2->type != runtime::ValueKind::Bool &&
                t2->type != runtime::ValueKind::Expr) {
                throw_error(ErrorType::Analysis, "unary`!` cannot applied to this type", expr->line, expr->col);
                break;
            }
        }
        node->type = type;
        break;
    }
    case ASTKind::Binary: {
        auto* node = reinterpret_cast<BinaryNode*>(expr.get());
        check_expr(node->lhs);
        check_expr(node->rhs);
        const auto lty = node->lhs->type;
        const auto rty = node->rhs->type;
        if (Type::is_null_type(lty.get()) || Type::is_null_type(rty.get())) break;
        const auto l_resolved = resolve_hm(lty);
        const auto r_resolved = resolve_hm(rty);
        const bool l_is_tv = l_resolved->kind == TypeKind::TypeVariable;
        const bool r_is_tv = r_resolved->kind == TypeKind::TypeVariable;
        const bool l_defer  = l_resolved->kind == TypeKind::Unknown || l_resolved->kind == TypeKind::None;
        const bool r_defer  = r_resolved->kind == TypeKind::Unknown || r_resolved->kind == TypeKind::None;
        if (l_defer || r_defer) {
            node->type = type_pool.unknown();
            break;
        }

        if (l_is_tv || r_is_tv) {
            (void)unify_hm(lty, rty);
            node->lhs->type = deep_resolve(lty);
            node->rhs->type = deep_resolve(rty);
            const auto l2 = resolve_hm(node->lhs->type);
            const auto r2 = resolve_hm(node->rhs->type);
            if (l2->kind == TypeKind::TypeVariable || r2->kind == TypeKind::TypeVariable) {
                node->type = type_pool.unknown();
                break;
            }
        }
        if (node->op == BinaryNode::Op::Bind) {
            if (is_expr_type(lty) || is_expr_type(rty)) {
                if (!is_expr_constructible(lty) || !is_expr_constructible(rty))
                    goto binary_type_mismatch;
                mark_expr_promotion(node->lhs);
                mark_expr_promotion(node->rhs);
                node->type = type_pool.named("Binding", {
                    type_pool.basic(runtime::ValueKind::Expr),
                    type_pool.basic(runtime::ValueKind::Expr)});
                break;
            }
            node->type = type_pool.named("Binding", {lty, rty});
            break;
        }
        if ((is_dimensioned_type(lty) || is_dimensioned_type(rty)) &&
            !is_expr_type(lty) && !is_expr_type(rty) &&
            node->op != BinaryNode::Op::In &&
            node->op != BinaryNode::Op::NotIn) {
            const auto left_dimensioned = is_dimensioned_type(lty);
            const auto right_dimensioned = is_dimensioned_type(rty);
            const auto plain_numeric = [](const std::shared_ptr<Type>& type) {
                return is_basic_type(type, runtime::ValueKind::Int) ||
                       is_basic_type(type, runtime::ValueKind::Fraction);
            };
            if (left_dimensioned && right_dimensioned) {
                const auto left = std::static_pointer_cast<DimensionedType>(lty);
                const auto right = std::static_pointer_cast<DimensionedType>(rty);
                switch (node->op) {
                case BinaryNode::Op::Add:
                case BinaryNode::Op::Sub:
                case BinaryNode::Op::Mod:
                    if (!left->equals(right.get())) {
                        throw_error(ErrorType::Analysis,
                                    "DimensionMismatch: addition, subtraction, and remainder require identical units",
                                    node->line, node->col);
                        break;
                    }
                    node->type = lty;
                    break;
                case BinaryNode::Op::Eq:
                case BinaryNode::Op::Ne:
                case BinaryNode::Op::Gt:
                case BinaryNode::Op::Ge:
                case BinaryNode::Op::Lt:
                case BinaryNode::Op::Le:
                    if (!left->equals(right.get())) {
                        throw_error(ErrorType::Analysis,
                                    "DimensionMismatch: comparison requires identical units",
                                    node->line, node->col);
                        break;
                    }
                    node->type = type_pool.basic(runtime::ValueKind::Bool);
                    break;
                case BinaryNode::Op::Mul:
                case BinaryNode::Op::Div: {
                    auto unit = combined_unit(left->unit, right->unit,
                                              node->op == BinaryNode::Op::Div);
                    if (!unit) {
                        throw_error(ErrorType::Analysis, "UnitScaleOverflow",
                                    node->line, node->col);
                        break;
                    }
                    node->type = unit->dimension.is_dimensionless()
                        ? type_pool.basic(runtime::ValueKind::Fraction)
                        : type_pool.dimensioned(std::move(*unit));
                    break;
                }
                default:
                    throw_error(ErrorType::Analysis,
                                "binary operation cannot be applied to dimensioned values",
                                node->line, node->col);
                    break;
                }
                break;
            }
            if (left_dimensioned && node->op == BinaryNode::Op::Pow) {
                const auto exponent_value = signed_integer_literal(node->rhs.get());
                if (!is_basic_type(rty, runtime::ValueKind::Int) || !exponent_value) {
                    throw_error(ErrorType::Analysis,
                                "DimensionExponentMustBeConstantInteger",
                                node->line, node->col);
                    break;
                }
                if (*exponent_value < -32 || *exponent_value > 32) {
                    throw_error(ErrorType::Analysis,
                                "DimensionExponentOutOfRange", node->line, node->col);
                    break;
                }
                const auto exponent = static_cast<int>(*exponent_value);
                const auto left = std::static_pointer_cast<DimensionedType>(lty);
                UnitDefinition unit;
                unit.dimension = left->unit.dimension.raised_to(exponent);
                const auto scale = left->unit.scale_to_base.raised_to(exponent);
                if (!scale) {
                    throw_error(ErrorType::Analysis, "UnitScaleOverflow",
                                node->line, node->col);
                    break;
                }
                unit.scale_to_base = *scale;
                unit.display_unit = left->unit.display_unit + "^" + std::to_string(exponent);
                node->type = unit.dimension.is_dimensionless()
                    ? type_pool.basic(runtime::ValueKind::Fraction)
                    : type_pool.dimensioned(std::move(unit));
                break;
            }
            if ((node->op == BinaryNode::Op::Mul || node->op == BinaryNode::Op::Div) &&
                ((left_dimensioned && plain_numeric(rty)) ||
                 (right_dimensioned && plain_numeric(lty)))) {
                if (left_dimensioned) {
                    node->type = lty;
                } else if (node->op == BinaryNode::Op::Mul) {
                    node->type = rty;
                } else {
                    const auto right = std::static_pointer_cast<DimensionedType>(rty);
                    UnitDefinition unit;
                    unit.dimension = right->unit.dimension.raised_to(-1);
                    const auto scale = right->unit.scale_to_base.raised_to(-1);
                    if (!scale) {
                        throw_error(ErrorType::Analysis, "UnitScaleOverflow",
                                    node->line, node->col);
                        break;
                    }
                    unit.scale_to_base = *scale;
                    unit.display_unit = "1/" + right->unit.display_unit;
                    node->type = type_pool.dimensioned(std::move(unit));
                }
                break;
            }
            throw_error(ErrorType::Analysis,
                        "DimensionMismatch: incompatible dimensioned operands",
                        node->line, node->col);
            break;
        }
        if (node->op == BinaryNode::Op::Eq || node->op == BinaryNode::Op::Ne) {
            if (auto unified = unify_types(lty, rty)) {
                if (!is_equality_comparable(unified)) {
                    throw_error(ErrorType::Analysis,
                                "values are not equality comparable",
                                node->line, node->col);
                    break;
                }
                if (contains_unknown_type(lty)) node->lhs->type = unified;
                if (contains_unknown_type(rty)) node->rhs->type = unified;
                node->type = type_pool.basic(runtime::ValueKind::Bool);
                break;
            }
        }
        if (node->op == BinaryNode::Op::In || node->op == BinaryNode::Op::NotIn) {
            if (is_named_type(rty, "set")) {
                const auto container = std::static_pointer_cast<NamedType>(rty);
                if (container->args.size() != 1)
                    goto binary_type_mismatch;
                const auto& element = container->args.front();
                if (is_expr_type(element) && is_expr_constructible(lty))
                    mark_expr_promotion(node->lhs);
                const bool assignable =
                    type_assignable(element, node->lhs->type) ||
                    (numeric_rank(element) >= 0 &&
                     numeric_rank(node->lhs->type) >= 0);
                if (!assignable) goto binary_type_mismatch;
                node->type = type_pool.basic(runtime::ValueKind::Bool);
                break;
            }
            if (is_expr_type(lty) || is_expr_type(rty)) {
                if (is_expr_constructible(lty))
                    mark_expr_promotion(node->lhs);
                if (node->rhs->kind == ASTKind::LiteralPayload ||
                    is_expr_constructible(rty))
                    mark_expr_promotion(node->rhs);
                node->type = type_pool.basic(runtime::ValueKind::Expr);
                break;
            }
            if (!is_named_type(rty, "interval"))
                goto binary_type_mismatch;
            const auto container = std::static_pointer_cast<NamedType>(rty);
            if (container->args.size() != 1 ||
                !interval_member_assignable(container->args.front(), lty))
                goto binary_type_mismatch;
            node->type = type_pool.basic(runtime::ValueKind::Bool);
            break;
        }
        {
        const bool explicit_set_operation =
            node->op == BinaryNode::Op::SetUnion ||
            node->op == BinaryNode::Op::SetIntersection ||
            node->op == BinaryNode::Op::SetSymmetricDifference ||
            node->op == BinaryNode::Op::Subset;
        const bool set_difference =
            node->op == BinaryNode::Op::Sub &&
            (is_named_type(lty, "set") || is_named_type(rty, "set"));
        if (explicit_set_operation || set_difference) {
            if (!is_named_type(lty, "set") ||
                !is_named_type(rty, "set")) {
                throw_error(ErrorType::Analysis, "SetOperandTypeMismatch",
                            node->line, node->col);
                break;
            }
            const auto left_set = std::static_pointer_cast<NamedType>(lty);
            const auto right_set = std::static_pointer_cast<NamedType>(rty);
            if (left_set->args.size() != 1 || right_set->args.size() != 1) {
                throw_error(ErrorType::Analysis, "SetOperandTypeMismatch",
                            node->line, node->col);
                break;
            }
            auto element = unify_types(left_set->args.front(),
                                       right_set->args.front());
            if (!element)
                element = unify_interval_bounds(left_set->args.front(),
                                                right_set->args.front());
            if (!element) {
                throw_error(ErrorType::Analysis, "SetElementTypeMismatch",
                            node->line, node->col);
                break;
            }
            if (element->kind != TypeKind::Unknown &&
                !is_equality_comparable(element)) {
                throw_error(ErrorType::Analysis, "SetElementNotHashable",
                            node->line, node->col);
                break;
            }
            const auto set_type = type_pool.named("set", {element});
            if (contains_unknown_type(lty)) node->lhs->type = set_type;
            if (contains_unknown_type(rty)) node->rhs->type = set_type;
            node->type = node->op == BinaryNode::Op::Subset
                ? type_pool.basic(runtime::ValueKind::Bool)
                : set_type;
            break;
        }
        }
        if (is_expr_type(lty) || is_expr_type(rty)) {
            switch (node->op) {
            case BinaryNode::Op::Add:
            case BinaryNode::Op::Sub:
            case BinaryNode::Op::Mul:
            case BinaryNode::Op::Div:
            case BinaryNode::Op::Pow:
            case BinaryNode::Op::Eq:
            case BinaryNode::Op::Ne:
            case BinaryNode::Op::Gt:
            case BinaryNode::Op::Ge:
            case BinaryNode::Op::Lt:
            case BinaryNode::Op::Le:
                if (is_numeric_or_expr_type(lty) && is_numeric_or_expr_type(rty)) {
                    node->type = type_pool.basic(runtime::ValueKind::Expr);
                    break;
                }
                goto binary_type_mismatch;
            case BinaryNode::Op::And:
            case BinaryNode::Op::Or:
                if (is_expr_type(lty) && is_expr_type(rty)) {
                    node->type = type_pool.basic(runtime::ValueKind::Expr);
                    break;
                }
                goto binary_type_mismatch;
            default:
                goto binary_type_mismatch;
            }
            if (is_expr_type(node->type)) {
                node->type = type_pool.basic(runtime::ValueKind::Expr);
                break;
            }
        }
        {
        const auto l_resolved = resolve_hm(lty);
        const auto r_resolved = resolve_hm(rty);
        if (!l_resolved || !r_resolved ||
            l_resolved->kind == TypeKind::TypeVariable ||
            r_resolved->kind == TypeKind::TypeVariable) {
            node->type = type_pool.unknown();
            break;
        }
        auto operand_type = l_resolved;
        if (!l_resolved->equals(r_resolved.get())) {
            if (numeric_rank(l_resolved) >= 0 && numeric_rank(r_resolved) >= 0) {
                operand_type = unify_interval_bounds(l_resolved, r_resolved);
            } else {
                throw_error(
                    ErrorType::Analysis,
                    "binary operation type mismatch, (" +
                    Type::to_string(l_resolved.get()) + " " +
                    BinaryNode::op_to_string(node->op) + " " +
                    Type::to_string(r_resolved.get()) + ")",
                    expr->line, expr->col);
                break;
            }
        }
        if (!operand_type || operand_type->kind != TypeKind::Basic)
            goto binary_type_mismatch;
        {
            const auto operand =
                std::reinterpret_pointer_cast<BasicType>(operand_type)->type;
            const auto result = basic_binary_result(operand, node->op);
            if (!result) goto binary_type_mismatch;
            node->type = type_pool.basic(*result);
        }
        }
        break;
        binary_type_mismatch:
            node->type = type_pool.unknown();
            throw_error(
                ErrorType::Analysis,
                "binary operation cannot applied to this type",
                expr->line,
                expr->col
            );
            break;
    }
    case ASTKind::LiteralPayload: {
        const auto node = reinterpret_cast<LiteralPayloadNode*>(expr.get());
        for (auto& element : node->elements) {
            check_expr(element);
        }
        node->type = literal_payload_type(*node);
        if (node->type->kind == TypeKind::Unknown) {
            const char* diagnostic = node->payload_kind == LiteralPayloadNode::Kind::Set
                ? "SetElementTypeMismatch"
                : "IntervalBoundTypeMismatch: interval bounds cannot be unified";
            throw_error(ErrorType::Analysis, diagnostic, node->line, node->col);
            break;
        }
        if (node->payload_kind == LiteralPayloadNode::Kind::Set) {
            const auto set = std::static_pointer_cast<NamedType>(node->type);
            if (set->args.size() == 1 &&
                is_expr_type(set->args.front())) {
                for (auto& element : node->elements) {
                    if (is_expr_constructible(element->type))
                        mark_expr_promotion(element);
                }
            }
            if (set->args.size() == 1 &&
                set->args.front()->kind != TypeKind::Unknown &&
                !is_equality_comparable(set->args.front())) {
                throw_error(ErrorType::Analysis, "SetElementNotHashable",
                            node->line, node->col);
                break;
            }
        }
        if (node->payload_kind == LiteralPayloadNode::Kind::Interval &&
            !is_expr_type(node->type)) {
            const auto interval = std::static_pointer_cast<NamedType>(node->type);
            if (interval->args.size() != 1 ||
                !is_interval_ordered_type(interval->args.front())) {
                throw_error(ErrorType::Analysis,
                            "IntervalBoundNotOrdered: interval bounds must be ordered values",
                            node->line, node->col);
                break;
            }
            const auto lower = constant_numeric_value(node->elements[0].get());
            const auto upper = constant_numeric_value(node->elements[1].get());
            if (lower && upper &&
                static_cast<long double>(lower->numerator) / lower->denominator >
                static_cast<long double>(upper->numerator) / upper->denominator) {
                throw_error(ErrorType::Analysis,
                            "IntervalBoundsReversed: lower bound exceeds upper bound",
                            node->line, node->col);
            }
        }
        break;
    }
    case ASTKind::Block: {
        const auto node = reinterpret_cast<BlockExprNode*>(expr.get());
        scope_stack.emplace_back(Scope::ScopeType::Block);
        for (auto& s : node->stmts) check_stmt(s);
        expr->type = scope_stack.back().return_type;
        scope_stack.pop_back();
        break;
    }
    case ASTKind::SuffixParen: {
        const auto node = reinterpret_cast<SuffixParenNode*>(expr.get());
        node->type = type_pool.unknown();
        if (const auto bounds = interval_constructor_bounds(node->expr.get())) {
            const auto standard_module = find_global("std");
            if (standard_module.has_value() &&
                (*standard_module)->type->kind == TypeKind::Module) {
                if (!node->suffix || node->suffix->exprs.size() != 2) {
                    throw_error(ErrorType::Analysis,
                                "IntervalArityMismatch: interval constructors require two bounds",
                                node->line, node->col);
                    break;
                }
                expr = std::make_shared<LiteralPayloadNode>(
                    node->line, node->col, LiteralPayloadNode::Kind::Interval,
                    std::move(node->suffix->exprs), bounds->first, bounds->second);
                check_expr(expr);
                break;
            }
        }
        if (node->expr->kind == ASTKind::Identifier) {
            const auto id = reinterpret_cast<IdentifierNode*>(node->expr.get());
              if (const auto found = find_global(id->id);
                found.has_value() && (*found)->type) {

                auto found_type = resolve_hm((*found)->type);

                if (found_type &&
                    found_type->kind == TypeKind::AdtConstructor) {
                const auto constructor = std::static_pointer_cast<AdtConstructorType>((*found)->type);
                if (constructor->fields.size() != node->suffix->exprs.size()) {
                 //   std::cout <<"field size:\n" << constructor->fields.size();
                 //   std::cout <<"suffix size:\n" << node->suffix->exprs.size();
                    throw_error(ErrorType::Analysis, "constructor `" + constructor->constructor + "` expects " +
                                std::to_string(constructor->fields.size()) + " field(s)", node->line, node->col);
                    break;
                }       
                const std::unordered_set<std::string> params(constructor->type_params.begin(), constructor->type_params.end());
                TypeBindings bindings;
                for (size_t i = 0; i < node->suffix->exprs.size(); ++i) {
                    check_expr(node->suffix->exprs[i]);
                    if (!bind_adt_type(constructor->fields[i], node->suffix->exprs[i]->type, params, bindings)) {
                        throw_error(ErrorType::Analysis, "constructor field type mismatch", node->line, node->col);
                    }
                }
                std::vector<std::shared_ptr<Type>> args;
                for (const auto& param : constructor->type_params) {
                    const auto it = bindings.find(param);
                    args.push_back(it == bindings.end() ? type_pool.unknown() : it->second);
                }
                node->is_adt_constructor = true;
                node->adt_type_name = constructor->type_name;
                node->adt_constructor = constructor->constructor;
                node->type = type_pool.named(constructor->type_name, std::move(args));
                break;
            }}
        }
        if (node->expr->kind == ASTKind::Identifier) {
            const auto id = reinterpret_cast<IdentifierNode*>(node->expr.get());
            if (!find_var(id->id).has_value() && !find_global(id->id).has_value()) {
                bool has_expr_arg = false;
                for (auto& arg : node->suffix->exprs) {
                    check_expr(arg);
                    has_expr_arg = has_expr_arg || is_expr_type(arg->type);
                }
                if (has_expr_arg || node->allow_symbolic_call) {
                    node->is_symbolic_call = true;
                    node->expr->type = type_pool.basic(runtime::ValueKind::Expr);
                    node->type = type_pool.basic(runtime::ValueKind::Expr);
                    break;
                }
            }
        }
        node->can_fast = false;
        bool selected_callable = false;
        const std::function<std::shared_ptr<Type>(ExprNode*)>
            candidate_type = [&](ExprNode* argument) -> std::shared_ptr<Type> {
            if (!argument) return type_pool.unknown();
            if (argument->type && argument->type->kind != TypeKind::Unknown)
                return argument->type;
            if (argument->kind == ASTKind::Identifier ||
                argument->kind == ASTKind::Literal ||
                argument->kind == ASTKind::Unary ||
                argument->kind == ASTKind::Binary)
                return inference_type(argument);
            if (argument->kind != ASTKind::ArrayLiteral)
                return type_pool.unknown();
            const auto* array = static_cast<ArrayLiteralNode*>(argument);
            if (array->exprs.empty()) return type_pool.array(type_pool.unknown());
            const auto element = candidate_type(array->exprs.front().get());
            for (std::size_t index = 1; index < array->exprs.size(); ++index) {
                if (!element->equals(candidate_type(array->exprs[index].get()).get()))
                    return type_pool.unknown();
            }
            return type_pool.array(element);
        };
        const auto parameters_match = [&](const auto& params) {
            if (params.size() != node->suffix->exprs.size()) return false;
            for (std::size_t index = 0; index < params.size(); ++index) {
                const auto argument = candidate_type(node->suffix->exprs[index].get());
                if (argument->kind != TypeKind::Unknown &&
                    !type_assignable(params[index], argument) &&
                    !(is_expr_type(params[index]) && is_expr_constructible(argument)))
                    return false;
            }
            return true;
        };
        if (node->expr->kind == ASTKind::Identifier) {
            auto* id = static_cast<IdentifierNode*>(node->expr.get());
            if (!find_var(id->id).has_value()) {
                std::vector<Scope::Var*> regular_candidates;
                for (auto& global : global_scope) {
                    if (global.name == id->id &&
                        global.type->kind == TypeKind::Function &&
                        !global.symbol.empty())
                        regular_candidates.push_back(&global);
                }
                if (!regular_candidates.empty()) {
                    const auto selected = std::find_if(
                        regular_candidates.begin(), regular_candidates.end(),
                        [&](const auto* candidate) {
                            return parameters_match(
                                std::static_pointer_cast<FunctionType>(
                                    candidate->type)->params_ty);
                        });
                    auto* chosen = selected == regular_candidates.end()
                        ? regular_candidates.front() : *selected;
                    id->type = chosen->type;
                    id->compiled_symbol = chosen->symbol;
                    node->can_fast = true;
                    selected_callable = true;
                }
            }
        } else if (node->expr->kind == ASTKind::DotExpr) {
            auto* dot = static_cast<DotExprNode*>(node->expr.get());
            const auto owner_type = inference_type(dot->expr.get());
            if (owner_type && owner_type->kind == TypeKind::Module) {
                dot->expr->type = owner_type;
                const auto module = std::static_pointer_cast<ModuleType>(owner_type);
                std::vector<Scope::Var*> regular_candidates;
                std::vector<std::shared_ptr<NativeFunctionType>> native_candidates;
                for (auto& exported : module->exports) {
                    if (exported.name != dot->rhs->id) continue;
                    if (exported.type->kind == TypeKind::Function)
                        regular_candidates.push_back(&exported);
                    else if (exported.type->kind == TypeKind::NativeFunction)
                        native_candidates.push_back(
                            std::static_pointer_cast<NativeFunctionType>(exported.type));
                }
                if (!regular_candidates.empty()) {
                    const auto selected = std::find_if(
                        regular_candidates.begin(), regular_candidates.end(),
                        [&](const auto* candidate) {
                            return parameters_match(
                                std::static_pointer_cast<FunctionType>(
                                    candidate->type)->params_ty);
                        });
                    auto* chosen = selected == regular_candidates.end()
                        ? regular_candidates.front() : *selected;
                    dot->rhs->type = chosen->type;
                    dot->type = chosen->type;
                    dot->compiled_symbol = chosen->symbol;
                    selected_callable = true;
                } else if (!native_candidates.empty()) {
                    const auto selected = std::find_if(
                        native_candidates.begin(), native_candidates.end(),
                        [&](const auto& candidate) {
                            return parameters_match(candidate->params_ty);
                        });
                    const auto& chosen = selected == native_candidates.end()
                        ? native_candidates.front() : *selected;
                    dot->rhs->type = chosen;
                    dot->type = chosen;
                    if (native_candidates.size() > 1)
                        node->adt_constructor = chosen->name;
                    selected_callable = true;
                }
            }
        }
        if (!selected_callable) check_expr(node->expr);
        const auto left = resolve_hm(node->expr->type);
        if (Type::is_null_type(left.get())) break;

        if (left->kind == TypeKind::TypeVariable) {
            const size_t arity = node->suffix->exprs.size();
            std::vector<std::shared_ptr<Type>> param_types;
            param_types.reserve(arity);
            for (size_t i = 0; i < arity; ++i) {
                check_expr(node->suffix->exprs[i]);
                if (!node->suffix->exprs[i]->type) {
                    node->suffix->exprs[i]->type = std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
                }
                param_types.push_back(std::static_pointer_cast<Type>(type_pool.fresh_type_variable()));
            }
            auto ret_var = type_pool.fresh_type_variable();
            auto expected_func = type_pool.lambda_function(
                std::move(param_types),
                std::static_pointer_cast<Type>(ret_var),
                {}
            );
            if (!unify_hm(left, expected_func)) {
                throw_error(ErrorType::Analysis,
                    "cannot unify type variable with function type at call site",
                    node->line, node->col);
                node->type = type_pool.unknown();
                break;
            }
            const auto lf = std::static_pointer_cast<LambdaFunctionType>(resolve_hm(expected_func));
            bool ok = true;
            for (size_t i = 0; i < arity && ok; ++i) {
                const auto ptype = resolve_hm(lf->params_ty[i]);
                const auto atype = resolve_hm(node->suffix->exprs[i]->type);

                if (!node->suffix->exprs[i]->type) {
                    node->suffix->exprs[i]->type = ptype;
                    continue;
                }
                if (!type_assignable(lf->params_ty[i], node->suffix->exprs[i]->type)) {
                    throw_error(ErrorType::Analysis,
                        "type mismatch arg in call (typevar instantiation) in arg(s) " + std::to_string(i) +
                        ": (" + Type::to_string(deep_resolve(node->suffix->exprs[i]->type).get()) +
                        " != " + Type::to_string(deep_resolve(ptype).get()) + ")"
                        , node->line, node->col);
                    ok = false;
                    break;
                }
            }
           if (!ok) {
                node->type = type_pool.unknown();
                break;
            }
            node->type = lf->ret_ty;
            break;
        }
        if (left->kind == TypeKind::AdtConstructor) {
            const auto constructor = std::static_pointer_cast<AdtConstructorType>(left);
            if (constructor->fields.size() != node->suffix->exprs.size()) {
                throw_error(ErrorType::Analysis, "constructor field count mismatch", node->line, node->col);
                break;
            }
            const std::unordered_set<std::string> params(constructor->type_params.begin(), constructor->type_params.end());
            TypeBindings bindings;
            for (size_t i = 0; i < node->suffix->exprs.size(); ++i) {
                check_expr(node->suffix->exprs[i]);
                if (!bind_adt_type(constructor->fields[i], node->suffix->exprs[i]->type, params, bindings))
                    throw_error(ErrorType::Analysis, "constructor field type mismatch", node->line, node->col);
            }
            std::vector<std::shared_ptr<Type>> args;
            for (const auto& param : constructor->type_params) {
                const auto it = bindings.find(param);
                args.push_back(it == bindings.end() ? type_pool.unknown() : it->second);
            }
            node->is_adt_constructor = true;
            node->adt_type_name = constructor->type_name;
            node->adt_constructor = constructor->constructor;
            node->type = type_pool.named(constructor->type_name, std::move(args));
        } else if (left->kind == TypeKind::Function) {
            const auto func_ty = std::reinterpret_pointer_cast<FunctionType>(left);
            const size_t given_args = node->suffix->exprs.size();
            const size_t need_args  = func_ty->params_ty.size();
            if (given_args > need_args) {
                throw_error(ErrorType::Analysis,
                    "mismatch args count in function calling, (param(s)"
                    + std::to_string(need_args) +
                    " != arg(s)" +
                    std::to_string(given_args) + ")",
                    node->line, node->col
                    );
                break;
            }
            const auto len = given_args;
            bool symbolic_fallback = false;
            for (auto i = 0; i < len; i++) {
                const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);
                if (is_expr_type(param) &&
                    is_expr_constructible(node->suffix->exprs[i]->type)) {
                    mark_expr_promotion(node->suffix->exprs[i]);
                }
                if (!node->suffix->exprs[i]->type) {
                    node->suffix->exprs[i]->type = std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
                }
               
                if (!type_assignable(param, node->suffix->exprs[i]->type)) {
                    if (node->expr->kind == ASTKind::Identifier &&
                        is_expr_type(node->suffix->exprs[i]->type)) {
                        node->is_symbolic_call = true;
                        node->type = type_pool.basic(runtime::ValueKind::Expr);
                        symbolic_fallback = true;
                        break;
                    }
                    throw_error(ErrorType::Analysis,
                        "type mismatch arg in function calling in arg(s) " + std::to_string(i) +
                        ": (" + Type::to_string(deep_resolve(node->suffix->exprs[i]->type).get()) +
                        " != " + Type::to_string(deep_resolve(param).get()) + ")"
                        , node->line, node->col);
                    break;
                }
            }
            if (symbolic_fallback) break;

            if (given_args < need_args) {
                std::vector<std::shared_ptr<Type>> remaining_params;
                std::vector<std::shared_ptr<Type>> remaining_captures;  // FunctionKind 无 captures
                remaining_params.reserve(need_args - given_args);
                for (size_t i = given_args; i < need_args; ++i) {
                    remaining_params.push_back(func_ty->params_ty[i]);
                }
                auto fresh_ret = func_ty->ret_ty;
                curry_freshen_intersection_tvs(type_pool, remaining_params, remaining_captures, fresh_ret);
                node->type = type_pool.lambda_function(
                    std::move(remaining_params),
                    std::move(fresh_ret),
                    {}
                );
            } else {
                auto fresh_ret = std::reinterpret_pointer_cast<FunctionType>(left)->ret_ty;
                std::vector<std::shared_ptr<Type>> empty_p, empty_c;
                shallow_freshen_tvs(type_pool, empty_p, empty_c, fresh_ret);
                node->type = fresh_ret;
            }
        } else if (left->kind == TypeKind::LambdaFunction) {
            const auto func_ty = std::reinterpret_pointer_cast<LambdaFunctionType>(left);
            const size_t given_args = node->suffix->exprs.size();
            const size_t need_args  = func_ty->params_ty.size();
            
            if (given_args > need_args) {
                throw_error(ErrorType::Analysis,
                    "mismatch args count in lambda calling, (param(s)"
                    + std::to_string(need_args) +
                    " != arg(s)" +
                    std::to_string(given_args) + ")",
                    node->line, node->col
                    );
                break;
            }
            const auto len = given_args;
            bool symbolic_fallback = false;
            for (auto i = 0; i < len; i++) {
                const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);

                if (is_expr_type(param) &&
                    is_expr_constructible(node->suffix->exprs[i]->type)) {
                    mark_expr_promotion(node->suffix->exprs[i]);
                }
                if (!node->suffix->exprs[i]->type) {
                    node->suffix->exprs[i]->type = std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
                }
                if (!type_assignable(param, node->suffix->exprs[i]->type)) {
                    if (node->expr->kind == ASTKind::Identifier &&
                        is_expr_type(node->suffix->exprs[i]->type)) {
                        node->is_symbolic_call = true;
                        node->type = type_pool.basic(runtime::ValueKind::Expr);
                        symbolic_fallback = true;
                        break;
                    }
                    throw_error(ErrorType::Analysis,
                        "type mismatch arg in lambda calling in arg(s) " + std::to_string(i) +
                        ": (" + Type::to_string(deep_resolve(node->suffix->exprs[i]->type).get()) +
                        " != " + Type::to_string(deep_resolve(param).get()) + ")"
                        , node->line, node->col);
                    break;
                }
            }
            if (symbolic_fallback) break;

            if (given_args < need_args) {
                std::vector<std::shared_ptr<Type>> remaining_params;
                remaining_params.reserve(need_args - given_args);
                for (size_t i = given_args; i < need_args; ++i) {
                    remaining_params.push_back(func_ty->params_ty[i]);
                }
                std::vector<std::shared_ptr<Type>> remaining_captures;
                remaining_captures.reserve(func_ty->capture_tys.size());
                for (const auto& c : func_ty->capture_tys) {
                    remaining_captures.push_back(c);
                }
                auto fresh_ret = func_ty->ret_ty;
                curry_freshen_intersection_tvs(type_pool, remaining_params, remaining_captures, fresh_ret);
                node->type = type_pool.lambda_function(
                    std::move(remaining_params),
                    std::move(fresh_ret),
                    std::move(remaining_captures)
                );
            } else {
                auto fresh_ret = std::reinterpret_pointer_cast<LambdaFunctionType>(left)->ret_ty;
                std::vector<std::shared_ptr<Type>> empty_p, empty_c;
                shallow_freshen_tvs(type_pool, empty_p, empty_c, fresh_ret);
                node->type = fresh_ret;
            }
        } else if (left->kind == TypeKind::NativeFunction) {
            const auto native_symbol = node->adt_constructor;
            new (expr.get()) NativeFuncCallExpr(node);
            const auto node = reinterpret_cast<NativeFuncCallExpr*>(expr.get());
            node->adt_constructor = native_symbol;
            const auto func_ty = std::reinterpret_pointer_cast<NativeFunctionType>(left);
            bool has_va_list = false;
            size_t fixed_arg_cnt = func_ty->params_ty.size();
            for (const auto& p : func_ty->params_ty) {
                if (p->kind == TypeKind::Basic &&
                    reinterpret_cast<BasicType*>(p.get())->type == runtime::ValueKind::C_VaList) {
                    if (p.get() != func_ty->params_ty.back().get()) {
                        throw_error(ErrorType::Analysis, "c_valist must be last type", node->line, node->col);
                        goto suffix_paren_break;
                    }
                    has_va_list = true;
                    fixed_arg_cnt = func_ty->params_ty.size() - 1;
                }
            }

            if (!has_va_list && func_ty->params_ty.size() != node->suffix->exprs.size()) {
                throw_error(ErrorType::Analysis,
                    "mismatch args count in function calling,(param(s): "
                    + std::to_string(func_ty->params_ty.size()) +
                    " != arg(s): " +
                    std::to_string(node->suffix->exprs.size()) + ")",
                    node->line, node->col
                    );
                break;
            }
            const auto len = node->suffix->exprs.size();
            size_t i = 0;
            for (; i < fixed_arg_cnt; i++) {
                const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);
                if (is_expr_type(param) &&
                    is_expr_constructible(node->suffix->exprs[i]->type)) {
                    mark_expr_promotion(node->suffix->exprs[i]);
                }
                if (!type_assignable(param, node->suffix->exprs[i]->type)) {
                    throw_error(
                        ErrorType::Analysis,
                        "type mismatch arg " + std::to_string(i) +
                            " in function calling: " +
                            Type::to_string(node->suffix->exprs[i]->type.get()) +
                            " is not assignable to " +
                            Type::to_string(param.get()),
                        node->line, node->col);
                    break;
                }
            }
            for (; i < len; i++) {
                check_expr(node->suffix->exprs[i]);
            }
          
            node->type = deep_resolve(std::reinterpret_pointer_cast<NativeFunctionType>(left)->ret_ty); 
        } else {
             throw_error(
                ErrorType::Analysis,
                "not a function type",
                node->line,
                node->col
            );

            node->type = type_pool.unknown();
            break;
        }

        break;
        suffix_paren_break:
        break;
    }
    case ASTKind::SuffixBracket: {
        const auto node = reinterpret_cast<SuffixBracketNode*>(expr.get());
        check_expr(node->expr);
        check_expr(node->suffix);
        const auto left = node->expr->type;
        if (left->kind != TypeKind::Array) {
            throw_error(ErrorType::Analysis, "must be array type but got `" + Type::to_string(left.get()) + "`", node->line, node->col);
            break;
        }
        if (node->suffix->type->kind != TypeKind::Basic ||
            std::reinterpret_pointer_cast<BasicType>(node->suffix->type)->type != runtime::ValueKind::Int) {
            throw_error(ErrorType::Analysis, "array index must be int", node->line, node->col);
            break;
        }
        node->type = std::reinterpret_pointer_cast<ArrayType>(left)->type;
        break;
    }
    case ASTKind::IfExpr: {
        const auto node = reinterpret_cast<IfExprNode*>(expr.get());
        check_expr(node->cond);
        const auto cond_ty = node->cond->type;
        const bool cond_unknown = Type::is_null_type(cond_ty.get()) ||
            cond_ty->kind == TypeKind::Unknown || cond_ty->kind == TypeKind::None;
        if (!cond_unknown && (node->cond->type->kind != TypeKind::Basic ||
            std::reinterpret_pointer_cast<BasicType>(node->cond->type)->type != runtime::ValueKind::Bool)) {
            throw_error(ErrorType::Analysis, "must be bool type but got `" + Type::to_string(node->cond->type.get()), node->line, node->col);
            break;
        }
        check_expr(node->then);
        if (node->els) {
            check_expr(node->els);
            const bool then_has = node->then->have_ret_value();
            const bool else_has = node->els->have_ret_value();
            auto then_ty = node->then->type;
            auto else_ty = node->els->type;
            const bool then_null = Type::is_null_type(then_ty.get()) || !then_ty;
            const bool else_null = Type::is_null_type(else_ty.get()) || !else_ty;
            if (then_has && else_has && !then_null && !else_null) {
                // 统一前先把两边 Unknown 替换为 TV，再做 unify
                auto then_tv = replace_unknowns_with_tvars(then_ty);
                auto else_tv = replace_unknowns_with_tvars(else_ty);
                auto unified = unify_types(then_tv, else_tv);
                if (!unified) {
                    throw_error(ErrorType::Analysis,
                        "if express then and else cannot type mismatch: then=(" +
                        Type::to_string(deep_resolve(then_tv).get()) + ") vs else=(" +
                        Type::to_string(deep_resolve(else_tv).get()) + ")",
                        node->line, node->col);
                    break;
                }
                node->type = deep_resolve(std::move(unified));
                // 统一后把 node->then/else 的类型同步更新，便于上层使用
                node->then->type = node->type;
                node->els->type  = node->type;
            }
        } else {
            node->type = node->then->type;
        }
        node->type = deep_resolve(replace_unknowns_with_tvars(node->then->type));
        if (cond_unknown && (!node->type || node->type->kind == TypeKind::Unknown)) {
            node->type = type_pool.unknown();
        }
        break;
    }
    case ASTKind::AsExpr: {
        auto* node = reinterpret_cast<AsExprNode*>(expr.get());
        if (node->cast_kind == AsExprNode::Kind::Unit) {
            check_expr(node->expr);
            const bool symbolic = is_expr_type(node->expr->type);
            if (!is_dimensioned_type(node->expr->type) && !symbolic) {
                throw_error(ErrorType::Analysis,
                            "UnitTypeMismatch: unit conversion requires a dimensioned value",
                            node->line, node->col);
                break;
            }
            const auto target = unit_system.resolve(node->unit_syntax);
            if (!target) {
                throw_error(ErrorType::Analysis,
                            "UnitInvalid: unknown or invalid target unit `" +
                                node->unit_syntax.to_string() + "`",
                            node->line, node->col);
                break;
            }
            const auto source = symbolic ? nullptr
                : std::static_pointer_cast<DimensionedType>(node->expr->type);
            if (source && source->unit.dimension != target->dimension) {
                throw_error(ErrorType::Analysis,
                            "DimensionMismatch: unit conversion requires equal dimensions",
                            node->line, node->col);
                break;
            }
            node->resolved_unit = *target;
            if (!symbolic) {
                const auto factor = source->unit.scale_to_base.divided_by(
                    target->scale_to_base);
                if (!factor || !runtime_scale_representable(*factor)) {
                    throw_error(ErrorType::Analysis, "UnitConversionOverflow",
                                node->line, node->col);
                    break;
                }
            }
            node->type = symbolic ? type_pool.basic(runtime::ValueKind::Expr)
                                  : type_pool.dimensioned(*target);
            break;
        }
        if (node->cast_kind == AsExprNode::Kind::Num ||
            node->cast_kind == AsExprNode::Kind::Scalar) {
            check_expr(node->expr);
            if (is_expr_type(node->expr->type)) {
                node->type = type_pool.basic(runtime::ValueKind::Expr);
                break;
            }
            if (is_dimensioned_type(node->expr->type)) {
                if (node->cast_kind == AsExprNode::Kind::Num) {
                    const auto dimensioned =
                        std::static_pointer_cast<DimensionedType>(node->expr->type);
                    if (!runtime_scale_representable(
                            dimensioned->unit.scale_to_base)) {
                        throw_error(ErrorType::Analysis, "UnitStripOverflow",
                                    node->line, node->col);
                        break;
                    }
                }
                node->type = type_pool.basic(runtime::ValueKind::Fraction);
                break;
            }
            if (is_basic_type(node->expr->type, runtime::ValueKind::Int) ||
                is_basic_type(node->expr->type, runtime::ValueKind::Fraction) ||
                is_basic_type(node->expr->type, runtime::ValueKind::Real)) {
                node->type = node->expr->type;
                break;
            }
            throw_error(ErrorType::Analysis, "UnitStripTypeMismatch",
                        node->line, node->col);
            break;
        }
        node->cast_type = resolve_type(node->cast_type);
        if (is_expr_type(node->cast_type) && node->expr->kind == ASTKind::SuffixParen)
            reinterpret_cast<SuffixParenNode*>(node->expr.get())->allow_symbolic_call = true;
        check_expr(node->expr);
        if (is_expr_type(node->cast_type) && is_expr_constructible(node->expr->type)) {
            mark_expr_promotion(node->expr);
        } else if (!node->cast_type->equals(node->expr->type.get())) {
            throw_error(ErrorType::Analysis, "cast type mismatch", node->line, node->col);
            break;
        }
        node->type = node->cast_type;
        break;
    }
    case ASTKind::DotExpr: {
        const auto node = reinterpret_cast<DotExprNode*>(expr.get());
        if (node->expr->kind == ASTKind::Identifier) {
            const auto* lhs = reinterpret_cast<IdentifierNode*>(node->expr.get());
            const auto type_it = adt_types.find(lhs->id);
            const auto constructor_it = adt_constructors.find(node->rhs->id);
            if (type_it != adt_types.end() && constructor_it != adt_constructors.end() &&
                constructor_it->second.first == type_it->second) {
                auto* declaration = type_it->second;
                auto* constructor = constructor_it->second.second;
                node->expr->type = type_pool.named(declaration->qualified_name);
                node->rhs->type = type_pool.adt_constructor(declaration->qualified_name, constructor->name,
                                                            declaration->type_params, constructor->fields);
                if (constructor->fields.empty()) {
                    node->is_zero_adt_constructor = true;
                    node->adt_type_name = declaration->qualified_name;
                    std::vector<std::shared_ptr<Type>> args(declaration->type_params.size(), type_pool.unknown());
                    node->type = type_pool.named(declaration->qualified_name, std::move(args));
                } else {
                    node->type = node->rhs->type;
                }
                break;
            }
        }
        check_expr(node->expr);
        if (!Type::is_null_type(node->expr->type.get()) && node->expr->type->kind != TypeKind::Module) {
            throw_error(ErrorType::Analysis, "must be module type", node->line, node->col);
            break;
        }
        const auto left_ty = std::reinterpret_pointer_cast<ModuleType>(node->expr->type);
        if (Type::is_null_type(left_ty.get())) break;
        if (const auto [declaration, constructor] = find_module_constructor(left_ty.get(), node->rhs->id);
            declaration && constructor) {
            node->rhs->type = type_pool.adt_constructor(
                declaration->qualified_name, constructor->name,
                declaration->type_params, constructor->fields);
            if (constructor->fields.empty()) {
                node->is_zero_adt_constructor = true;
                node->adt_type_name = declaration->qualified_name;
                std::vector<std::shared_ptr<Type>> args(
                    declaration->type_params.size(), type_pool.unknown());
                node->type = type_pool.named(declaration->qualified_name, std::move(args));
            } else {
                node->type = node->rhs->type;
            }
        } else {
            Scope::Var* resolved = nullptr;
            size_t regular_function_count = 0;
            for (auto& exported : left_ty->exports) {
                if (exported.name != node->rhs->id) continue;
                if (!resolved) resolved = &exported;
                if (exported.type->kind == TypeKind::Function)
                    ++regular_function_count;
            }
            if (regular_function_count > 1) {
                throw_error(ErrorType::Analysis, "ambiguous overloaded function",
                            node->line, node->col);
                break;
            }
            if (resolved) {
                node->rhs->type = resolved->type;
                node->type = resolved->type;
                node->compiled_symbol = resolved->symbol;
                break;
            }
            throw_error(ErrorType::Analysis, "module not have var `" +
                        node->rhs->id + "`", node->line, node->col);
            break;
        }
        break;
    }
    case ASTKind::MatchExpr: {
        auto* node = reinterpret_cast<MatchExprNode*>(expr.get());
        check_expr(node->target);
        bool catch_all = false;
        std::vector<Pattern> unguarded_patterns;
        std::shared_ptr<Type> result_type;
        auto target_named = node->target->type && node->target->type->kind == TypeKind::Named
            ? std::static_pointer_cast<NamedType>(node->target->type) : nullptr;

        for (auto& arm : node->arms) {
            if (catch_all) {
                throw_error(ErrorType::Analysis, "UnreachablePattern", arm.pattern.line, arm.pattern.col);
                continue;
            }
            scope_stack.emplace_back(Scope::ScopeType::Block);
            std::function<void(Pattern&, const std::shared_ptr<Type>&)> check_pattern;
            check_pattern = [&](Pattern& pattern, const std::shared_ptr<Type>& expected) {
                if (pattern.kind == Pattern::Kind::Wildcard) return;
                if (pattern.kind == Pattern::Kind::Binding) {
                    if (const auto constructor_it = adt_constructors.find(pattern.name);
                        constructor_it != adt_constructors.end() && constructor_it->second.second->fields.empty()) {
                        pattern.kind = Pattern::Kind::Constructor;
                        pattern.adt_type_name = constructor_it->second.first->qualified_name;
                        return;
                    }
                    new_cur_scope_var(pattern.name, expected);
                    return;
                }
                if (pattern.kind == Pattern::Kind::Literal) {
                    std::shared_ptr<ExprNode> literal = pattern.literal;
                    check_expr(literal);
                    if (!type_assignable(expected, literal->type)) {
                        throw_error(ErrorType::Analysis, "PatternTypeMismatch", pattern.line, pattern.col);
                    }
                    return;
                }
                TypeDeclNode* declaration = nullptr;
                AdtConstructorDecl* constructor = nullptr;
                if (!pattern.adt_type_name.empty()) {
                    if (const auto module_var = find_global(pattern.adt_type_name);
                        module_var.has_value() && (*module_var)->type->kind == TypeKind::Module) {
                        auto module = std::static_pointer_cast<ModuleType>((*module_var)->type);
                        std::tie(declaration, constructor) = find_module_constructor(module.get(), pattern.name);
                        if (declaration && constructor) pattern.adt_type_name = declaration->qualified_name;
                    }
                }
                if (!declaration || !constructor) {
                    const auto it = adt_constructors.find(pattern.name);
                    if (it != adt_constructors.end()) {
                        declaration = it->second.first;
                        constructor = it->second.second;
                    }
                }
                if (!declaration || !constructor) {
                    throw_error(ErrorType::Analysis, "unknown constructor `" + pattern.name + "`", pattern.line, pattern.col);
                    return;
                }
                if (!pattern.adt_type_name.empty() && pattern.adt_type_name != declaration->name &&
                    pattern.adt_type_name != declaration->qualified_name) {
                    throw_error(ErrorType::Analysis, "PatternTypeMismatch", pattern.line, pattern.col);
                    return;
                }
                pattern.adt_type_name = declaration->qualified_name;
                if (!expected || expected->kind != TypeKind::Named ||
                    std::static_pointer_cast<NamedType>(expected)->name != declaration->qualified_name) {
                    throw_error(ErrorType::Analysis, "PatternTypeMismatch", pattern.line, pattern.col);
                    return;
                }
                if (constructor->fields.size() != pattern.fields.size()) {
                    throw_error(ErrorType::Analysis, "constructor pattern field count mismatch", pattern.line, pattern.col);
                    return;
                }
                TypeBindings bindings;
                const auto expected_named = std::static_pointer_cast<NamedType>(expected);
                for (size_t i = 0; i < declaration->type_params.size() && i < expected_named->args.size(); ++i) {
                    bindings[declaration->type_params[i]] = expected_named->args[i];
                }
                for (size_t i = 0; i < pattern.fields.size(); ++i) {
                    check_pattern(pattern.fields[i], instantiate_adt_type(constructor->fields[i], bindings));
                }

            };
            check_pattern(arm.pattern, node->target->type);

            std::function<bool(const Pattern&, const Pattern&)> subsumes;
            subsumes = [&](const Pattern& previous, const Pattern& current) {
                if (previous.kind == Pattern::Kind::Wildcard || previous.kind == Pattern::Kind::Binding) return true;
                if (previous.kind != current.kind) return false;
                if (previous.kind == Pattern::Kind::Literal) {
                    return previous.literal && current.literal &&
                           previous.literal->kind == current.literal->kind &&
                           previous.literal->val == current.literal->val;
                }
                if (previous.kind != Pattern::Kind::Constructor ||
                    previous.adt_type_name != current.adt_type_name ||
                    previous.name != current.name ||
                    previous.fields.size() != current.fields.size()) return false;
                for (size_t i = 0; i < previous.fields.size(); ++i) {
                    if (!subsumes(previous.fields[i], current.fields[i])) return false;
                }
                return true;
            };
            if (std::any_of(unguarded_patterns.begin(), unguarded_patterns.end(),
                            [&](const Pattern& previous) { return subsumes(previous, arm.pattern); })) {
                throw_error(ErrorType::Analysis, "UnreachablePattern", arm.pattern.line, arm.pattern.col);
            }
            if (!arm.guard) unguarded_patterns.push_back(arm.pattern);
            if ((arm.pattern.kind == Pattern::Kind::Wildcard || arm.pattern.kind == Pattern::Kind::Binding) && !arm.guard) catch_all = true;
            if (arm.guard) {
                check_expr(arm.guard);
                if (!is_basic_type(arm.guard->type, runtime::ValueKind::Bool))
                    throw_error(ErrorType::Analysis, "match guard must be bool", arm.guard->line, arm.guard->col);
            }
            check_expr(arm.value);
            if (!result_type) {
                result_type = arm.value->type;
            } else if (auto unified = unify_types(result_type, arm.value->type)) {
                result_type = std::move(unified);
            } else {
                throw_error(ErrorType::Analysis, "MatchBranchTypeMismatch", arm.value->line, arm.value->col);
            }
            scope_stack.pop_back();
        }
        if (!catch_all) {
            using PatternRow = std::vector<Pattern>;
            using PatternMatrix = std::vector<PatternRow>;
            std::function<bool(const PatternMatrix&, const std::vector<std::shared_ptr<Type>>&)> exhaustive;
            exhaustive = [&](const PatternMatrix& matrix, const std::vector<std::shared_ptr<Type>>& types) -> bool {
                if (types.empty()) return !matrix.empty();
                const auto& head_type = types.front();
                auto tail_types = std::vector<std::shared_ptr<Type>>(types.begin() + 1, types.end());
                const auto make_wildcards = [](size_t count) {
                    PatternRow row;
                    row.reserve(count);
                    for (size_t i = 0; i < count; ++i)
                        row.emplace_back(Pattern::Kind::Wildcard, 0, 0);
                    return row;
                };
                const auto specialize_default = [&] {
                    PatternMatrix specialized;
                    for (const auto& row : matrix) {
                        if (row.empty()) continue;
                        if (row.front().kind != Pattern::Kind::Wildcard && row.front().kind != Pattern::Kind::Binding) continue;
                        specialized.emplace_back(row.begin() + 1, row.end());
                    }
                    return specialized;
                };

                auto default_matrix = specialize_default();
                if (!default_matrix.empty() && exhaustive(default_matrix, tail_types)) return true;

                if (head_type && head_type->kind == TypeKind::Named) {
                    const auto named = std::static_pointer_cast<NamedType>(head_type);
                    if (const auto declaration_it = adt_types.find(named->name); declaration_it != adt_types.end()) {
                        auto* declaration = declaration_it->second;
                        TypeBindings bindings;
                        for (size_t i = 0; i < declaration->type_params.size() && i < named->args.size(); ++i)
                            bindings[declaration->type_params[i]] = named->args[i];
                        for (const auto& constructor : declaration->constructors) {
                            PatternMatrix specialized;
                            for (const auto& row : matrix) {
                                if (row.empty()) continue;
                                PatternRow next;
                                const auto& head = row.front();
                                if (head.kind == Pattern::Kind::Wildcard || head.kind == Pattern::Kind::Binding) {
                                    next = make_wildcards(constructor.fields.size());
                                } else if (head.kind == Pattern::Kind::Constructor && head.name == constructor.name) {
                                    next = head.fields;
                                } else {
                                    continue;
                                }
                                next.insert(next.end(), row.begin() + 1, row.end());
                                specialized.push_back(std::move(next));
                            }
                            std::vector<std::shared_ptr<Type>> specialized_types;
                            specialized_types.reserve(constructor.fields.size() + tail_types.size());
                            for (const auto& field : constructor.fields)
                                specialized_types.push_back(instantiate_adt_type(field, bindings));
                            specialized_types.insert(specialized_types.end(), tail_types.begin(), tail_types.end());
                            if (!exhaustive(specialized, specialized_types)) return false;
                        }
                        return true;
                    }
                }

                if (is_basic_type(head_type, runtime::ValueKind::Bool)) {
                    for (const auto value : {"true", "false"}) {
                        PatternMatrix specialized;
                        for (const auto& row : matrix) {
                            if (row.empty()) continue;
                            const auto& head = row.front();
                            if (head.kind == Pattern::Kind::Wildcard || head.kind == Pattern::Kind::Binding ||
                                (head.kind == Pattern::Kind::Literal && head.literal && head.literal->val == value)) {
                                specialized.emplace_back(row.begin() + 1, row.end());
                            }
                        }
                        if (!exhaustive(specialized, tail_types)) return false;
                    }
                    return true;
                }

                return false;
            };

            PatternMatrix matrix;
            for (const auto& arm : node->arms) {
                if (!arm.guard) matrix.push_back({arm.pattern});
            }
            if (!exhaustive(matrix, {node->target->type}))
                throw_error(ErrorType::Analysis, "MissingWildcard", node->line, node->col);
        }
        node->type = result_type ? result_type : type_pool.none();
        break;
    }
    case ASTKind::ArrayLiteral: {
        auto* node = reinterpret_cast<ArrayLiteralNode*>(expr.get());
        std::shared_ptr<Type> element_type;
        for (auto& element : node->exprs) {
            check_expr(element);
            const auto& candidate = element->type;
            if (Type::is_null_type(candidate.get())) continue;
            if (!element_type) {
                element_type = candidate;
            } else if (!element_type->equals(candidate.get())) {
                throw_error(ErrorType::Analysis,
                    "array literal elements must be the same type, (" +
                    Type::to_string(element_type.get()) + " != " + Type::to_string(candidate.get()) + ")",
                    node->line, node->col);
                break;
            }
        }
        node->type = type_pool.array(element_type ? element_type : type_pool.unknown());
        break;
    }
    case ASTKind::PipeExpr: {
        const auto node = reinterpret_cast<PipeExprNode*>(expr.get());
        check_expr(node->lhs);
        check_expr(node->rhs);
        std::shared_ptr<ExprNode> result;
        std::shared_ptr<Type> ret_ty;
            
        const auto lhs_ty = resolve_hm(node->lhs->type);
        const auto rhs_ty = resolve_hm(node->rhs->type);
        if (!node->lhs->type || !node->rhs->type) {
            break;
        }
        if (rhs_ty->kind == TypeKind::LambdaFunction) {
            const auto rhs_fty = std::reinterpret_pointer_cast<LambdaFunctionType>(rhs_ty);
            if (rhs_fty->params_ty.empty()) {
                throw_error(
                    ErrorType::Analysis,
                    "`|>` right lambda requires at least one parameter",
                    node->line,
                    node->col
                );
                break;
            }
            if (!type_assignable(rhs_fty->params_ty.front(), lhs_ty)) {
                throw_error(
                    ErrorType::Analysis,
                    "`|>` lambda first argument type mismatch, ("
                    + Type::to_string(lhs_ty.get())
                    + " |> "
                    + Type::to_string(rhs_fty->params_ty[0].get())
                    + ")",
                    node->line,
                    node->col
                );
                break;
            }

            ret_ty = rhs_fty->ret_ty;
        } else if (rhs_ty->kind == TypeKind::Function) {
            const auto rhs_fty = std::reinterpret_pointer_cast<FunctionType>(rhs_ty);
            if (rhs_fty->params_ty.empty()) {
                throw_error(
                    ErrorType::Analysis,
                    "`|>` op right function calling not arg(1)",
                    node->line,
                    node->col
                );
                break;
            }
            if (!type_assignable(rhs_fty->params_ty.front(), lhs_ty)) {
                throw_error(
                    ErrorType::Analysis,
                    "`|>` op in right, function arg type and left type mismatch, ("
                    + Type::to_string(lhs_ty.get())
                    + " |> "
                    + Type::to_string(rhs_fty->params_ty[0].get())
                    + ")",
                    node->line,
                    node->col
                );
                break;
            }
            ret_ty = rhs_fty->ret_ty;
        } else {
            throw_error(
                ErrorType::Analysis,
                "`|>` op not return func on right",
                node->line,
                node->col
            );
            break;
        }
        decltype(ExprsNode::exprs) exprs;
        exprs.push_back(node->lhs);
        result = std::make_shared<SuffixParenNode>(
            node->line,
            node->col,
            node->rhs,
            std::make_shared<ExprsNode>(
                node->line,
                node->col,
                std::move(exprs)
            )
        );

        result->type = deep_resolve(ret_ty);
        expr = result;
        break;
    }
    case ASTKind::TupleLiteral: {
        const auto node = reinterpret_cast<TupleLiteralNode*>(expr.get());
        decltype(TupleType::tys) tys;
        for (auto& e : node->exprs) {
            check_expr(e);
            tys.push_back(e->type);
        }
        node->type = type_pool.tuple(std::move(tys));
        break;
    }
    case ASTKind::TupleGetExpr: {
        auto* node = reinterpret_cast<TupleGetExprNode*>(expr.get());
        check_expr(node->tup);
        if (node->tup->type->kind != TypeKind::Tuple) {
            throw_error(ErrorType::Analysis,
                        "TupleTypeMismatch: position access requires a tuple",
                        node->line, node->col);
            break;
        }
        const auto* tup_ty = reinterpret_cast<TupleType*>(node->tup->type.get());
        if (node->i >= tup_ty->tys.size()) {
            throw_error(ErrorType::Analysis,
                "TupleIndexOutOfBounds: tuple has " +
                    std::to_string(tup_ty->tys.size()) +
                    " elements but position " + std::to_string(node->i + 1) +
                    " was requested",
                node->line, node->col
                );
            break;
        }
        node->type = tup_ty->tys[node->i];
        break;
    }
    case ASTKind::LambdaExpr: {
        const auto node = reinterpret_cast<LambdaExprNode*>(expr.get());
        if (!node->params || !node->body) {
            throw_error(ErrorType::Analysis, "invalid lambda expression", node->line, node->col);
            break;
        }
        Scope scope(Scope::ScopeType::Function);
        scope.name = "@lambda";
        scope.is_lambda = true;  // 标记为 lambda 作用域
        node->captured_vars.clear();
        std::unordered_set<std::string> referenced_names;
        collect_referenced_identifiers(node->body.get(), referenced_names);
        std::unordered_set<std::string> param_names;
        if (node->params) {
            for (const auto& [pname, _] : node->params->stmts) {
                param_names.insert(pname);
            }
        }
        auto try_capture = [&](const std::string& name, const std::shared_ptr<Type>& type) {
            if (referenced_names.count(name) == 0) return;
            if (param_names.count(name) > 0) return;
            if (scope.readonly_captured_vars.count(name) > 0) return;
            scope.readonly_captured_vars.insert(name);
            if (type && type->kind != TypeKind::NativeFunction)
            {
                node->captured_vars.push_back(LambdaExprNode::CapturedVar{name, type});
            }
        };
        for (const auto& s : scope_stack | std::views::reverse) {
            for (const auto& v : s.vars) {
                try_capture(v.name, v.type);
            }
        }
        for (const auto& gv : global_scope) {
            try_capture(gv.name, gv.type);
        }
        {
            std::string pnames;
            for (const auto& p : param_names) { pnames += p + " "; }
            std::string cvnames;
            for (const auto& cv : node->captured_vars) { cvnames += cv.name + " "; }
        }

        for (auto& [name, ty] : node->params->stmts) {
            if (!ty || Type::is_null_type(ty.get()) || ty->kind == TypeKind::Unknown) {
                ty = std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
            }
            scope.vars.emplace_back(name, ty, true);
        }
        scope_stack.push_back(scope);

        const bool had_err_before_body = errd;
        errd = false;
        check_expr(node->body);
        const bool body_err = errd;
        errd = had_err_before_body || body_err;
        if (body_err) {
            scope_stack.pop_back();
            node->type = type_pool.unknown();
            break;
        }
        if (node->body->kind == ASTKind::Block) {
            auto *blk = reinterpret_cast<BlockExprNode*>(node->body.get());
            const bool has_tail_return =
                !blk->stmts.empty() && blk->stmts.back() &&
                blk->stmts.back()->kind == ASTKind::TailReturn;
            const bool has_any_return = has_explicit_return(node->body.get());

            if (!has_tail_return && !has_any_return &&
                (Type::is_null_type(scope.return_type.get()) ||
                 scope.return_type->kind == TypeKind::Unknown ||
                 scope.return_type->kind == TypeKind::TypeVariable ||
                 scope.return_type->kind == TypeKind::None)) {
                auto dummy = make_lambda_dummy_return(node->line, node->col);
                blk->stmts.push_back(std::make_shared<ReturnNode>(
                    node->line, node->col, std::move(dummy)));
                scope_stack.back().return_type = type_pool.basic(runtime::ValueKind::Int);
            }
        }
        scope_stack.pop_back();

        Scope scope2(Scope::ScopeType::Function);
        scope2.name = "@lambda";
        scope2.is_lambda = true;
        scope2.readonly_captured_vars = scope.readonly_captured_vars;
        for (const auto& [name, ty] : node->params->stmts) {
            scope2.vars.emplace_back(name, ty, true);
        }
        for (const auto& cv : node->captured_vars) {
            if (cv.type && !Type::is_null_type(cv.type.get())) {
                scope2.vars.emplace_back(cv.name, cv.type, false);
            }
        }
        for (const auto& s : scope_stack | std::views::reverse) {
            for (const auto& v : s.vars) {
                bool dup = false;
                for (const auto& pv : scope2.vars) {
                    if (pv.name == v.name) { dup = true; break; }
                }
                if (!dup) scope2.vars.emplace_back(v.name, v.type, v.is_mut);
            }
        }
        for (const auto& gv : global_scope) {
            bool dup = false;
            for (const auto& pv : scope2.vars) {
                if (pv.name == gv.name) { dup = true; break; }
            }
            if (!dup) scope2.vars.emplace_back(gv.name, gv.type, false);
        }
        scope_stack.push_back(scope2);
        const bool had_err_before_body2 = errd;
        errd = false;
        check_expr(node->body);
        const bool body_err2 = errd;
        errd = had_err_before_body2 || body_err2;
        if (body_err2) {
            scope_stack.pop_back();
            node->type = type_pool.unknown();
            break;
        }

        auto inferred_return = inference_type(node->body.get());
        const auto& scope_ret = scope_stack.back().return_type;

        const auto is_unresolved = [](const std::shared_ptr<Type>& t) {
            if (Type::is_null_type(t.get())) return true;
            if (t->kind == TypeKind::Unknown) return true;
            if (t->kind == TypeKind::TypeVariable) {
                auto tv = std::static_pointer_cast<TypeVariable>(t);
                return !tv->binding;
            }
            return t->kind == TypeKind::None;
        };

        const auto inferred_has_value = !is_unresolved(inferred_return);
        const auto scope_ret_has_value = !is_unresolved(scope_ret);

        if (inferred_has_value && scope_ret_has_value) {
            if (!unify_hm(scope_ret, inferred_return)) {
                throw_error(ErrorType::Analysis,
                    "lambda return type mismatch: body infers (" +
                    Type::to_string(deep_resolve(inferred_return).get()) +
                    ") but return statement gives (" +
                    Type::to_string(deep_resolve(scope_ret).get()) + ")",
                    node->line, node->col);
                scope_stack.pop_back();
                break;
            }
        }
        auto return_type = inferred_return;
        if (!inferred_has_value && scope_ret_has_value) {
            return_type = scope_ret;
        }

        scope_stack.pop_back();

        decltype(LambdaFunctionType::params_ty) params_ty;
        for (auto &[name, ty] : node->params->stmts) {
            if (!ty || Type::is_null_type(ty.get()) || ty->kind == TypeKind::Unknown || ty->kind == TypeKind::None) {
                ty = std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
            } else {
                ty = replace_unknowns_with_tvars(ty);
            }
            params_ty.push_back(ty);
        }
        decltype(LambdaFunctionType::capture_tys) capture_tys;
        capture_tys.reserve(node->captured_vars.size());
        for (auto& cv : node->captured_vars) {
            if (!cv.type || Type::is_null_type(cv.type.get()) || cv.type->kind == TypeKind::Unknown || cv.type->kind == TypeKind::None) {
                cv.type = std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
            } else {
                cv.type = replace_unknowns_with_tvars(cv.type);
            }
            capture_tys.push_back(cv.type);
        }
        return_type = replace_unknowns_with_tvars(return_type);
        node->type = type_pool.lambda_function(std::move(params_ty), return_type, std::move(capture_tys));
        break;
    }
    default: std::unreachable();
    }
}

void TypeCkContext::check_stmt(std::shared_ptr<StmtNode>& stmt) noexcept {
    switch (stmt->kind) {
    case ASTKind::TypeDecl:
        break;
    case ASTKind::ExprStmt: {
        auto* node = reinterpret_cast<ExprStmtNode*>(stmt.get());
        check_expr(node->expr);
        if (node->expr && contains_adt_unknown_args(node->expr->type))
            throw_error(ErrorType::Analysis, "cannot infer ADT type arguments", node->line, node->col);
        break;
    }
    case ASTKind::ImportStmt: {
        const auto* node = reinterpret_cast<ImportStmtNode*>(stmt.get());
        if (!module_resolver) {
            throw_error(ErrorType::Analysis, "module resolver is unavailable for `" + node->name + "`", node->line, node->col);
            break;
        }
        const auto resolved = module_resolver->resolve_module({
            node->name,
            cur_module->name,
            node->line,
            node->col,
        });
        if (!resolved || errd) break;
        if (cur_module->module_is_imported(resolved->source_path)) break;

        for (const auto& declaration : resolved->type->adt_exports)
            adt_types[declaration->qualified_name] = declaration.get();
        for (const auto& [name, definition] : resolved->type->unit_exports)
            unit_system.import_unit(resolved->binding_name + "." + name, definition);
        new_global_var(resolved->binding_name, resolved->type);
        cur_module->imports[resolved->source_path] = resolved->type;
        break;
    }
    case ASTKind::UnitDecl: {
        auto* node = static_cast<UnitDeclNode*>(stmt.get());
        if (!is_global_scope()) {
            throw_error(ErrorType::Analysis, "unit declarations must be module scoped",
                        node->line, node->col);
            break;
        }
        if (!node->definition) {
            if (!unit_system.declare_base(
                    node->name, cur_module->name + "::" + node->name)) {
                throw_error(ErrorType::Analysis, "UnitRedefined: `" + node->name + "`",
                            node->line, node->col);
                break;
            }
            node->resolved_unit = *unit_system.resolve(node->name);
        } else {
            check_expr(node->definition);
            if (!is_dimensioned_type(node->definition->type)) {
                throw_error(ErrorType::Analysis,
                            "UnitInvalid: derived unit requires a dimensioned constant",
                            node->line, node->col);
                break;
            }
            const auto scale = constant_numeric_value(node->definition.get());
            if (!scale || scale->numerator <= 0) {
                throw_error(ErrorType::Analysis,
                            "UnitInvalid: derived unit scale must be a positive compile-time constant",
                            node->line, node->col);
                break;
            }
            const auto dimensioned =
                std::static_pointer_cast<DimensionedType>(node->definition->type);
            UnitDefinition definition{
                dimensioned->unit.dimension, *scale, node->name};
            if (!unit_system.declare_derived(node->name, definition)) {
                throw_error(ErrorType::Analysis, "UnitRedefined: `" + node->name + "`",
                            node->line, node->col);
                break;
            }
            node->resolved_unit = *unit_system.resolve(node->name);
        }
        cur_module->unit_exports.emplace_back(node->name, node->resolved_unit);
        break;
    }
    case ASTKind::SymDecl: {
        const auto* node = reinterpret_cast<SymDeclNode*>(stmt.get());
        for (const auto& id : node->ids) {
            if (id == "I") {
                throw_error(ErrorType::Analysis, "ImaginaryUnitReserved", node->line, node->col);
                break;
            }
            if (is_global_scope()) {
                new_global_var(id, type_pool.basic(runtime::ValueKind::Expr));
            } else {
                new_cur_scope_var(id, type_pool.basic(runtime::ValueKind::Expr));
            }
        }
        break;
    }
    case ASTKind::FuncImpl: {
        auto* node = reinterpret_cast<FuncImplNode*>(stmt.get());
        if (!is_global_scope()) throw_error(ErrorType::Analysis, "function only define in GlobalScope", stmt->line, stmt->col);

        for (auto& [name, type] : node->params->stmts) type = resolve_type(type);
        node->return_type = resolve_type(node->return_type);

        if (Type::is_null_type(node->return_type.get())) {
            node->return_type = std::static_pointer_cast<Type>(type_pool.fresh_type_variable());
        }

        new_global_var(node->func_id, node->make_type(), false,
                       node->compiled_symbol);
        auto& ref = global_scope.back();
        Scope scope;
        scope.name = node->func_id;
        scope.return_type = node->return_type;
        for (const auto& [name, type] : node->params->stmts) {
            scope.vars.emplace_back(name, type, true);
        }
        scope_stack.push_back(scope);
        std::shared_ptr<Type> implicit_return;
        if (node->block->kind == ASTKind::Block) {
            auto* block = reinterpret_cast<BlockExprNode*>(node->block.get());
            for (auto& s : block->stmts) {
                check_stmt(s);
            }

            if (!block->stmts.empty()) {
                auto& last = block->stmts.back();
                if (last->kind == ASTKind::ExprStmt) {
                    auto* es = reinterpret_cast<ExprStmtNode*>(last.get());
                    if (es->expr) implicit_return = es->expr->type;
                }
            }
            if (block->type && !Type::is_null_type(block->type.get())) {
                implicit_return = block->type;
            }
        } else {
            check_expr(node->block);
            implicit_return = node->block->type;
        }

        if (implicit_return && !Type::is_null_type(implicit_return.get())) {
            implicit_return = replace_unknowns_with_tvars(implicit_return);
            auto& cur_ret = scope_stack.back().return_type;
            if (Type::is_null_type(cur_ret.get())) {
                cur_ret = implicit_return;
            } else {
                (void)unify_hm(cur_ret, implicit_return);
            }
            cur_ret = resolve_hm(cur_ret);
        }

        if (!node->return_type->equals(scope_stack.back().return_type.get())) {
            node->return_type = scope_stack.back().return_type;
        }
        if (Type::is_null_type(node->return_type.get()) && implicit_return) {
            node->return_type = resolve_hm(implicit_return);
        }
        node->return_type = resolve_hm(node->return_type);

        scope_stack.pop_back();

        std::shared_ptr<Type> final_func_type = std::static_pointer_cast<Type>(node->make_type());
        final_func_type = replace_unknowns_with_tvars(final_func_type);
        final_func_type = resolve_hm(final_func_type);
        ref.type = final_func_type;

        std::unordered_set<TypeVariable*> env_free, visited_env;
        auto collect_scope_vars = [&](const std::vector<Scope::Var>& vars) {
            for (const auto& v : vars) {
                if (v.scheme.has_value()) continue;
                collect_free_type_vars(v.type, env_free, visited_env);
            }
        };
        for (size_t i = 0; i < global_scope.size(); ++i) {
            if (&global_scope[i] == &ref) continue;
            if (global_scope[i].scheme.has_value()) continue;
            collect_free_type_vars(global_scope[i].type, env_free, visited_env);
        }

        std::unordered_set<TypeVariable*> mono_free, visited_mono;
        collect_free_type_vars(final_func_type, mono_free, visited_mono);
        std::vector<TypeVariable*> quantified;
        for (auto* tv : mono_free) {
            if (env_free.count(tv) == 0) {
                quantified.push_back(tv);
            }
        }
        std::optional<TypeScheme> scheme = std::nullopt;
        if (!quantified.empty()) {
            auto frozen = freeze_scheme_monotype(final_func_type, quantified);
            scheme = TypeScheme{std::move(frozen.second), std::move(frozen.first)};
        }
        
        ref.scheme = std::move(scheme);
        break;
    }
    case ASTKind::Return: {
        const auto node = reinterpret_cast<ReturnNode*>(stmt.get());
        if (!node->expr) break;
        check_expr(node->expr);
        for (auto& s : scope_stack | std::views::reverse) {
            if (s.scope == Scope::ScopeType::Function) {
                if (Type::is_null_type(s.return_type.get())) {
                    s.return_type = node->expr->type;
                    break;
                }
                if (contains_unknown_type(node->expr->type) && type_assignable(s.return_type, node->expr->type))
                    node->expr->type = s.return_type;
                if (is_expr_type(s.return_type) && is_expr_constructible(node->expr->type))
                    mark_expr_promotion(node->expr);
                if (!type_assignable(s.return_type, node->expr->type)) {
                    if (!unify_hm(s.return_type, node->expr->type)) {
                        throw_error(ErrorType::Analysis, "return type mismatch in function `" + s.name + "`", node->line, node->col);
                        goto return_fail_break;
                    }
                }
                s.return_type = resolve_hm(s.return_type);
                break;
            }
        }
        return_fail_break:
        break;
    }
    case ASTKind::TailReturn: {
        const auto node = reinterpret_cast<TailReturnNode*>(stmt.get());
        if (is_expr_type(scope_stack.back().return_type) &&
            node->expr && node->expr->kind == ASTKind::SuffixParen)
            reinterpret_cast<SuffixParenNode*>(node->expr.get())->allow_symbolic_call = true;
        check_expr(node->expr);
        if (Type::is_null_type(scope_stack.back().return_type.get()))
            scope_stack.back().return_type = node->expr->type;
        else {
            if (contains_unknown_type(node->expr->type) &&
                type_assignable(scope_stack.back().return_type, node->expr->type))
                node->expr->type = scope_stack.back().return_type;
            if (is_expr_type(scope_stack.back().return_type) &&
                is_expr_constructible(node->expr->type))
                mark_expr_promotion(node->expr);
            if (!type_assignable(scope_stack.back().return_type, node->expr->type)) {
                if (!unify_hm(scope_stack.back().return_type, node->expr->type)) {
                    throw_error(ErrorType::Analysis, "return type is inconsistent with the above", node->line, node->col);
                    break;
                }
            }
            scope_stack.back().return_type = resolve_hm(scope_stack.back().return_type);
        }
        break;
    }
    case ASTKind::VarDecl: {
        const auto node = reinterpret_cast<VarDeclNode*>(stmt.get());
        if (node->id == "I") {
            throw_error(ErrorType::Analysis, "ImaginaryUnitReserved", node->line, node->col);
            break;
        }
        if (!Type::is_null_type(node->type.get())) node->type = resolve_type(node->type);
        if (is_expr_type(node->type) && node->init_value &&
            node->init_value->kind == ASTKind::SuffixParen)
            reinterpret_cast<SuffixParenNode*>(node->init_value.get())->allow_symbolic_call = true;
       // std::cout << node->init_value << std::endl;
        check_expr(node->init_value);
        if (node->init_value && !node->init_value->type) {
            node->init_value->type = type_pool.unknown();
        }
        //std::cout << node->init_value << std::endl;
        if (Type::is_null_type(node->type.get())) {
            if (!node->init_value) {
                throw_error(ErrorType::Analysis, "the var `" + node->id + "` type not found", node->line, node->col);
                break;
            } else {
                if (contains_unknown_type(node->init_value->type)) {
                    throw_error(ErrorType::Analysis, "cannot infer ADT type arguments for `" + node->id + "`", node->line, node->col);
                    break;
                }
                auto tval = replace_unknowns_with_tvars(
                    node->init_value->type
                );
                if (!tval) tval = type_pool.unknown();
                if (node->type &&
                    !Type::is_null_type(node->type.get()) &&
                    node->type->kind != TypeKind::Unknown) {

                    auto declared_ty = replace_unknowns_with_tvars(node->type);

                    if (!unify_hm(declared_ty, tval)) {
                        throw_error(
                            ErrorType::Analysis,
                            "the var `" + node->id +
                            "` type mismatch with the initialization type",
                            node->line,
                            node->col
                        );
                        break;
                    }

                    tval = deep_resolve(declared_ty);
                }

                node->init_value->type = tval;
                node->type = tval;
                if (contains_adt_unknown_args(node->type)) {
                    throw_error(ErrorType::Analysis, "cannot infer ADT type arguments for `" + node->id + "`", node->line, node->col);
                    break;
                }
            }
        } else {
            node->init_value->type = replace_unknowns_with_tvars(node->init_value->type);
            if (!node->init_value->type) {
                node->init_value->type = type_pool.unknown();
            }
            if (is_expr_type(node->type) && is_expr_constructible(node->init_value->type)) {
                mark_expr_promotion(node->init_value);
            } else if (contains_unknown_type(node->init_value->type) &&
                       type_assignable(node->type, node->init_value->type)) {
                node->init_value->type = node->type;
            } else if (!type_assignable(node->type, node->init_value->type) &&
                       !unify_hm(node->type, node->init_value->type)) {
                throw_error(ErrorType::Analysis, "the var `" + node->id + "` type mismatch with the initialization type", node->line, node->col);
                break;
            }
        }
       
        node->type = resolve_hm(node->type);
        if (!node->type) node->type = type_pool.unknown();
        std::unordered_set<TypeVariable*> env_free, visited_env;
        auto collect_scope_vars = [&](const std::vector<Scope::Var>& vars) {
            for (const auto& v : vars) {
                if (v.scheme.has_value()) continue;
                collect_free_type_vars(v.type, env_free, visited_env);
            }
        };
        collect_scope_vars(global_scope);
        for (const auto& s : scope_stack) collect_scope_vars(s.vars);
        std::unordered_set<TypeVariable*> mono_free, visited_mono;
        collect_free_type_vars(node->type, mono_free, visited_mono);
        std::vector<TypeVariable*> quantified;
        for (auto* tv : mono_free) {
            if (env_free.count(tv) == 0) {
                quantified.push_back(tv);
            }
        }
        std::optional<TypeScheme> scheme = std::nullopt;
        if (!quantified.empty()) {
            auto frozen = freeze_scheme_monotype(node->type, quantified);
            scheme = TypeScheme{std::move(frozen.second), std::move(frozen.first)};
        }
        if (is_global_scope()) {
            static const std::unordered_set<std::string_view> kWatch = {
                "zero","succ","add","mul","pair","first","second","t","pred","sub","to_int"
            };
          
        }
        if (is_global_scope()) {
            new_global_var_with_scheme(node->id, node->type, scheme, node->is_mutable);
        } else {
            new_cur_scope_var_with_scheme(node->id, node->type, scheme, node->is_mutable);
        }
        break;
    }
    case ASTKind::AssignStmt: {
        const auto node = reinterpret_cast<AssignStmtNode*>(stmt.get());
        check_expr(node->lhs);
        check_expr(node->rhs);
        if (node->lhs->kind == ASTKind::SuffixBracket) {
        } else if (node->lhs->kind == ASTKind::TupleGetExpr) {
            throw_error(ErrorType::Analysis,
                        "TupleAssignment: tuple element bindings are immutable",
                        node->line, node->col);
            break;
        } else if (node->lhs->kind == ASTKind::Identifier) {
            const auto id = reinterpret_cast<IdentifierNode*>(node->lhs.get());
            auto var = find_var(id->id);
            if (!var.has_value()) var = find_global(id->id);
            if (!var.has_value()) {
                throw_error(ErrorType::Analysis, "undefined var `" + id->id + "`", node->line, node->col);
                break;
            }
            if (!(*var)->is_mut) {
                throw_error(ErrorType::Analysis, "cannot assign to immutable var `" + id->id + "`", node->line, node->col);
                break;
            }
        } else {
            throw_error(ErrorType::Analysis, "left side of assignment must be an identifier", node->line, node->col);
            break;
        }
        if (!unify_hm(node->lhs->type, node->rhs->type)) {
            throw_error(ErrorType::Analysis, "assignment type mismatch", node->line, node->col);
        }
        node->lhs->type = deep_resolve(node->lhs->type);
        node->rhs->type = deep_resolve(node->rhs->type);
        break;
    }
    case ASTKind::BreakStmt:
    case ASTKind::ContinueStmt:{
        bool in_loop = false;
        for (const auto& s : scope_stack | std::views::reverse) {
            if (s.scope == Scope::ScopeType::Loop) {
                in_loop = true;
                break;
            }
        }
        if (!in_loop) {
            throw_error(ErrorType::Analysis, "break stmt must be in loop body", stmt->line, stmt->col);
            break;
        }
        break;
    }
    case ASTKind::LoopStmt: {
        auto node = std::reinterpret_pointer_cast<LoopStmtNode>(stmt);
        if (node->expr) {
            check_expr(node->expr);
            if (!Type::is_null_type(node->expr->type.get()) &&
                !node->expr->type->equals(type_pool.basic(runtime::ValueKind::Int).get())
                ) {
                    throw_error(ErrorType::Analysis, "loop condition type must be int", node->line, node->col);
                    break;
                }
        }
        scope_stack.emplace_back(Scope::ScopeType::Loop);
        for (auto& s : node->body) {
            check_stmt(s);
        }
        scope_stack.pop_back();
        if (node->expr) {
            stmt = sugar_loop_count(node);
        }
        break;
    }
    default: std::unreachable();
    }
}

bool TypeCkContext::is_global_scope() const noexcept {
    return scope_stack.size() == 1;
}

void TypeCkContext::new_var(std::string name, std::shared_ptr<Type> type, Scope *scope, bool is_mut) noexcept {
    scope->vars.emplace_back(std::move(name), std::move(type), is_mut);
}

void TypeCkContext::new_cur_scope_var(std::string name, std::shared_ptr<Type> type, bool is_mut) noexcept {
    scope_stack.back().vars.emplace_back(std::move(name), std::move(type), is_mut);
}

void TypeCkContext::new_global_var(std::string name, std::shared_ptr<Type> type,
                                   bool is_mut, std::string symbol,
                                   bool is_export) noexcept {
    global_scope.emplace_back(std::move(name), std::move(type), is_mut,
                              std::move(symbol), is_export);
}

void TypeCkContext::new_cur_scope_var_with_scheme(std::string name, std::shared_ptr<Type> type,
    std::optional<TypeScheme> scheme, bool is_mut) noexcept {
    Scope::Var var{std::move(name), std::move(type), is_mut, {}, true, std::move(scheme)};
    scope_stack.back().vars.push_back(std::move(var));
}

void TypeCkContext::new_global_var_with_scheme(std::string name, std::shared_ptr<Type> type,
    std::optional<TypeScheme> scheme, bool is_mut) noexcept {
    Scope::Var var{std::move(name), std::move(type), is_mut, {}, true, std::move(scheme)};
    global_scope.push_back(std::move(var));
}

std::vector<Scope::Var> &TypeCkContext::get_global() noexcept {
    return global_scope;
}
