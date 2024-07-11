#include "GPUThreadReduction.h"

#include "Associativity.h"
#include "CodeGen_GPU_Dev.h"
#include "IR.h"
#include "IRMutator.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

namespace {

bool has_thread_reduction_inner(const For *op) {
    struct DetectInnerReduction : public IRVisitor {
        using IRVisitor::visit;
        bool has_inner_reduction = false;

        void visit(const For *op) override {
            if (op->for_type == ForType::GPUThreadReduction) {
                has_inner_reduction = true;
            }
            return IRVisitor::visit(op);
        }
    };

    DetectInnerReduction dir({});
    dir.visit(op);

    return dir.has_inner_reduction;
}

bool has_thread_rvar_inner_args(const std::string &rvar, const Call *op) {
    struct DetectInnerReductionVar : public IRVisitor {
        using IRVisitor::visit;
        std::string thread_rvar;
        bool has_rvar = false;

        DetectInnerReductionVar(std::string rvar)
            : thread_rvar(std::move(rvar)) {
        }

        // only visit call args
        void visit(const Call *op) override {
            for (const auto &arg : op->args) {
                arg->accept(this);
            }
        }

        void visit(const Variable *op) override {
            if (op->name == thread_rvar) {
                has_rvar = true;
            }
            return IRVisitor::visit(op);
        }
    };

    DetectInnerReductionVar detector(rvar);
    detector.visit(op);
    return detector.has_rvar;
}

// f() = 100
// TODO(xylonx): try to handle case `f() = f() + input(r) + r + 2;` for InitialSubstituter and ReductionSubstituter

struct InitialSubstituter : public IRMutator {
    using IRMutator::visit;

    const std::string &reduction_func_name;
    const Expr &identity_ele;

    InitialSubstituter(const Expr &identity, const std::string &f_name)
        : reduction_func_name(f_name), identity_ele(identity) {
    }

    Expr visit(const Call *op) override {
        if (op->name == reduction_func_name) {
            return identity_ele;
        }
        return IRMutator::visit(op);
    }
};

struct ReductionSubstituter : public IRMutator {
    using IRMutator::visit;

    const std::string &reduction_func_name, &tid;
    const Expr &identity_ele, &reduction_pos, &reduction_neighbor;

    ReductionSubstituter(const std::string &rfn, const std::string &tid, const Expr &i, const Expr &rp, const Expr &rn)
        : reduction_func_name(rfn), tid(tid), identity_ele(i), reduction_pos(rp), reduction_neighbor(rn) {
    }

    Expr visit(const IntImm *) override {
        return identity_ele;
    }
    Expr visit(const UIntImm *) override {
        return identity_ele;
    }
    // Expr visit(const )
    Expr visit(const FloatImm *) override {
        return identity_ele;
    }
    Expr visit(const StringImm *) override {
        return identity_ele;
    }

    Expr visit(const Call *op) override {
        if (op->name == reduction_func_name) {
            return reduction_pos;
        }
        if (has_thread_rvar_inner_args(tid, op)) {
            return reduction_neighbor;
        }
        return identity_ele;
    }
};

class GPUThreadReduction : public IRMutator {
private:
    using IRMutator::visit;

    bool inner_reduction = false;

    Expr block_var, block_min, block_extend;
    Expr thread_var, thread_extend;
    std::string thread_var_name;

    std::string reduce_provider_name, intermediate_buffer_name;
    std::vector<Expr> reduce_provider_args;

    Partition thread_reduction_partition;
    DeviceAPI device_api;

    const Stmt gpu_sync_call =
        Evaluate::make(Call::make(Int(32), Call::gpu_thread_barrier,
                                  {IntImm::make(Int(32), CodeGen_GPU_Dev::MemoryFenceType::Shared)}, Call::Intrinsic));

    Stmt visit(const For *op) override {
        if (op->for_type != ForType::GPUThreadReduction) {
            if (has_thread_reduction_inner(op)) {
                if (op->for_type == ForType::GPUBlock) {
                    block_var = Variable::make(Int(32), op->name);
                    block_min = op->min;
                    block_extend = op->extent;

                    // allocate intermediate buffer
                    Stmt stmt = mutate(op->body);

                    stmt = Allocate::make(intermediate_buffer_name, Int(32), MemoryType::GPUShared, {thread_extend},
                                          const_true(), stmt);

                    // NOTE(xylonx): hack way to specify min, stride and extend for single dimension.
                    // Do we need multi-dimensional reduction?
                    stmt = LetStmt::make(intermediate_buffer_name + ".stride.0", 1, stmt);
                    stmt = LetStmt::make(intermediate_buffer_name + ".extend.0", op->extent, stmt);
                    stmt = LetStmt::make(intermediate_buffer_name + ".min.0", 0, stmt);

                    return For::make(op->name, op->min, op->extent, op->for_type, op->partition_policy, op->device_api,
                                     stmt);
                }
            }

            return IRMutator::visit(op);
        }

        inner_reduction = true;

        thread_var_name = op->name;
        thread_var = Variable::make(Int(32), op->name);
        thread_extend = op->extent;

        thread_reduction_partition = op->partition_policy;
        device_api = op->device_api;

        Stmt stmt = mutate(op->body);

        // // NOTE(xylonx): hack way to determine whether it is a single block reduction or not. maybe it is better to add another ForType
        // const IntImm *block_extend_imm = block_extend.as<IntImm>();
        // std::vector<Expr> return_args;
        // if (block_extend_imm == nullptr || block_extend_imm->value > 1) {
        //     return_args.push_back(block_var);
        // }

        stmt = Block::make({
            stmt,
            IfThenElse::make(
                thread_var == 0,
                // TODO(xylonx): now just gives the result value to the original function. try to combine it with existed value.
                // like f() = 100; f() += input(r);
                // For final result, it should add 100 to the final result.
                Provide::make(reduce_provider_name, {Call::make(Int(32), intermediate_buffer_name, {}, Call::Halide)},
                              reduce_provider_args, const_true())),
        });

        inner_reduction = false;

        return For::make(op->name, op->min, op->extent, ForType::GPUThread, op->partition_policy, op->device_api, stmt);
    }

    Stmt visit(const Provide *op) override {
        if (!inner_reduction) {
            return IRMutator::visit(op);
        }

        reduce_provider_name = op->name;
        intermediate_buffer_name = reduce_provider_name + "_gpu_shard_intm";
        reduce_provider_args = op->args;

        std::string log_step_var_name = op->name + "_log_step";
        Expr log_step_var = Variable::make(Int(32), log_step_var_name);

        AssociativeOp associative_op = prove_associativity(op->name, op->args, op->values);
        user_assert(associative_op.associative()) << "function inner gpu thread reduction must be associative";

        std::map<std::string, Expr> replacements;
        const auto replace_op = [&replacements](AssociativeOp::Replacement r) { replacements.emplace(r.var, r.expr); };
        std::for_each(associative_op.xs.begin(), associative_op.xs.end(), replace_op);
        std::for_each(associative_op.ys.begin(), associative_op.ys.end(), replace_op);

        std::vector<Expr> f_values;
        for (const auto &op : associative_op.pattern.ops) {
            f_values.emplace_back(substitute(replacements, op));
        }

        // use two ir mutator
        std::vector<Expr> f_initial_values, f_reduction_values;
        for (int i = 0; i < f_values.size(); i++) {
            f_initial_values.emplace_back(
                InitialSubstituter(associative_op.pattern.identities[i], reduce_provider_name).mutate(f_values[i]));

            const Expr rp = Call::make(Int(32), intermediate_buffer_name, {thread_var}, Call::Halide);
            const Expr rn =
                Call::make(Int(32), intermediate_buffer_name, {thread_var + (1 << log_step_var)}, Call::Halide);

            f_reduction_values.emplace_back(ReductionSubstituter(reduce_provider_name, thread_var_name,
                                                                 associative_op.pattern.identities[i], rp, rn)
                                                .mutate(f_values[i]));
        }

        const Stmt initial_stmt = Provide::make(intermediate_buffer_name, f_initial_values, {thread_var}, const_true());

        // HACK(xylonx): maybe add another extern function to get the logged thread extend
        const IntImm *t_extend_imm = thread_extend.as<IntImm>();
        user_assert(t_extend_imm != nullptr)
            << "(HACK): now we assume thread reduction only support integer imm extend";
        const Expr log_step_extend = IntImm::make(Int(32), (long long)log2(t_extend_imm->value) + 1);

        // tree reduction
        Stmt reduction_stmt = Provide::make(
            intermediate_buffer_name,
            f_reduction_values, {thread_var}, const_true());

        reduction_stmt = IfThenElse::make(
            ((thread_var % (2 * (1 << log_step_var))) == 0) && ((thread_var + (1 << log_step_var)) < thread_extend),
            reduction_stmt);

        reduction_stmt = Block::make({
            reduction_stmt,
            gpu_sync_call,
        });

        reduction_stmt = For::make(log_step_var_name, 0, log_step_extend, ForType::Serial,
                                   thread_reduction_partition, device_api, reduction_stmt);

        return Block::make({
            initial_stmt,
            gpu_sync_call,
            reduction_stmt,
        });
    }
};

}  // namespace

Stmt gpu_thread_reduction(Stmt s) {
    s = GPUThreadReduction().mutate(s);
    return s;
}

}  // namespace Internal
}  // namespace Halide