from dataclasses import dataclass

from mongodsl.ast import (BinCmp, BinOp, Call, Const, ExprWrapper, FieldBinCmp,
                          PyVar, Raw, Sym, Var)

STAGES = ("set", "group", "addFields")

OPERATORS = ("floor", "add", "multiply", "subtract", "divide")

# Meta-Syntax Assumption:
# pipestage = $stagename :{"string field name":expression}
# expression = varname | expression (expression) |expression <op> expression
# op = + | * | - | /


def analyse_var(e):
    if isinstance(e, BinOp):
        return BinOp(e.op_name, analyse_var(e.operand_a), analyse_var(e.operand_b))
    elif isinstance(e, PyVar):
        return e
    elif isinstance(e, Call):
        return Call(e.fn_name, [analyse_var(x) for x in e.args], e.listp)
    elif isinstance(e, Var):
        return e
    elif isinstance(e, Sym):
        return Var(e.name)
    elif isinstance(e, Raw):
        return e
    elif isinstance(e, Const):
        return e
    elif isinstance(e, ExprWrapper):
        return ExprWrapper(analyse_var(e.expr))
    elif isinstance(e, BinCmp):
        a = analyse_var(e.operand_a)
        b = analyse_var(e.operand_b)
        # if isinstance(a, Var):
        #     bv = b
        #     if isinstance(b, Const):
        #         bv = Raw(b.val)
        #     elif isinstance(b, Var):
        #         bv = Raw(b.name)
        #     return FieldBinCmp(e.op_name, a.name, bv)
        # if isinstance(b, Var):
        #     av = a
        #     if isinstance(a, Const):
        #         av = Raw(a.val)
        #     elif isinstance(a, Var):
        #         av = Raw(a.name)
        #     return FieldBinCmp(flip_cmp(e.op_name), b.name, av)
        return BinCmp(e.op_name, a, b)
    else:
        raise Exception(f"unknown ast node: {e}")


def flip_cmp(op):
    return {
        "gte": "lte",
        "gt": "lt",
        "lte": "gte",
        "lt": "gt",
        "eq": "eq",
        "ne": "ne",
    }[op]
