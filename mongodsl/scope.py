from dataclasses import dataclass

from mongodsl.ast import BinOp, Call, Sym, Var, Const

STAGES = ("set", "group", "addFields")

OPERATORS = ("floor", "add", "multiply", "subtract", "divide")

# Meta-Syntax Assumption:
# pipestage = $stagename :{"string field name":expression}
# expression = varname | expression (expression) |expression <op> expression
# op = + | * | - | /


def analyse_var(e):
    if isinstance(e, BinOp):
        return BinOp(e.op_name, analyse_var(e.operand_a), analyse_var(e.operand_b))
    elif isinstance(e, Call):
        return Call(e.fn_name, [analyse_var(x) for x in e.args], e.listp)
    elif isinstance(e, Var):
        return e
    elif isinstance(e, Sym):
        return Var(e.name)
    elif isinstance(e, Const):
        return e
