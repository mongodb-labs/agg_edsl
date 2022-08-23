from dataclasses import dataclass
from typing import Any, List

import bytecode as bc
from fpy.composable.collections import and_, apply, const, mp1, or_, trans0
from fpy.data.either import Right
from fpy.experimental.do import do
from fpy.parsec.parsec import many, one, ptrans
from fpy.utils.placeholder import __

from mongodsl.ast import BinOp, Call, Sym, Var
from mongodsl.scope import analyse_var

"""
from bson.son import SON

# before
pipeline = [
    {"$unwind": "$tags"},
    {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
    {"$sort": SON([("count", -1), ("_id": -1)])}
]

db.col.aggregate(pipeline)

# after
from mongodsl import aggregate

@aggregate
def pipeline():
    unwind(tags)
    group({"_id": tags, "count": {sum: 1}})
    # _. is an escape for evaluation of Python expr
    sort(_.SON([("count", -1), ("_id": -1)])})

pipeline.apply(db.col)
"""


@dataclass
class Ret:
    lineno: int


@dataclass
class Esc:
    name: str


@dataclass
class SetField:
    name: List[str]
    expr: List[List[bc.Instr]]


@dataclass
class Stage:
    name: str
    expr: Any

    def to_json(self):
        return {f"{self.name}": self.expr()}


@dataclass
class Pipeline:
    stages: List[Stage]

    def apply(self, col):
        return col.aggregate(self.to_json())

    def to_json(self):
        return [s.to_json() for s in self.stages if s is not None]


instr = lambda x: isinstance(x, bc.Instr)
loadg = and_(instr, __.name == "LOAD_GLOBAL")
loadf = and_(instr, __.name == "LOAD_FAST")
loadv = or_(loadg, loadf)
load_ = and_(loadg, __.arg == "_")
loadm = and_(instr, __.name == "LOAD_METHOD")
escape = one(load_) >> one(loadm)

fncall = and_(instr, __.name == "CALL_FUNCTION")
popTop = and_(instr, __.name == "POP_TOP")
storeFast = and_(instr, __.name == "STORE_FAST")
none = and_(instr, and_(__.name == "LOAD_CONST", __.arg == None))
ret = and_(instr, __.name == "RETURN_VALUE")

binAdd = and_(instr, __.name == "BINARY_ADD")
binSub = and_(instr, __.name == "BINARY_SUBTRACT")
binMul = and_(instr, __.name == "BINARY_MULTIPLY")
binDiv = and_(instr, __.name == "BINARY_TRUE_DIVIDE")
binOp = or_(binAdd, or_(binSub, or_(binMul, binDiv)))

OPMAP = {
    "BINARY_ADD": "add",
    "BINARY_SUBTRACT": "subtract",
    "BINARY_MULTIPLY": "multiply",
    "BINARY_TRUE_DIVIDE": "divide",
}

parseNoneRet = many(
    ptrans(
        one(popTop) << one(none) << one(ret),
        trans0(trans0(__.lineno ^ Ret)),
    )
    | one(const(True))
)


def partitionInst(insts, n):
    if not insts:
        return [], []
    if n == 0:
        return [], insts
    head = insts[-1]
    pre, post = head.pre_and_post_stack_effect()
    if pre >= 0:
        if pre == n:
            return [head], insts[:-1]
        if pre < n:
            nxt, rst = partitionInst(insts[:-1], n - pre)
            return nxt + [head], rst
    pre = abs(pre)
    nxt, rst = partitionInst(insts[:-1], pre)
    if post == n:
        return nxt + [head], rst
    if post < n:
        head = nxt + [head]
        nxt, rst = partitionInst(rst, n - post)
        return nxt + head, rst
    return None, None


def transformEscapes(code):
    if not code:
        return code
    transform = ptrans(
        escape,
        trans0(trans0(__.name ^ Esc)),
    ) | one(const(True))
    return transform(code) >> apply(
        lambda hd, tl: transformEscapes(tl) >> (lambda x: [hd] + x)
    )


def transformExpr(code):
    if not code:
        return code
    transform = ptrans(
        loadg, trans0(trans0(__.arg ^ (lambda x: bc.Instr("LOAD_CONST", f"${x}"))))
    ) | one(const(True))
    return transform(code) >> apply(
        lambda hd, tl: transformEscapes(tl) >> (lambda x: [hd] + x)
    )


@do(Right)
def transformArgs(args):
    escTrans < -transformEscapes(args)
    loadgTrans < -transformExpr(escTrans)


def parseNormalStage(stage):
    assert loadg(stage[0]), "The first thing in a stage must be the stage name"
    assert fncall(stage[-1]), "A stage must appear as a function call"
    name = stage[0].arg
    nargs = stage[-1].arg
    parts = []
    rest = stage[1:-1]
    for _ in range(nargs):
        part, rest = partitionInst(rest, 1)
        parts.append(part)
    assert len(parts) == 1, "A stage cannot have more than one set of args"
    return Stage(name, transformArgs(parts[0]))


def parseExpr(instrs: List[bc.Instr]):
    if not instrs:
        return None
    if loadv(instrs[-1]):
        return Sym(instrs[-1].arg)
    if binOp(instrs[-1]):
        op = instrs[-1].name
        rem = instrs[:-1]
        b, rem = partitionInst(rem, 1)
        a, rem = partitionInst(rem, 1)
        print(f"operand {a = }")
        print(f"operand {b = }")
        assert not rem
        return BinOp(OPMAP[op], parseExpr(a), parseExpr(b))
    if fncall(instrs[-1]):
        fn_name = instrs[0].arg
        parts = []
        rem = instrs[1:-1]
        for _ in range(instrs[-1].arg):
            part, rem = partitionInst(rem, 1)
            parts.append(parseExpr(part))
        return Call(fn_name, parts)


def parseSetStage(stage: SetField):
    field_name = stage.name
    raw_expr = list(map(parseExpr, stage.expr))
    return Stage(
        "$set",
        lambda: {
            name: analyse_var(expr).to_json()
            for name, expr in zip(field_name, raw_expr)
        },
    )


def parseStage(stage):
    if isinstance(stage, SetField):
        print("parsing set stage")
        return parseSetStage(stage)
    else:
        if none(stage[0]) and ret(stage[1]):
            return None
        print("parsing non set stage")
        return parseNormalStage(stage)


def aggregate(fn):
    b = bc.Bytecode.from_code(fn.__code__)
    parts = []
    buf = []
    for i in b:
        if popTop(i):
            parts.append(buf)
            buf = []
        elif storeFast(i):
            if parts and isinstance(parts[-1], SetField):
                parts[-1].name.append(i.arg)
                parts[-1].expr.append(buf)
            else:
                parts.append(SetField([i.arg], [buf]))
            buf = []
        else:
            buf.append(i)
    if buf:
        parts.append(buf)

    print(parts)
    res = Pipeline(mp1(parseStage, parts))
    print(res)
    return res
