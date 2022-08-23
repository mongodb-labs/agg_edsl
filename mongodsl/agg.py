from dataclasses import dataclass
from typing import Any, List

import bytecode as bc
from fpy.composable.collections import and_, apply, const, mp1, or_, trans0
from fpy.data.either import Right
from fpy.experimental.do import do
from fpy.parsec.parsec import many, one, ptrans
from fpy.utils.placeholder import __

from mongodsl.ast import BinOp, Call, Sym, Var, Const
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
class GroupBlock:
    idField: List[bc.Instr]
    inner: List[SetField]


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


BLOCKS = {"group": GroupBlock}


instr = lambda x: isinstance(x, bc.Instr)
loadg = and_(instr, __.name == "LOAD_GLOBAL")
loadf = and_(instr, __.name == "LOAD_FAST")
loadv = or_(loadg, loadf)
load_ = and_(loadg, __.arg == "_")
loadm = and_(instr, __.name == "LOAD_METHOD")
const = and_(instr, __.name == "LOAD_CONST")
escape = one(load_) >> one(loadm)

fncall = and_(instr, __.name == "CALL_FUNCTION")
popTop = and_(instr, __.name == "POP_TOP")
popBlock = and_(instr, __.name == "POP_BLOCK")
storeFast = and_(instr, __.name == "STORE_FAST")
none = and_(const, __.arg == None)
ret = and_(instr, __.name == "RETURN_VALUE")

setupWith = and_(instr, __.name == "SETUP_WITH")
jumpForward = and_(instr, __.name == "JUMP_FORWARD")

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
    if const(instrs[-1]):
        return Const(instrs[-1].arg)


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


def parseGroupStage(stage: GroupBlock):
    id_expr = parseExpr(stage.idField)
    inner_name = []
    inner_expr = []
    for inner in stage.inner:
        if not inner:
            continue
        assert isinstance(inner, SetField)
        inner_name.append(inner.name[0])
        inner_expr.append(parseExpr(inner.expr[0]))
    return Stage(
        "$group",
        lambda: {"_id": analyse_var(id_expr).to_json()}
        | {
            name: analyse_var(expr).to_json()
            for name, expr in zip(inner_name, inner_expr)
        },
    )


def parseStage(stage):
    if isinstance(stage, SetField):
        print("parsing set stage")
        return parseSetStage(stage)
    elif isinstance(stage, GroupBlock):
        print("parsing group stage")
        return parseGroupStage(stage)
    else:
        if none(stage[0]) and ret(stage[1]):
            return None
        print("parsing non set stage")
        return parseNormalStage(stage)


def partitionBlock(head, instrs):
    if not instrs:
        return None
    assert loadg(head[0])
    assert fncall(head[-1])
    inner = []
    while instrs:
        i = instrs.pop(0)
        if popBlock(i):
            while instrs and not jumpForward(instrs[0]):
                instrs.pop(0)
            assert instrs, "instrs cannot be empty here"
            end_label = instrs[0].arg
            while instrs and instrs[0] != end_label:
                instrs.pop(0)
            assert instrs[0] == end_label
            instrs.pop(0)
            break
        inner.append(i)
    print(f"{inner = }")
    inner_parts = []
    while inner:
        part, inner = partitionPipeline(inner)
        inner_parts.append(part)
    block_name = head[0].arg
    return BLOCKS[block_name](head[1:-1], inner_parts), instrs


def partitionPipeline(instrs):
    if not instrs:
        return None
    buf = []
    while instrs:
        i = instrs.pop(0)
        if popTop(i):
            return buf, instrs
        elif setupWith(i):
            return partitionBlock(buf, instrs)
        elif storeFast(i):
            return SetField([i.arg], [buf]), instrs
        else:
            buf.append(i)
    return buf, []


def aggregate(fn):
    b = bc.Bytecode.from_code(fn.__code__)
    parts = []
    while b:
        part, b = partitionPipeline(b)
        if isinstance(part, SetField):
            if parts and isinstance(parts[-1], SetField):
                parts[-1].name += part.name
                parts[-1].expr += part.expr
            else:
                parts.append(part)
        else:
            parts.append(part)
    print(parts)
    res = Pipeline(mp1(parseStage, parts))
    print(res)
    return res
