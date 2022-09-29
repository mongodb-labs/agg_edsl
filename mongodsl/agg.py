from dataclasses import dataclass
from typing import Any, List

import bytecode as bc
from fpy.composable.collections import and_, apply, get0, mp1, of_, or_, trans0
from fpy.control.functor import Functor, fmap
from fpy.data.either import Right
from fpy.data.function import const, id_
from fpy.experimental.do import do
from fpy.parsec.parsec import many, one, ptrans
from fpy.utils.placeholder import __


from mongodsl.udf import transformUDF
from mongodsl.ast import (
    BinCmp,
    BinOp,
    Call,
    Const,
    ExprWrapper,
    PyVar,
    Raw,
    Sym,
    Var,
    Bytecode,
)
from mongodsl.scope import analyse_var

AGG_REG = dict()


@dataclass
class Ret:
    lineno: int


@dataclass
class Esc:
    name: str


@dataclass
class PyLocalVar:
    name: str

    def pre_and_post_stack_effect(self):
        # for instr partition to work
        return 1, 0


@dataclass
class DSLVar:
    name: str

    def pre_and_post_stack_effect(self):
        # for instr partition to work
        return 1, 0


@dataclass
class Subpipe:
    name: str

    def pre_and_post_stack_effect(self):
        # for instr partition to work
        return 1, 0


@dataclass
class Del:
    name: str

    def pre_and_post_stack_effect(self):
        # for instr partition to work
        return 0, 0


@dataclass
class UnsetField:
    name: List[str]


@dataclass
class SetField:
    name: List[str]
    expr: List[List[bc.Instr]]


@dataclass
class Block:
    name: str
    argExpr: List[bc.Instr]
    inner: List[SetField]


@dataclass
class OpBlock:
    fieldName: str
    op: str
    argExpr: List[bc.Instr]
    inner: List[SetField]


@dataclass
class Stage:
    name: str
    expr: Any

    def to_json(self):
        return {f"{self.name}": self.expr()}


@dataclass
class RawBC:
    instrs: List[bc.Instr]


@dataclass
class Pipeline:
    stages: List

    def apply(self, col):
        return col.aggregate(self.to_json())

    def to_json(self):
        res = []
        for s in self.stages:
            if isinstance(s, Stage):
                res.append(s.to_json())
            elif isinstance(s, ApplicablePipeline):
                res.extend(s.to_json())
        return res

    def concat(self, o):
        assert isinstance(
            o, Pipeline
        ), "pipeline can only be composed with pipeline, what's the problem?"
        return Pipeline(self.stages + o.stages)


@dataclass
class ApplicablePipeline(Functor[Pipeline]):
    def __call__(self, col):
        return self.val.apply(col)

    def __add__(self, o: Functor[Pipeline]):
        return fmap(o, lambda x: self.val.concat(x))

    def to_json(self):
        return self.val.to_json()


BLOCK_ARG = {"group": "_id", "groups": "_id"}


instr = lambda x: isinstance(x, bc.Instr)
loadg = and_(instr, __.name == "LOAD_GLOBAL")
loadf = and_(instr, __.name == "LOAD_FAST")
loadv = or_(loadg, loadf)
load_ = and_(loadg, __.arg == "_")
loadm = and_(instr, __.name == "LOAD_METHOD")
load = and_(
    instr,
    __.name
    ^ of_(
        "LOAD_GLOBAL",
        "LOAD_NAME",
        "LOAD_FAST",
        "LOAD_CLOSURE",
        "LOAD_DEREF",
        "LOAD_CLASSDEREF",
    ),
)
lattr = and_(instr, __.name == "LOAD_ATTR")
const = and_(instr, __.name == "LOAD_CONST")
delv = and_(instr, __.name == "DELETE_FAST")
escape = one(load_) >> one(loadm)

fncall = and_(instr, __.name == "CALL_FUNCTION")
kwcall = and_(instr, __.name == "CALL_FUNCTION_KW")
popTop = and_(instr, __.name == "POP_TOP")
popBlock = and_(instr, __.name == "POP_BLOCK")
popExcept = and_(instr, __.name == "POP_EXCEPT")
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

binCmp = and_(instr, __.name == "COMPARE_OP")

dupTop = and_(instr, __.name == "DUP_TOP")
dupTop2 = and_(instr, __.name == "DUP_TOP_TWO")
rot2 = and_(instr, __.name == "ROT_TWO")
rot3 = and_(instr, __.name == "ROT_THREE")
rot4 = and_(instr, __.name == "ROT_FOUR")

OPMAP = {
    "BINARY_ADD": "add",
    "BINARY_SUBTRACT": "subtract",
    "BINARY_MULTIPLY": "multiply",
    "BINARY_TRUE_DIVIDE": "divide",
}

CMPMAP = {
    bc.Compare.GE: "gte",
    bc.Compare.GT: "gt",
    bc.Compare.LE: "lte",
    bc.Compare.LT: "lt",
    bc.Compare.EQ: "eq",
    bc.Compare.NE: "ne",
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
    # if not instr(head):
    #     nxt, rst = partitionInst(insts[:-1], n)
    #     return nxt + [head], rst
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


def parseBareStage(stage):
    assert isinstance(
        stage[0], DSLVar
    ), f"The first thing in a stage must be the stage name, got: {stage}"
    assert fncall(stage[-1]), "A stage must appear as a function call"
    name = stage[0].name
    nargs = stage[-1].arg
    # print(f"{name = }")
    # print(f"{nargs = }")
    parts = []
    rest = stage[1:-1]
    for _ in range(nargs):
        part, rest = partitionInst(rest, 1)
        expr = parseExpr(part)
        if name in ("match",):
            # a simple and stupid solution for wrapping agg expr in stages like $match
            expr = ExprWrapper(expr)
        parts.append(expr)
    # print(f"{parts = }")
    assert len(parts) == 1, "A stage cannot have more than one set of args"
    return lambda env, glob: Stage(
        f"${name}", lambda: analyse_var(parts[0]).to_json(env, glob)
    )


def parseNamedStage(stage):
    assert isinstance(
        stage[0], DSLVar
    ), f"The first thing in a stage must be the stage name, got: {stage}"
    assert kwcall(stage[-1]), "A stage must appear as a function call"
    name = stage[0].name
    fields = stage[-2].arg
    nargs = stage[-1].arg
    # print(f"{name = }")
    # print(f"{fields = }")
    # print(f"{nargs = }")
    parts = []
    rest = stage[1:-2]
    for _ in range(nargs):
        part, rest = partitionInst(rest, 1)
        parts.append(parseExpr(part))
    # print(f"{parts = }")
    assert len(parts) == nargs, "Number of args doesn't match number of fields"
    return lambda env, glob: Stage(
        f"${name}",
        lambda: {
            field_name: analyse_var(expr).to_json(env, glob)
            for field_name, expr in zip(reversed(fields), parts)
        },
    )


def parseSubpipe(stage):
    # print(f"Subpipe {stage = }")
    assert isinstance(stage[0], Subpipe)
    sub_co = AGG_REG[stage[0].name]
    instrs = []
    for instr in stage[1:]:
        if isinstance(instr, bc.Instr):
            instrs.append(instr)
        elif isinstance(instr, PyLocalVar):
            instrs.append(bc.Instr("LOAD_FAST", instr.name))
    # print(f"{sub_co = }")
    return RawBC([bc.Instr("LOAD_CONST", sub_co)] + instrs)


def parseNormalStage(stage):
    # print(stage)
    if fncall(stage[-1]):
        return parseBareStage(stage)
    if kwcall(stage[-1]):
        return parseNamedStage(stage)
    raise Exception(stage)


def parseUDF(instrs):
    udf_name = instrs[1].name
    # print(f"{udf_name = }")
    code = [
        bc.Instr("LOAD_CONST", transformUDF),
        bc.Instr("LOAD_GLOBAL", udf_name),
        bc.Instr("CALL_FUNCTION", 1),
        bc.Instr("LOAD_CONST", ("$udf",)),
        bc.Instr("BUILD_CONST_KEY_MAP", 1),
        bc.Instr("RETURN_VALUE"),
    ]
    return Bytecode(code)


def parseExpr(instrs: List[bc.Instr]):
    if not instrs:
        return None
    if isinstance(instrs[-1], DSLVar):
        return Sym(instrs[-1].name)
    if isinstance(instrs[-1], PyLocalVar):
        return PyVar(instrs[-1].name)
    if loadv(instrs[-1]):
        return Sym(instrs[-1].arg)
    if binOp(instrs[-1]):
        op = instrs[-1].name
        rem = instrs[:-1]
        b, rem = partitionInst(rem, 1)
        a, rem = partitionInst(rem, 1)
        # print(f"operand {a = }")
        # print(f"operand {b = }")
        assert not rem
        return BinOp(OPMAP[op], parseExpr(a), parseExpr(b))
    if binCmp(instrs[-1]):
        op = instrs[-1].arg
        rem = instrs[:-1]
        b, rem = partitionInst(rem, 1)
        a, rem = partitionInst(rem, 1)
        # print(f"operand {a = }")
        # print(f"operand {b = }")
        assert not rem
        return BinCmp(CMPMAP[op], parseExpr(a), parseExpr(b))
    if fncall(instrs[-1]):
        assert isinstance(
            instrs[0], DSLVar
        ), f"a function call must happen to a DSL variable, got: {instrs[0]}"
        fn_name = instrs[0].name
        if fn_name == "udf":
            return parseUDF(instrs)
        parts = []
        rem = instrs[1:-1]
        for _ in range(instrs[-1].arg):
            part, rem = partitionInst(rem, 1)
            parts.append(parseExpr(part))
        return Call(fn_name, parts)
    if const(instrs[-1]):
        return Raw(instrs[-1].arg)


def parseSetStage(stage: SetField):
    field_name = stage.name
    raw_expr = list(map(parseExpr, stage.expr))
    return lambda env, glob: Stage(
        "$set",
        lambda: {
            name: analyse_var(expr).to_json(env, glob)
            for name, expr in zip(field_name, raw_expr)
        },
    )


def parseBlockStage(stage: Block):
    arg = parseExpr(stage.argExpr)
    argFieldName = BLOCK_ARG.get(stage.name, None)
    inner_name = []
    inner_expr = []
    for inner in stage.inner:
        if not inner:
            continue
        # assert isinstance(inner, SetField)
        inner_name.append(inner.name[0])
        inner_expr.append(parseExpr(inner.expr[0]))
    return lambda env, glob: Stage(
        f"${stage.name}",
        lambda: (
            {argFieldName: analyse_var(arg).to_json(env, glob)} if argFieldName else {}
        )
        | {
            name: analyse_var(expr).to_json(env, glob)
            for name, expr in zip(inner_name, inner_expr)
        },
    )


def parseStage(stage):
    # print(f"{stage = }")
    if isinstance(stage, SetField):
        # print("parsing set stage")
        return parseSetStage(stage)
    elif isinstance(stage, Block):
        # print("parsing block stage")
        return parseBlockStage(stage)
    elif isinstance(stage, UnsetField):
        # print("del stage")
        return lambda env, glob: Stage("$unset", lambda: stage.name)
    else:
        if none(stage[0]) and ret(stage[1]):
            return None
        # print("parsing normal stage")
        if isinstance(stage[0], Subpipe):
            return parseSubpipe(stage)
        return parseNormalStage(stage)


def partitionBareCall(head, instrs):
    assert isinstance(head[0], DSLVar)
    assert fncall(head[-1])
    inner = []
    while instrs:
        i = instrs.pop(0)
        if popBlock(i):
            while instrs:
                if popExcept(instrs.pop(0)):
                    break
            instrs.pop(0)
            break
        inner.append(i)
    inner_parts = []
    while inner:
        part, inner = partitionPipeline(inner)
        inner_parts.append(part)
    block_name = head[0].name
    return Block(block_name, head[1:-1], inner_parts), instrs


def partitionNamedCall(head, instrs):
    assert isinstance(head[0], DSLVar)
    assert kwcall(head[-1])
    # print(f"{head = }")
    # print(f"{instrs = }")


def partitionBareBlock(head, instrs):
    if not instrs:
        return None
    if fncall(head[-1]):
        return partitionBareCall(head, instrs)
    if kwcall(head[-1]):
        return partitionNamedCall(head, instrs)
    raise Exception(f"Don't know how to handle: {head[-1]}")


def partitionBlock(head, instrs):
    if not instrs:
        return None
    alias = instrs.pop(0)
    if popTop(alias):
        return partitionBareBlock(head, instrs)
    if storeFast(alias):
        blk, rest = partitionBareBlock(head, instrs)
        return OpBlock(alias.arg, blk.block_name, blk.argExpr, blk.inner), rest
    raise Exception(f"SETUP_WITH followed by unknown instruction: {alias}")


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
        elif isinstance(i, Del):
            return UnsetField([i.name]), instrs
        elif lattr(i):
            assert isinstance(
                buf[-1], DSLVar
            ), "path needs to be a part of dsl var name"
            prev = buf.pop()
            prev.name += f".{i.arg}"
            buf.append(prev)

        else:
            buf.append(i)
    return buf, []


def aggregate(fn):
    raw_b = bc.Bytecode.from_code(fn.__code__)
    b = [i for i in raw_b if instr(i)]
    # print(f"{b = }")
    fn_args = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    # print(f"{fn_args = }")
    transArg = ptrans(
        one(and_(loadf, __.arg ^ of_(*fn_args))), trans0(trans0(__.arg ^ PyLocalVar))
    )
    transSubpipe = ptrans(
        one(and_(load, __.arg ^ of_(*AGG_REG.keys()))), trans0(trans0(__.arg ^ Subpipe))
    )
    transLoad = ptrans(one(load), trans0(trans0(__.arg ^ DSLVar)))
    transDel = ptrans(one(delv), trans0(trans0(__.arg ^ Del)))
    transformedB = (
        many(transDel | transArg | transSubpipe | transLoad | one(instr))(b) >> get0
    )
    # print(f"{transformedB = }")
    parts = []
    while transformedB:
        part, transformedB = partitionPipeline(transformedB)
        # print(f"{part = }")
        if isinstance(part, SetField):
            if parts and isinstance(parts[-1], SetField):
                parts[-1].name += part.name
                parts[-1].expr += part.expr
                continue
        elif isinstance(part, UnsetField):
            if parts and isinstance(parts[-1], UnsetField):
                parts[-1].name += part.name
                continue
        parts.append(part)
    # print(parts)
    stages = mp1(parseStage, parts)
    # print(stages)
    res_fn = lambda env: Pipeline(list(map(lambda x: x(env), stages)))
    res_bc = bc.Bytecode([])
    if fn_args:
        for arg in fn_args:
            res_bc.append(bc.Instr("LOAD_FAST", arg=arg))
        res_bc.append(bc.Instr("LOAD_CONST", arg=fn_args))
        res_bc.append(bc.Instr("BUILD_CONST_KEY_MAP", arg=len(fn_args)))
    else:
        res_bc.append(bc.Instr("LOAD_CONST", arg=None))
    res_bc.extend(
        [
            bc.Instr("LOAD_GLOBAL", "globals"),
            bc.Instr("CALL_FUNCTION", 0),
            bc.Instr("BUILD_TUPLE", 2),
        ]
    )
    stage_count = 0
    for stage in stages:
        if not stage:
            continue
        stage_count += 1
        if isinstance(stage, RawBC):
            res_bc.extend(stage.instrs)
            res_bc.append(bc.Instr("ROT_TWO"))
            continue
        # print(f"{stage = }")
        res_bc.append(bc.Instr("LOAD_CONST", arg=stage))
        res_bc.append(bc.Instr("ROT_TWO"))
        res_bc.append(bc.Instr("DUP_TOP"))
        res_bc.append(bc.Instr("ROT_THREE"))
        res_bc.append(bc.Instr("CALL_FUNCTION_EX", 0))
        res_bc.append(bc.Instr("ROT_TWO"))
    res_bc.append(bc.Instr("POP_TOP"))
    res_bc.append(bc.Instr("BUILD_LIST", arg=stage_count))
    res_bc.append(bc.Instr("LOAD_CONST", arg=Pipeline))
    res_bc.append(bc.Instr("ROT_TWO"))
    res_bc.append(bc.Instr("CALL_FUNCTION", arg=1))
    res_bc.append(bc.Instr("LOAD_CONST", arg=ApplicablePipeline))
    res_bc.append(bc.Instr("ROT_TWO"))
    res_bc.append(bc.Instr("CALL_FUNCTION", arg=1))
    res_bc.append(bc.Instr("RETURN_VALUE"))

    # print(res_bc)

    res_bc.argcount = len(fn_args)
    res_bc.argnames.extend(fn_args)
    res_bc.name = fn.__name__
    res_bc.filename = fn.__code__.co_filename
    res_bc.flags = res_bc.flags | 16
    res_bc.update_flags()
    res_co = res_bc.to_code()
    fn.__code__ = res_co

    AGG_REG[fn.__name__] = fn
    return fn
