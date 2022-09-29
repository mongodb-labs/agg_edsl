import bytecode as bc

from typing import List
from dataclasses import dataclass


@dataclass
class Path:
    parts: List[str]


UDF_INSTR = [
    "NOP",
    "RET",
    "CONST",
    "POP",
    "BINOP",
    "BINCMP",
    "READ_FIELD",
    "LABEL",
    "JUMP_IF",
    "JUMP_IF_NOT",
    "JUMP_ABS",
    "LAST_OP",
]

UDF_BINOP = ["ADD", "SUB", "MUL", "DIV"]
UDF_BIMCMP = ["LT", "LE", "GT", "GE", "EQ", "NE"]


def transformUDF(fn):
    raw_b = bc.Bytecode.from_code(fn.__code__)
    fn_args = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    assert (
        len(fn_args) == 1
    ), "Currently only one argument referring to the doc is supported"
    transB = []
    labels = {}
    label_num = 1
    for instr in raw_b:
        if isinstance(instr, bc.Label):
            labels[instr] = label_num
            label_num += 1
            transB.append(instr)
            continue
        if instr.name == "LOAD_ATTR":
            assert isinstance(
                transB[-1], Path
            ), "Field path name must be done on document"
            transB[-1].parts.append(instr.arg)
            continue
        if instr.name == "LOAD_FAST" and instr.arg in fn_args:
            transB.append(Path([]))
            continue
        transB.append(instr)

    res = []
    data = []
    for instr in transB:
        if isinstance(instr, bc.Label):
            res.append(UDF_INSTR.index("LABEL"))
            res.append(labels[instr])
        elif isinstance(instr, bc.Instr):
            name = instr.name
            arg = instr.arg

            if name == "LOAD_CONST":
                res.append(UDF_INSTR.index("CONST"))
                res.append(arg)
            elif name == "POP_TOP":
                res.append(UDF_INSTR.index("POP"))
                res.append(0)
            elif name == "RETURN_VALUE":
                res.append(UDF_INSTR.index("RET"))
                res.append(0)
            elif name == "BINARY_ADD":
                res.append(UDF_INSTR.index("BINOP"))
                res.append(UDF_BINOP.index("ADD"))
            elif name == "BINARY_SUBTRACT":
                res.append(UDF_INSTR.index("BINOP"))
                res.append(UDF_BINOP.index("SUB"))
            elif name == "BINARY_MULTIPLY":
                res.append(UDF_INSTR.index("BINOP"))
                res.append(UDF_BINOP.index("MUL"))
            elif name == "BINARY_TRUE_DIVIDE":
                res.append(UDF_INSTR.index("BINOP"))
                res.append(UDF_BINOP.index("DIV"))
            elif name == "COMPARE_OP":
                res.append(UDF_INSTR.index("BINCMP"))
                res.append(
                    UDF_BIMCMP.index(
                        {
                            bc.Compare.GE: "GE",
                            bc.Compare.GT: "GT",
                            bc.Compare.LE: "LE",
                            bc.Compare.LT: "LT",
                            bc.Compare.EQ: "EQ",
                            bc.Compare.NE: "NE",
                        }[arg]
                    )
                )
            elif name == "POP_JUMP_IF_FALSE":
                res.append(UDF_INSTR.index("JUMP_IF_NOT"))
                res.append(labels[arg])
            elif name == "POP_JUMP_IF_TRUE":
                res.append(UDF_INSTR.index("JUMP_IF"))
                res.append(labels[arg])
            elif name == "JUMP_ABSOLUTE":
                res.append(UDF_INSTR.index("JUMP_ABS"))
                res.append(labels[arg])

        elif isinstance(instr, Path):
            res.append(UDF_INSTR.index("READ_FIELD"))
            path_name = ".".join(instr.parts)
            if path_name not in data:
                data.append(path_name)
            res.append(data.index(path_name))

    if len(res) % 2 == 1:
        res.append(0)
    res.append(127)
    for d in data:
        res.extend(map(ord, d))
        res.append(0)
    return "".join(map(chr, res))
