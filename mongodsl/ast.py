from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import bytecode as bc


class Node(ABC):
    @abstractmethod
    def to_json(self, env, glob) -> str:
        ...


@dataclass
class PyVar(Node):
    name: str

    def to_json(self, env, glob):
        return env.get(self.name)


@dataclass
class Raw(Node):
    val: Any

    def to_json(self, env, glob):
        return self.val


@dataclass
class Bytecode(Node):
    code: List[bc.Instr]

    def to_json(self, env, glob):
        bc_ = bc.Bytecode(self.code)
        bc_.update_flags()
        co = bc_.to_code()
        return eval(co, glob or globals(), {**(env or {}), **locals()})


@dataclass
class Const(Node):
    val: Any

    def to_json(self, env, glob):
        return {"$const": self.val}


@dataclass
class Sym(Node):
    name: str

    def to_json(self, env, glob):
        return self.name


@dataclass
class Var(Node):
    name: str

    def to_json(self, env, glob):
        return f"${self.name}"


@dataclass
class BinOp(Node):
    op_name: str
    operand_a: Node
    operand_b: Node

    def to_json(self, env, glob):
        return {
            f"${self.op_name}": [
                self.operand_a.to_json(env, glob),
                self.operand_b.to_json(env, glob),
            ]
        }


@dataclass
class BinCmp(Node):
    op_name: str
    operand_a: Node
    operand_b: Node

    def to_json(self, env, glob):
        return {
            f"${self.op_name}": [
                self.operand_a.to_json(env, glob),
                self.operand_b.to_json(env, glob),
            ]
        }


@dataclass
class ExprWrapper(Node):
    expr: Node

    def to_json(self, env, glob):
        return {"$expr": self.expr.to_json(env, glob)}


@dataclass
class FieldBinCmp(Node):
    op_name: str
    field_name: str
    val: Node

    def to_json(self, env, glob):
        return {
            "$expr": {
                self.field_name: {f"${self.op_name}": self.val.to_json(env, glob)}
            }
        }


@dataclass
class Call(Node):
    fn_name: str
    args: List[Node]
    listp: bool = False

    def to_json(self, env, glob):
        return {
            f"${self.fn_name}": [x.to_json(env, glob) for x in self.args]
            if self.listp
            else self.args[0].to_json(env, glob)
        }
