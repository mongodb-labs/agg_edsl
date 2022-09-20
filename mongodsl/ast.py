from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List


class Node(ABC):
    @abstractmethod
    def to_json(self, env) -> str:
        ...


@dataclass
class PyVar(Node):
    name: str

    def to_json(self, env):
        return env.get(self.name)


@dataclass
class Raw(Node):
    val: Any

    def to_json(self, env):
        return self.val


@dataclass
class Const(Node):
    val: Any

    def to_json(self, env):
        return {"$const": self.val}


@dataclass
class Sym(Node):
    name: str

    def to_json(self, env):
        return self.name


@dataclass
class Var(Node):
    name: str

    def to_json(self, env):
        return f"${self.name}"


@dataclass
class BinOp(Node):
    op_name: str
    operand_a: Node
    operand_b: Node

    def to_json(self, env):
        return {
            f"${self.op_name}": [
                self.operand_a.to_json(env),
                self.operand_b.to_json(env),
            ]
        }


@dataclass
class BinCmp(Node):
    op_name: str
    operand_a: Node
    operand_b: Node

    def to_json(self, env):
        return {
            f"${self.op_name}": [
                self.operand_a.to_json(env),
                self.operand_b.to_json(env),
            ]
        }


@dataclass
class ExprWrapper(Node):
    expr: Node

    def to_json(self, env):
        return {"$expr": self.expr.to_json(env)}


@dataclass
class FieldBinCmp(Node):
    op_name: str
    field_name: str
    val: Node

    def to_json(self, env):
        return {"$expr": {self.field_name: {f"${self.op_name}": self.val.to_json(env)}}}


@dataclass
class Call(Node):
    fn_name: str
    args: List[Node]
    listp: bool = False

    def to_json(self, env):
        return {
            f"${self.fn_name}": [x.to_json(env) for x in self.args]
            if self.listp
            else self.args[0].to_json(env)
        }
