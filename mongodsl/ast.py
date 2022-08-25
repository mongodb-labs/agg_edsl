from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List


class Node(ABC):
    @abstractmethod
    def to_json(self) -> str:
        ...


@dataclass
class Raw(Node):
    val: Any

    def to_json(self):
        return self.val


@dataclass
class Const(Node):
    val: Any

    def to_json(self):
        return {"$const": self.val}


@dataclass
class Sym(Node):
    name: str

    def to_json(self):
        return self.name


@dataclass
class Var(Node):
    name: str

    def to_json(self):
        return f"${self.name}"


@dataclass
class BinOp(Node):
    op_name: str
    operand_a: Node
    operand_b: Node

    def to_json(self):
        return {
            f"${self.op_name}": [self.operand_a.to_json(), self.operand_b.to_json()]
        }


@dataclass
class BinCmp(Node):
    op_name: str
    operand_a: Node
    operand_b: Node

    def to_json(self):
        return {
            f"${self.op_name}": [self.operand_a.to_json(), self.operand_b.to_json()]
        }


@dataclass
class FieldBinCmp(Node):
    op_name: str
    field_name: str
    val: Node

    def to_json(self):
        return {self.field_name: {f"${self.op_name}": self.val.to_json()}}


@dataclass
class Call(Node):
    fn_name: str
    args: List[Node]
    listp: bool = False

    def to_json(self):
        return {
            f"${self.fn_name}": [x.to_json() for x in self.args]
            if self.listp
            else self.args[0].to_json()
        }
