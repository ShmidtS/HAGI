from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Self


@dataclass(frozen=True, slots=True, init=False)
class Term:
    name: str
    args: tuple[Term, ...]
    _hash: int = field(init=False, repr=False, compare=False)

    def __init__(self, name: str, args: Iterable[Term] | None = None):
        args_tuple = tuple(args or ())
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "args", args_tuple)
        object.__setattr__(self, "_hash", hash((self.name, args_tuple)))

    @classmethod
    def Atom(cls, name: str) -> Self:
        return cls(name)

    @classmethod
    def Var(cls, name: str) -> Self:
        normalized = str(name)
        return cls(normalized if normalized.startswith("$") else f"${normalized}")

    @classmethod
    def Compound(cls, name: str, args: Iterable[Term]) -> Self:
        return cls(name, args)

    @property
    def is_atom(self) -> bool:
        return not self.args and not self.name.startswith("$")

    @property
    def is_var(self) -> bool:
        return not self.args and self.name.startswith("$")

    @property
    def is_compound(self) -> bool:
        return bool(self.args)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Term) and self.name == other.name and self.args == other.args

    def __repr__(self) -> str:
        if self.args:
            return f"{self.name}({', '.join(repr(arg) for arg in self.args)})"
        return self.name


Atom = Term.Atom
Var = Term.Var
Compound = Term.Compound
