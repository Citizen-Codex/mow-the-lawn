from typing import Literal, TypeAlias, TypedDict

Move: TypeAlias = Literal["u", "d", "l", "r"]
Point: TypeAlias = tuple[int, int]
Grid: TypeAlias = list[list[int]]


MOVE_DELTAS: dict[Move, tuple[int, int]] = {
    "u": (-1, 0),
    "d": (1, 0),
    "l": (0, -1),
    "r": (0, 1),
}


class Path(TypedDict):
    start: Point | None
    moves: list[Move]
