from typing import TypeVar
from enum import Enum

T = TypeVar("T", bound=Enum)

def parse_type(type_str: str, TypeClass: T) -> T:
    from itertools import combinations

    for i in range(1, 3 + 1):
        for combination in combinations(TypeClass, i):  # pyright: ignore
            combo = list(combination)
            test_type = None
            if len(combo) == 1:
                test_type = combo[0]
            elif len(combo) == 2:
                test_type = combo[0] | combo[1]
            elif len(combo) == 3:
                test_type = combo[0] | combo[1] | combo[2]
            try:
                if test_type.to_str() == type_str:
                    return test_type
            except:
                pass
    assert False
