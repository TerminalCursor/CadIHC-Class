from pytest import Subtests


def test_maybe_monad(subtests: Subtests) -> None:
    """# Test Maybe Monad"""
    from histclass.interfaces.monad.maybe import Maybe

    m: Maybe[int] = Maybe()
    with subtests.test("Initialization"):
        assert m.data is None, "Default Null"
    with subtests.test("Immutable Catch"):
        m.catch(3)
        assert m.data is None, "Immutable by default"
    with subtests.test("Boolean Treatment"):
        assert not m, "Null so should be false"
    with subtests.test("Catch Replacement"):
        m = m.catch(3)
        assert m.unwrap() == 3, "Replacement"
    with subtests.test("Catch Non-Nulled"):
        m = m.catch(4)
        assert m.unwrap() == 3, "Catch should not replace data"
    with subtests.test("Boolean Treatment"):
        assert m, "Non-null so should be true"
    with subtests.test("Fun Call Testing"):

        def inc(x: int) -> int:
            """# Increments a number"""
            return x + 1

        m = m(inc)
        assert m.unwrap() == 4, "Should increment data"
    n: Maybe[str]
    with subtests.test("Fun Call Mutation"):

        def tostr(x: int) -> str:
            """# Returns number as string"""
            return str(x)

        n = m(tostr)
        assert n.unwrap() == "4", "Should stringify"
