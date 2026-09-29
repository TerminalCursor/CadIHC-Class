def test_callgraph() -> None:
    """# Test Callgraph Interface"""
    from histclass.interfaces.callgraph import Callgraph

    cg = Callgraph()
    assert not cg.enabled, "Should start disabled!"
    assert len(cg.data) == 0, "Should start empty!"
    cg.toggle()
    assert cg.enabled, "Toggle should enable"
    cg.toggle(to=True)
    assert cg.enabled, "Toggle should remain on"
    cg.toggle()
    cg.toggle(to=False)
    assert not cg.enabled, "Toggle should remain off"

    def throwaway(x: int) -> int:
        """# Increment by one"""
        return x + 1

    assert cg(throwaway, 1) == 2, "Should return result"
