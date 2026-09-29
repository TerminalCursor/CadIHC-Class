from pytest import Subtests


def test_Callgraph(subtests: Subtests) -> None:
    """# Test Basic Callgraph"""
    from histclass.backends.basic.callgraph import BasicCallgraph

    b_cg = BasicCallgraph()
    with subtests.test("Initialization"):
        assert not b_cg.enabled, "Callgraph should be disabled by default"
        assert len(b_cg.data) == 0
    with subtests.test("Adding Elements"):
        b_cg.add("First Test (B_CG)")
        assert len(b_cg.data) == 0
        b_cg.toggle()
        b_cg.add("Second Test (B_CG)")
        assert len(b_cg.data) == 1
        b_cg.toggle(to=True)
        b_cg.add("Third Test (B_CG)")
        assert len(b_cg.data) == 2
        b_cg.toggle(to=False)
        b_cg.add("Fourth Test (B_CG)")
        assert len(b_cg.data) == 2
    with subtests.test("Clearing Elements"):
        b_cg.clear()
        assert len(b_cg.data) == 2, "Clearing should only happen on enabled callgraphs"
        b_cg.toggle()
        b_cg.clear()
        assert len(b_cg.data) == 0, "Clearing should only happen on enabled callgraphs"
    with subtests.test("Saving Callgraph"):
        b_cg.save()
        assert True, "Made it here"
