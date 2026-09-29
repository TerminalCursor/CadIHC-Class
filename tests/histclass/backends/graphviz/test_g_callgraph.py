from pytest import Subtests


def test_Callgraph(subtests: Subtests) -> None:
    """# Test Basic Callgraph"""
    from histclass.backends.graphviz.callgraph import GraphvizCallgraph

    gv_cg = GraphvizCallgraph()
    print(gv_cg, gv_cg.data, gv_cg.enabled)
    with subtests.test("Initialization"):
        assert not gv_cg.enabled, "Callgraph should be disabled by default"
        assert len(gv_cg.data) == 0
    with subtests.test("Adding Elements"):
        gv_cg.add("First Test (GV_CG)")
        assert len(gv_cg.data) == 0
        gv_cg.toggle()
        gv_cg.add("Second Test (GV_CG)")
        assert len(gv_cg.data) == 1
        gv_cg.toggle(to=True)
        gv_cg.add("Third Test (GV_CG)")
        assert len(gv_cg.data) == 2
        gv_cg.toggle(to=False)
        gv_cg.add("Fourth Test (GV_CG)")
        assert len(gv_cg.data) == 2
    with subtests.test("Clearing Elements"):
        gv_cg.clear()
        assert len(gv_cg.data) == 2, "Clearing should only happen on enabled callgraphs"
        gv_cg.toggle()
        gv_cg.clear()
        assert len(gv_cg.data) == 0, "Clearing should only happen on enabled callgraphs"
    with subtests.test("Saving Callgraph"):
        gv_cg.save("tests/outputs/tests_callgraph")
        assert True, "Made it here"
