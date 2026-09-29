# Select from backends
from ..backends.basic.callgraph import BasicCallgraph

cg = BasicCallgraph()

try:
    from ..backends.graphviz.callgraph import GraphvizCallgraph

    cg = GraphvizCallgraph()
except Exception as e:
    print("Ran into exception attempting to use Graphviz Callgraph backend")
    print(e)
