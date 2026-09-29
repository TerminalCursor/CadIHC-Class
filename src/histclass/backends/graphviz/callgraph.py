from ...interfaces.callgraph import Callgraph
import graphviz  # pyright: ignore[reportMissingTypeStubs]
from typing import Self, List, Any


class GraphvizCallgraph(Callgraph):
    """# Graphviz Callgraph Backend"""

    data: List[str] = []
    enabled: bool = False

    def __init__(self: Self):
        """# Initialize Graphviz Digraph"""
        super(Callgraph, self).__init__()
        self.graph = graphviz.Digraph(comment="Callgraph")
        self.graph.attr(rankdir="LR")  # pyright: ignore

    def add(self: Self, data: str) -> Self:
        """# Add Data To Callgraph"""
        if not self.enabled:
            return self
        self.data.append(data)
        current_node_id = len(self.data) - 1
        previous_node_id_m: int | None = None
        if current_node_id > 0:
            previous_node_id_m = current_node_id - 1

        cnis = f"{current_node_id}"
        self.graph.node(  # pyright: ignore
            cnis,
            label=f"<{cnis}_port> {self.data[current_node_id]}",
            shape="record",
        )
        if previous_node_id_m is not None:
            pnis = f"{previous_node_id_m}"
            self.graph.edge(  # pyright: ignore
                f"{pnis}:{pnis}_port",
                f"{cnis}:{cnis}_port",
            )
        return self

    def clear(self: Self) -> Self:
        """# Clear Callgraph"""
        if not self.enabled:
            return self
        self.data = []
        self.graph = graphviz.Digraph(comment="Callgraph")
        self.graph.attr(rankdir="LR")  # pyright: ignore
        return self

    def save(
        self: Self,
        name: str,
        *args: Any,
        view: bool = False,
        fmt: str = "png",
        **kwargs: Any,
    ) -> Self:
        """# Save Callgraph
        Only saves when the callgraph is enabled

        name: The name to save it to
        view: Whether or not to show immediately
        fmt: What format to save as
        """
        if not self.enabled:
            return self
        self.graph.render(  # pyright: ignore
            name, *args, view=view, format=fmt, **kwargs
        )
        return self
