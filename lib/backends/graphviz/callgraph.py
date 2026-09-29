from ...interfaces.callgraph import Callgraph
from ...interfaces.monad.maybe import Maybe
import graphviz
from typing import Self


class GraphvizCallgraph(Callgraph):
    def __init__(self):
        self.graph = graphviz.Digraph(comment="Callgraph")
        self.graph.attr(rankdir="LR")

    def add(self, data: str) -> Self:
        """# Add Data To Callgraph"""
        if not self.enabled:
            return self
        self.data.append(data)
        current_node_id = len(self.data) - 1
        previous_node_id_m = Maybe()
        if current_node_id > 0:
            previous_node_id_m = Maybe(current_node_id - 1)

        cnis = f"{current_node_id}"
        self.graph.node(
            cnis,
            label=f"<{cnis}_port> {self.data[current_node_id]}",
            shape="record",
        )
        if previous_node_id_m:
            pnis = f"{previous_node_id_m.unwrap()}"
            self.graph.edge(
                f"{pnis}:{pnis}_port",
                f"{cnis}:{cnis}_port",
            )
        return self

    def clear(self) -> Self:
        """# Clear Callgraph"""
        if not self.enabled:
            return self
        self.data = []
        self.graph = graphviz.Digraph(comment="Callgraph")
        self.graph.attr(rankdir="LR")
        return self

    def save(self, name, *args, view=False, format="png", **kwargs) -> Self:
        """# Save Callgraph"""
        if not self.enabled:
            return self
        self.graph.render(name, *args, view=view, format=format, **kwargs)
        return self
