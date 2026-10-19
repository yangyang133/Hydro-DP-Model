import json
from collections import defaultdict, deque

class SubbasinGraph:
    """构建子流域有向图并生成拓扑排序"""

    def __init__(self, topo_file):
        """
        topo_file: json, {subbasin_id: downstream_id}
        """
        with open(topo_file, "r") as f:
            self.topo = json.load(f)
        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)
        self._build_graph()

    def _build_graph(self):
        for sb, downstream in self.topo.items():
            if downstream is not None:
                self.graph[downstream].append(sb)
                self.in_degree[sb] += 1
            if sb not in self.in_degree:
                self.in_degree[sb] = self.in_degree.get(sb, 0)

    def topological_order(self):
        """返回子流域拓扑排序，从上游到下游"""
        queue = deque([sb for sb, deg in self.in_degree.items() if deg == 0])
        order = []
        in_degree = self.in_degree.copy()
        while queue:
            sb = queue.popleft()
            order.append(sb)
            for downstream in self.graph.get(sb, []):
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)
        return order
