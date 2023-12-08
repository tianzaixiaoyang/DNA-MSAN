import time
from collections import defaultdict


class Personalized():
    def __init__(self, nx_G, mask, walks_num):
        self.G = nx_G
        self.mask = mask
        self.walks_num = walks_num

    # iterative version
    def dfs(self, start_node, walks_num):
        stack=[]
        stack.append(start_node)
        seen=set()
        seen.add(start_node)
        walks = []
        mask_list = set(self.mask[start_node])
        while (len(stack)>0):
            vertex=stack.pop()
            nodes=self.G[vertex]
            for w in nodes:
                if w not in seen:
                    stack.append(w)
                    seen.add(w)
            if vertex in mask_list:
                pass
            else:
                if vertex != start_node:
                    walks.append(vertex)
            if len(walks) >= walks_num:
                break
        return walks
    
    def intermediate(self):
        candidate = defaultdict(list)
        for node in self.G.nodes():
            walk = self.dfs(node, self.walks_num)
            candidate[node].extend(walk)
        return candidate


def candidate_choose(nx_Graph, mask, walks_num):
    G = Personalized(nx_Graph, mask, walks_num) 
    candidates = G.intermediate() 
    return candidates
