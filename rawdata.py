from collections import deque, defaultdict

from utils.unit import NodeUnit, AttrUnit
from utils import fileTool

def stuffNodeTree(
        nodeList_raw:list[str],
        attrList_raw:list[str], 
        upperDict_raw:dict[str,str],
        attrDict_raw:dict[str,list[str]],
        )-> tuple[  list[NodeUnit], 
                    list[AttrUnit], 
                    dict[str, NodeUnit], 
                    dict[str, AttrUnit], 
                    dict[NodeUnit, NodeUnit], 
                    dict[NodeUnit, set[AttrUnit]]]: 
    nodeList = [NodeUnit(node) for node in nodeList_raw]
    attrList = [AttrUnit(attr) for attr in attrList_raw]
    nodeUnitDict = {node.essence: node for node in nodeList}
    attrUnitDict = {attr.essence: attr for attr in attrList}
    upperDict = {nodeUnitDict[key]: nodeUnitDict[value] for key, value in upperDict_raw.items()}
    attrDict = {nodeUnitDict[key]: {attrUnitDict[attr] for attr in value} for key, value in attrDict_raw.items()}
    for concept in nodeList:
        father = upperDict.get(concept, None)
        attributes = attrDict.get(concept, set())
        concept.setFather(father)
        concept.setAttributes(attributes)
        if father is not None:
            father.addChild(concept)
        for attr in attributes:
            attr.addChild(concept)
    return nodeList, attrList, nodeUnitDict, attrUnitDict, upperDict, attrDict

def set_tree_depth(
        nodeUnitDict:dict[str, NodeUnit], 
        origname:str)->tuple[NodeUnit, list[NodeUnit], dict]:
    origin = nodeUnitDict[origname]
    origin.setDepth(0)
    # BFS to set depths
    queue = deque([origin])
    sorted_conceptsList:list[NodeUnit] = []
    visited = set()
    current_depth = 0
    count = 0
    depth_range_record = {0: [0, -1]}  # depth: (start_index, end_index)
    while queue:
        current_node = queue.popleft()
        visited.add(current_node)
        if current_node.depth > current_depth:
            depth_range_record[current_depth][1] = depth_range_record[current_depth][0] + count
            depth_range_record[current_node.depth] = [depth_range_record[current_depth][1], -1]
            current_depth = current_node.depth
            count = 0
        count += 1
        sorted_conceptsList.append(current_node)
        for child in current_node.children:
            if child not in visited:
                child.setDepth(current_node.depth + 1)
                queue.append(child)
    depth_range_record[current_depth][1] = depth_range_record[current_depth][0] + count
    sorted_conceptsList.append(nodeUnitDict[None])
    depth_range_record = {key: tuple(value) for key, value in depth_range_record.items()}
    return origin, sorted_conceptsList, depth_range_record

def create_homoDict( origin:NodeUnit):
    homoDict:dict[NodeUnit, set[AttrUnit]] = {None: set()}
    queue = deque([origin])
    while queue:
        current_node = queue.popleft()
        queue.extend(current_node.children)
        homoDict[current_node] = homoDict[current_node.father] | current_node.attributes
    del homoDict[None]
    return homoDict

def create_ancestorDict(origin:NodeUnit):
    ancestorDict:dict[NodeUnit, set[NodeUnit]] = {None: set()}
    queue = deque([origin])
    while queue:
        current_node = queue.popleft()
        queue.extend(current_node.children)
        ancestorDict[current_node] = ancestorDict[current_node.father] | {current_node}
    del ancestorDict[None]
    return ancestorDict


def create_negtDict(attrList:list[AttributeError], origin:NodeUnit):
    negtDict:dict[NodeUnit, set[AttrUnit]] = {None: set(attrList)}
    queue = deque([origin])
    while queue:
        current_node = queue.popleft()
        queue.extend(current_node.children)
        negtDict[current_node] = negtDict[current_node.father] - current_node.attributes
    del negtDict[None]
    return negtDict



# ---------- Main extraction ----------
class RawData():
    def __init__(self, data_dir:str):
        self.root_concept:str = fileTool.load_json(data_dir + "root_concept.json")
        self.nodeList_raw:list[str] = fileTool.load_json(data_dir + "node_concepts.json")
        self.attrList_raw:list[str] = fileTool.load_json(data_dir + "attributes.json")
        self.upperDict_raw:dict[str,str] = fileTool.load_json(data_dir + "upper_concept.json")
        self.attrDict_raw:dict[str,list[str]] = fileTool.load_json(data_dir + "node_attributes.json")
        self.nodeList_raw.append(None)
        self.attrList_raw.append(None)
        nodeList, attrList, \
            nodeUnitDict, attrUnitDict, \
                upperDict, attrDict = stuffNodeTree(self.nodeList_raw, self.attrList_raw, self.upperDict_raw, self.attrDict_raw)
        unitDict:dict[str, dict[str, list[NodeUnit|AttrUnit]]] \
            = {'attr':attrUnitDict, 'node': nodeUnitDict}
        origin, nodeList, depth_range_record = set_tree_depth(nodeUnitDict, self.root_concept)
        self.nodeList = nodeList[:-1]
        self.attrList = attrList[:-1]
        self.nodeAllList = nodeList
        self.attrAllList = attrList
        self.nodeUnitDict = nodeUnitDict
        self.attrUnitDict = attrUnitDict
        self.unitDict   = unitDict
        self.upperDict = upperDict
        self.attrDict = attrDict
        self.origin = origin
        self.depth_range_record = depth_range_record

        self.homoDict = create_homoDict(self.origin)
        self.anceDict = create_ancestorDict(self.origin)
        self.negtDict = create_negtDict(self.attrList, self.origin)
        a = 0
    
    def print_tree(self, max_depth: int | None = None):
        def dfs(node: NodeUnit, depth: int):
            if max_depth is not None and depth > max_depth:
                return
            prefix = "  " * depth + ("└─ " if depth > 0 else "") + f" {depth} "
            print(prefix + node.name)
            for c in node.children:
                dfs(c, depth + 1)
        dfs(self.origin, 0)


# ---------- Example usage ----------
if __name__ == "__main__":
    data_dir = "./data_full/"
    rawdata_instance = RawData(data_dir)
    rawdata_instance.print_tree()
    a = 0
