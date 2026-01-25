import json
from collections import defaultdict, deque
from utils.treeunit import NodeUnit, AttributeUnit




# ---------- Main extraction ----------
class RawData():
    def __init__(self, data_dir:str):
        definedConcept_list_raw:list[str] = self.load_json(data_dir + "node_concepts_small.json")
        attribute_list_raw:list[str] = self.load_json(data_dir + "attributes_small.json")
        upper_dict_raw:dict[str,str] = self.load_json(data_dir + "upper_concept_small.json")
        attributes_dict_raw:dict[str,list[str]] = self.load_json(data_dir + "node_attributes_small.json")
        root_concept:str = self.load_json(data_dir + "root_concept.json")
        edges = self.load_json(data_dir + "edges_small.json")

        definedConcept_list = [NodeUnit(concept) for concept in definedConcept_list_raw]
        attribute_list = [AttributeUnit(attr) for attr in attribute_list_raw]
        defiUnit_dict = {concept.essence: concept for concept in definedConcept_list}
        attrUnit_dict = {attr.essence: attr for attr in attribute_list}
        unitDict:dict[str, dict[str, list[NodeUnit|AttributeUnit]]] = {'attr':attrUnit_dict, 'defi': defiUnit_dict}

        
        upper_dict = {defiUnit_dict[key]: defiUnit_dict[value] for key, value in upper_dict_raw.items()}
        attributes_dict = {defiUnit_dict[key]: [attrUnit_dict[attr] for attr in value] for key, value in attributes_dict_raw.items()}
        origin = defiUnit_dict[root_concept]

        self.defiList = definedConcept_list
        self.attrList = attribute_list
        self.defiUnitDict = defiUnit_dict
        self.attrUnitDict = attrUnit_dict
        self.unitDict   = unitDict
        self.upperDict = upper_dict
        self.attrDict = attributes_dict
        self.origin = origin
        self.edges = edges  

        self.stuffNodeTree()
        self.set_tree_depth()

    # ---------- Refinement functions ----------
    def stuffNodeTree(self) -> dict[str, NodeUnit]: 
        for concept in self.defiList:
            if concept is None:
                continue
            father = self.upperDict.get(concept, None)
            attributes = self.attrDict.get(concept, set())
            concept.setFather(father)
            concept.setAttributes(attributes)
            if father is not None:
                father.addChild(concept)
            for attr in attributes:
                attr.addChild(concept)
        
    def set_tree_depth(self):
        self.origin.setDepth(0)
        # BFS to set depths
        queue = deque([self.origin])
        sorted_concepts_list:list[NodeUnit] = [self.defiUnitDict[None]]
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
            sorted_concepts_list.append(current_node)
            for child in current_node.children:
                if child not in visited:
                    child.setDepth(current_node.depth + 1)
                    queue.append(child)
        self.depth_range_record = depth_range_record
        self.defiList= sorted_concepts_list

    @staticmethod
    def load_json(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except:
            raise Exception(f"Error loading JSON file: {path}")
        
    @staticmethod
    def save_json(path: str, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

# ---------- Example usage ----------
if __name__ == "__main__":
    data_dir = "source/data/"
    rawdata_instance = RawData(data_dir)
    
    a = 0
