from collections import deque
from collections import defaultdict
from nltk.corpus import wordnet as wn
import nltk

def collect_hyponyms_unique_parent_first_sense(s: str, k: int, *, pos: str = 'n'):
    """
    输入: 字符串 s, 深度 k
    输出:
      - nodelist: [s, ...] 以及从 WordNet 中向下 k 层收集到的所有下位词（去重，按首次出现顺序）
      - upperdict: {child_word: parent_word} 唯一父节点；只保留第一次遇到的父子关系

    规则:
      1) 多义词只保留第一个意思：wn.synsets(s, pos)[0]
      2) 节点命名：每个 synset 取 ss.lemmas()[0].name()
      3) depth=0 为根；向下走 k 层（k=0 只返回 [s], {}）
    """
    if k < 0:
        raise ValueError("k must be >= 0")

    synsets = wn.synsets(s, pos=pos)
    if not synsets:
        return [s], {}

    root_ss = synsets[0]

    nodelist = []
    nset = set()
    upperdict = {}  # child -> unique parent (first seen)

    def add_node(w: str):
        if w not in nset:
            nset.add(w)
            nodelist.append(w)

    # 根节点用输入字符串本身
    add_node(s)

    q = deque()
    q.append((root_ss, 0))

    visited_synsets = set([root_ss.name()])

    while q:
        cur_ss, depth = q.popleft()
        if depth >= k:
            continue

        # 当前 synset 对应的“父词”
        # depth=0 时，父词强制用输入词 s（而不是 root_ss 的 lemma）
        parent_word = s if depth == 0 else cur_ss.lemmas()[0].name()
        add_node(parent_word)

        for child_ss in cur_ss.hyponyms():
            child_word = child_ss.lemmas()[0].name()
            add_node(child_word)

            # 只保留第一次遇到的父子关系（唯一父节点）
            if child_word != parent_word and child_word not in upperdict:
                upperdict[child_word] = parent_word

            # BFS 继续展开（按 synset 去重，避免循环）
            sid = child_ss.name()
            if sid not in visited_synsets:
                visited_synsets.add(sid)
                q.append((child_ss, depth + 1))
    return nodelist, upperdict



def print_tree(root: str, upperdict: dict[str, str], max_depth: int | None = None):
    children = defaultdict(list)
    for child, parent in upperdict.items():
        children[parent].append(child)
    # 排序让输出更稳定（也可以去掉）
    for p in children:
        children[p].sort()

    def dfs(node: str, depth: int):
        if max_depth is not None and depth > max_depth:
            return
        prefix = "  " * depth + ("└─ " if depth > 0 else "")
        print(prefix + node)
        for c in children.get(node, []):
            dfs(c, depth + 1)
    dfs(root, 0)


#示例：
nodelist, upperdict = collect_hyponyms_unique_parent_first_sense("animal", 2, pos='n')
print(len(nodelist), nodelist[:20])
print(list(upperdict.items())[:10])
print_tree("animal", upperdict, max_depth=2)
a=0