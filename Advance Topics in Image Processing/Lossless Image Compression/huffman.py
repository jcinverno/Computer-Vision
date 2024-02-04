# from multiprocessing import Queue
from queue import PriorityQueue
from collections import Counter


def huffman(symbol_list):
    """
    This function generates the huffman tree for the given input.
    The input is a list of "symbols".
    """
    # figure out the frequency of each symbol
    counts = Counter(symbol_list).most_common()

    total = len(symbol_list)
    if len(counts) < 2:
        # 0 or 1 unique symbols, so no sense in performing huffman coding
        return

    queue = PriorityQueue()
    for (val, count) in counts:
        queue.put((count, val))

    # Create the huffman tree
    largest_node_count = 0
    while total != largest_node_count:
        node1 = queue.get(False)
        node2 = queue.get(False)

        new_count = node1[0] + node2[0]
        largest_node_count = new_count if new_count > largest_node_count else largest_node_count
        queue.put((new_count, (node1, node2)))
    huffman_tree_root = queue.get(False)

    # generate the symbol to huffman code mapping
    lookup_table = huffman_tree_to_table(huffman_tree_root, "", {})
    return lookup_table


def huffman_tree_to_table(root, prefix, lookup_table):
    """Converts the Huffman tree rooted at "root" to a lookup table"""
    if type(root[1]) != tuple:
        # leaf node
        lookup_table[root[1]] = prefix
    else:
        huffman_tree_to_table(root[1][0], prefix + "0", lookup_table)
        huffman_tree_to_table(root[1][1], prefix + "1", lookup_table)

    return lookup_table


def text_to_huffman_code(input_text):
    """Helper function to convert an input string into its huffman symbol table"""
    return huffman([c for c in input_text])
