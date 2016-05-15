import pydot

def getLabel(key):
    if isinstance(key, tuple):
        if key[0] == -sys.maxint:
            label = 'x<%.2f' % key[1]
        elif key[1] == sys.maxint:
            label = 'x>=%.2f' % key[0]
        else:
            label = '%.2f<=x<%.2f' % key
    else:
        return key
def exploreNode(node, graph, i):
    j = i
    for key, child in node['children'].iteritems():
        label = getLabel(key)
        j += 1
        if not child['children']:
            probs = child['probs']
            mx, name = (0, None)
            for k, v in probs.iteritems():
                if v > mx:
                    mx = v
                    name = k
            node = pydot.Node(str(j), label=str(name))
            graph.add_node(node)
            edge = pydot.Edge(str(i), str(j), label=label)
            graph.add_edge(edge)
        else:
            node = pydot.Node(str(j), label=child['feature'][1])
            graph.add_node(node)
            edge = pydot.Edge(str(i), str(j), label=label)
            graph.add_edge(edge)
            j = exploreNode(child, graph, j)
    return j

def printTree(tree):
    graph = pydot.Dot(graph_type='graph')
    node = pydot.Node('0', label=tree['feature'][1])
    graph.add_node(node)
    j = exploreNode(tree, graph, 0)
    graph.write_png('DT.png')

