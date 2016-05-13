class DecisionTreeNode(object):
    def __init__(self, indices, feature, children):
        self.indices = indices
        self.feature = feature
        self.children = children
        self.klass = None
