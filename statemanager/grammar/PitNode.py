from typing import Callable


class NodeBase:
    """
    Base class for a Node in Grammar Input Tree
    """
    def __init__(self, children: list):
        # Store the children for a node of a Grammar tree
        self._children = children
    
    @property
    def children():
        return self._children
    
    @property
    def value():
        pass


class DataModelNode(NodeBase):

    def __init__(self, children: list, name: str):
        super().__init__(children)
        self.name = name
    
    @property
    def value():
        value = ''

        for child in children:
            value += child.value
        return value

class StringNode(NodeBase):

    def __init__(self, children: list, value: str, mutable: bool, constraint: str):
        super().__init__(children)
        self._mutable = mutable
        self._value = value
        self._constraint = constraint

    @property
    def value(self):
        return self._value

    def constraint(self, mutated_):
        # TODO: Decide on a good way to implement
        pass

class NumberNode(NodeBase):

    def __init__(self, children: list, value: int, mutable: bool, constraint: str):
        super().__init__(children)
        self._value = value
        self._constraint = constraint
        self._mutable = mutable
    
    @property
    def value(self):
        return int(self._value)

    def constraint(self):
        # TODO
        pass


class BlockNode(NodeBase):

    def __init__(self, children):
        super().__init__(children)
