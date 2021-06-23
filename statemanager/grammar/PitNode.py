from typing import Callable
from base64 import b64decode

class NodeBase:
    """
    Base class for a Node in Grammar Input Tree
    """
    def __init__(self, children: list, name: str):
        # Store the children for a node of a Grammar tree
        self._children = children
        self.name = name
    
    @property
    def children(self):
        return self._children
    
    def add_child(self, child: NodeBase):
        self._children.append(child)
    
    def find_child(name: str):
        for child in self.children:
            if child.name == name:
                return child
            

    @property
    def value():
        pass


class DataModelNode(NodeBase):

    def __init__(self, children: list, name: str):
        super().__init__(children, name)
        self.name = name
    
    @property
    def value():
        value = ''

        for child in children:
            value += child.value
        return value

class StringNode(NodeBase):

    def __init__(self, children: list, name: str, value: str, mutable: bool, constraint: str):
        super().__init__(children, name)
        self._mutable = mutable
        self._value = value
        self._constraint = constraint

    @property
    def value(self):
        return self._value

    def constraint(self, mutated_string):
        # TODO: Decide on a good way to implement
        pass

class NumberNode(NodeBase):

    def __init__(self, children: list, name: str, value: int, mutable: bool, constraint: str):
        super().__init__(children)
        self._value = value
        self._constraint = constraint
        self._mutable = mutable
    
    @property
    def value(self):
        return int(self._value)

    def constraint(self, mutated_number):
        # TODO
        pass


class BlobNode(NodeBase):
    
    def __init__(self, children: str, name: str, value: str):
        super().__init__(children, name)

        self._value = b64decode(value)

class BlockNode(NodeBase):

    def __init__(self, children: list, name: str):
        super().__init__(children)
