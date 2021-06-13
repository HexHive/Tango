from statemanager import StateBase
from xml.etree import ElementTree
from statemanager import (
    StateMachine, 
    GrammarTransition)

class GrammarState(StateBase):
    """
    XML File has a state attribute
    <State>
        <Action></Action>
    </State>

    state name - Adds the ability to be a hashable thing.
    Every out transition/The set of reachable states can be stored.
    """
    
    def __init__(self, name: str):
        # TODO populate with grammar information
        self.name = name
        self.reachable_states = list()

    def __eq__(self, other: GrammarState):
        # TODO: Since grammar generated state machine is "static", we can hold equality of two states on 
        # the basis of their names only, Reachable states arent needed either
        if (self.name == other.name) and (self.reachable_states == other.reachable_states):
            return True

def analyze_xml(fname: str):

    tree = ElementTree.parse(fname)
    root = tree.getroot()
    for element in list(elem):
        if element.tag == 'StateModel':
            statemodel = element
    
    # We have the statemodel now, lets start enumerating the states.
    entry_state = statemodel.find('./State[@name="Initial]"')
    state_machine = StateMachine(GrammarState(entry_state.attrib.get('name', None)), None)

    for state in list(statemodel):
        source_state = GrammarState(state.attrib.get('name'))
        # TODO: Add this grammar state's transitions to the transitions list.
        for transition in list(state):
            dest_state = GrammarState(transition.attrib.get('finalState'))
            state_transition = GrammarTransition(source_state, dest_state, None)

            state_machine.add_transition(source_state, dest_state, state_transition)
