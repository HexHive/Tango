from input         import (InputBase,
                          PreparedInput)
from fuzzer        import FuzzerConfig
import os

from xml.etree     import ElementTree 

from statemanager  import (
    NodeBase,
    DataModelNode,
    StringNode,
    NumberNode,
    BlockNode
)

class FuzzerSession:
    """
    This class initializes and tracks the global state of the fuzzer.
    """
    def __init__(self, config: FuzzerConfig):
        """
        FuzzerSession initializer.
        Takes the fuzzer configuration, initial seeds, and other parameters as
        arguments.
        Initializes the target and establishes a communication channel.

        :param      config: The fuzzer configuration object
        :type       config: FuzzerConfig
        """

        self._loader = config.loader
        self._sman = config.state_manager
        self._seed_dir = config.seed_dir
        self._ch_env = config.ch_env # we may need this for parsing PCAP
        self._pit_file = config.pit_file
        self._grammar_tree_base = NodeBase(None, "start_symbol")

        self._load_seeds()

    def _load_seeds(self):
        """
        Loops over the initial set of seeds to populate the state machine with
        known states.
        """
        if self._seed_dir is None or not os.path.isdir(self._seed_dir):
            return

        seeds = []
        for root, _, files in os.walk(self._seed_dir):
            seeds.extend(os.path.join(root, file) for file in files)

        for seed in seeds:
            # parse seed to PreparedInput
            input = PCAPInput(seed, self._ch_env)

            # restore to entry state
            self._sman.reset_state()
            # feed input to target and populate state machine
            self._loader.execute_input(input, self._loader.channel, self._sman)

    def _loop(self):
        # FIXME is there ever a proper terminating condition for fuzzing?
        while True:
            cur_state = self._sman.state_tracker.current_state
            input = cur_state.get_escaper()
            self._loader.execute_input(input, self._loader.channel, self._sman)

    def start(self):
        # reset state after the seed initialization stage
        self._sman.reset_state()

        # launch fuzzing loop
        self._loop()

        # TODO anything else?
        pass


    def load_pit(self):

        tree = ElementTree.parse(self._pit_file)
        root = tree.getroot()

        for element in list(root):

            if element.tag != 'DataModel':
                continue

            self._grammar_tree_base.add_child(self.parse_single_xml_element(element, tree))

        for element in list(root):

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
                    # Let's create the input now

                    datamodel_name = transition.find('DataModel').attrib('ref', '')
                    datamodel = root.find(f'./DataModel[@name="{datamodel_name}"]')
                    input = GrammarInput(name, None)
                    data = input.parse_element(datamodel)
                    input.add_interaction(data, transition.attrib.get('type'))
                    state_transition = GrammarTransition(source_state, dest_state, input)

                    state_machine.add_transition(source_state, dest_state, state_transition)

    def parse_single_xml_element(self, element: ElementTree.Element, tree: ElementTree):

        if element.tag == 'DataModel':
            node = DataModelNode(list(), element.attrib.get('name'))
        
            for child_element in list(element):
                node.add_child(self.parse_single_xml_element(child, tree))
        elif element.tag == 'String':

            node = StringNode(
                list(), 
                element.attrib.get('name'), 
                element.attrib.get('value'), 
                element.attrib.get('mutable', True), 
                element.attrib.get('constraint', None)
            )

        elif element.tag == 'Number':
            node = NumberNode(
                list(), 
                element.attrib.get('name'), 
                int(element.attrib.get('value')), 
                element.attrib.get('mutable', True), 
                element.attrib.get('constraint', None)
            )
        elif element.tag == 'Block':
            node = BlockNode(list(), element.attrib.get('name'))

            for child_element in list(element):
                node.add_child(self.parse_single_xml_element(child_element, tree))
        
        return node if (node is not None) else None 