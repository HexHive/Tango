from input import PreparedInput
from xml.etree import ElementTree

from interaction import TransmitInteraction, ReceiveInteraction

class GrammarInput(PreparedInput):

    def __init__(self, name: str, parent: ElementTree.Element, interaction_type = ''):
        super().__init__(self)

        # DataModel has a name attribute which is reqired, so can be used here.
        self.name = name
        self.parent = parent

        if parent is not None:
            data = self.parse_element(parent)


    def parse_element(self, element: ElementTree.Element):
        data = ''
        if element.tag in ['String', 'Number']:
            print(f"Yess {element.attrib.get('value', '')}")
            data += element.attrib.get('value', '')
        elif element.tag == 'Blob':
            bytestring = element.attrib.get('value', '')
            if element.attrib.get('valueType') == 'hex':
                bytestring = ''.join(
                    [chr(int(k, 16)) for k in bytestring.split(' ')]
                )
            data += bytestring

        for child in list(element):
            data += self.parse_element(child)

        return data

    def add_interaction(self, data: str, interaction_type: str):
        if interaction_type == 'output':
            interaction = TransmitInteraction(data)
            self._interactions.append(interaction)
        elif interaction_type == 'input':
            interaction = ReceiveInteraction()
            interaction._expected = data
            self._interactions.append(interaction)

        return interaction