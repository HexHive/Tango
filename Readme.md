# Refuzz

Network Protocol fuzzer - State based.

## Structure of the fuzzer

1. **Fuzzer Config** - This class holds the configuration content for the fuzzer. The user provides a JSON of configurations for the fuzzer (target details, connection type, snapshot/replay, StateMachine/Grammar) etc.

2. **Input** - `InputBase` class is the basic input class for the fuzzer. An input is thought of as a path between two states of the server. Thus, one input can contain multiple *interactions* with the server.

3. **Interaction** - Interaction is a single *packet* exchange between fuzzer and target. Can be of multiple types, like `RecieveInteraction` `TransmitInteraction`, `DelayInteraction`, `EOFInteraction`. <u>Currently mutation is defined in this class</u> i.e each interaction with target is mutated with this method.

4. **Channel** - A channel is a connection between the fuzzer and the target. `TCPChannel`, `UDPChannel` are two channels. Every channel also has the `tx_callback` and `rx_callback` which are used for setting up callback functions on the send and recv calls.

5. **ChannelFactory** - A creator class for the channel. Simply creates the reuired channel.

6. **Mutator** - Mutator class

### Classes and parts related to State Machine

1. **Loader** - Work is to load/reset the StateMachine and the target to a specific state we want. Can be uesd when a roadblock is hit on a state.
The input is also executed by the Loader class only.

2. **StateMachine** - Directed Bigraph with circular edge support. The nodes represent the states and the edges represent the transitions between these states. State Machine is common between Grammar Machine and Coverage state Machine

3. **State** - Each state is a `StateBase` class derivative. Has to be hashable (for it to be stored in graph) and `__eq__` method to be defined for states to be equal.

4. **Transition** - Represents Transition between states. Each transition contains an `Input` member.

5. **StateManager** - The controlling class for the State Based stuff. `StateManager` has the `StateMachine, Loader, StateTracker` members. It is used to update the `StateTracker` and the `StateMachine` and uses the loader to load a state. 

6. **StateTraker** - StateTracker is used to hold info about the current status of state machine like current state and used to perform state updation.