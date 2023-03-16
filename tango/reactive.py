from . import debug, critical

from tango.core import (AbstractInstruction, TransmitInstruction,
    ReceiveInstruction, DelayInstruction, AbstractInput, EmptyInput,
    BaseDecorator, AbstractState, BaseMutator, BaseInputGenerator,
    CountProfiler, ValueProfiler)
from tango.havoc import havoc_handlers, RAND, MUT_HAVOC_STACK_POW2

from typing import Sequence, Iterable
from random import Random
from enum import Enum
from copy import deepcopy
from math import exp
from struct import pack
import time
import json
import os

__all__ = [
    'ReactiveHavocMutator', 'ReactiveInputGenerator',
    'StatelessReactiveInputGenerator'
]

class ReactiveHavocMutator(BaseMutator):
    class RandomOperation(Enum):
        DELETE = 0
        PUSHORDER = 1
        POPORDER = 2
        REPEAT = 3
        CREATE = 4
        MUTATE = 5

    class RandomInstruction(Enum):
        TRANSMIT = 0
        RECEIVE = 1
        DELAY = 2

    def __init__(self, havoc_actions: Iterable, **kwargs):
        super().__init__(**kwargs)
        self._actions = havoc_actions
        self._actions_taken = False

    def _iter_helper(self, orig):
        with self.entropy_ctx as entropy:
            i = -1
            reorder_buffer = []
            for i, instruction in enumerate(orig()):
                new_instruction = deepcopy(instruction)
                seq = self._mutate(new_instruction, reorder_buffer, entropy)
                yield from seq
            if i == -1:
                yield from self._mutate(None, reorder_buffer, entropy)

            # finally, we flush the reorder buffer
            yield from entropy.sample(reorder_buffer, k=len(reorder_buffer))
            reorder_buffer.clear()

    def ___iter___(self, input, orig):
        self._actions_taken = False
        for instruction in self._iter_helper(orig):
            yield instruction
            self._actions_taken = False

    def ___repr___(self, input, orig):
        return f'HavocMutatedInput:0x{input.id:08X} (0x{self._input_id:08X})'

    def _apply_actions(self, data, entropy):
        for func in self._actions:
            # this copies the data buffer into a new array
            data = bytearray(data)
            data = func(data, entropy)
        self._actions_taken = True
        return data

    def _mutate(self, instruction: AbstractInstruction, reorder_buffer: Sequence, entropy: Random) -> Sequence[AbstractInstruction]:
        if instruction is not None:
            low = 0
            for _ in range(entropy.randint(3, 7)):
                if low > 5:
                    return
                oper = self.RandomOperation(entropy.randint(low, 5))
                low = oper.value + 1
                if oper == self.RandomOperation.DELETE:
                    return
                elif oper == self.RandomOperation.PUSHORDER:
                    reorder_buffer.append(instruction)
                    return
                elif oper == self.RandomOperation.POPORDER:
                    yield instruction
                    if reorder_buffer:
                        yield from self._mutate(reorder_buffer.pop(), reorder_buffer, entropy)
                elif oper == self.RandomOperation.REPEAT:
                    yield from (instruction for _ in range(2))
                elif oper == self.RandomOperation.CREATE:
                    buffer = entropy.randbytes(entropy.randint(1, 256))
                    yield TransmitInstruction(buffer)
                elif oper == self.RandomOperation.MUTATE:
                    if isinstance(instruction, TransmitInstruction):
                        instruction._data = self._apply_actions(instruction._data, entropy)
                    else:
                        # no mutations on other instruction types for now
                        pass
                    yield instruction
        else:
            buffer = entropy.randbytes(entropy.randint(1, 256))
            yield TransmitInstruction(buffer)

HAVOC_MIN_WEIGHT = 1e-3

class ReactiveInputGenerator(BaseInputGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seen_transitions = set()
        self._state_model = dict()
        # self._model_history = dict()

        self._log_counter = 0
        self._log_path = os.path.join(self._work_dir, "model_history.bin")
        self._log_buffer = bytearray()
        self._pack_log_header()

        with open(self._log_path, "w"):
            # clear the log file
            pass

    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super().match_config(config) and \
            config['generator'].get('type') == 'reactive'

    def generate(self, state: AbstractState) -> AbstractInput:
        out_edges = list(state.out_edges)
        if out_edges:
            _, dst, data = self._entropy.choice(out_edges)
            candidate = data['minimized']
        else:
            in_edges = list(state.in_edges)
            if in_edges:
                _, dst, data = self._entropy.choice(in_edges)
                candidate = data['minimized']
            elif self.seeds:
                candidate = self._entropy.choice(self.seeds)
            else:
                candidate = EmptyInput()

        if (model := self._state_model.get(state)) is None:
            model = self._init_state_model(state)

        havoc_actions = self._entropy.choices(havoc_handlers,
            weights=map(lambda t: model['actions'][t][1], havoc_handlers), # we use probabilities as weights
            k=RAND(MUT_HAVOC_STACK_POW2, self._entropy) + 1
        )

        return ReactiveHavocMutator(havoc_actions, entropy=self._entropy)(candidate)

    def update_state(self, state: AbstractState, /, *, input: AbstractInput,
            orig_input: AbstractInput, exc: Exception=None, **kwargs):
        if state not in self._state_model:
            self._init_state_model(state)

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, /, *,
            state_changed: bool, orig_input: AbstractInput, exc: Exception=None,
            **kwargs):
        if state_changed:
            assert source != destination, "No state change detected!"
            if (t := (source, destination)) in self._seen_transitions:
                # TODO handle case where mutator results in a seen transition
                return
            else:
                self._seen_transitions.add(t)

        if not orig_input.decorated:
            # input was not generated by us (e.g. seed input), ignore
            CountProfiler('undecorated_inputs')(1)
            return

        mut = orig_input.search_decorator_stack(lambda d: isinstance(d, ReactiveHavocMutator), max_depth=1)

        ancestors = set()
        src_model = self._state_model[source]
        normalized_reward = self._calculate_reward(source, destination)
        if mut._actions_taken:
            self._update_weights(src_model['actions'], mut._actions, normalized_reward)

        if state_changed:
            # initialize the destination model
            dst_model = self._init_state_model(destination, copy_from=source)

            # update feature counts
            fcount = self._count_features(source, destination)
            src_model['features'] += fcount
            state = source
            while state is not None and state not in ancestors:
                ancestors.add(state)
                self._state_model[state]['cum_features'] += fcount
                # FIXME verify that state._parent == state.[ancestor]
                state = state._parent

        if state_changed or mut._actions_taken:
            self._log_model(source, destination, *ancestors)

        return normalized_reward, mut._actions_taken, mut._actions

    def _init_state_model(self, state: AbstractState, copy_from: AbstractState=None):
        if copy_from is None:
            # initialize model to map of actions to tuple(weight, prob)
            self._state_model[state] = {
                'actions': dict.fromkeys(havoc_handlers, (1., 1. / len(havoc_handlers))),
                'features': 0,
                'cum_features': 0
            }
        else:
            self._state_model[state] = {
                'actions': self._state_model[copy_from]['actions'].copy(),
                'features': 0,
                'cum_features': 0
            }
        return self._state_model[state]

    def _log_model(self, *states: list[AbstractState]):
        now = time.time()
        models = dict()
        for state in states:
            models[state] = self._state_model[state]
        self._pack_log_entry(now, models)

        self._log_counter += 1
        ValueProfiler('log_counter')(self._log_counter)
        if self._log_counter & 0xff == 0:
            self._log_counter = 0

            with open(self._log_path, 'ab') as f:
                self._flush_log(f)

    def _pack_into_log(self, fmt, *args):
        p = pack(fmt, *args)
        self._log_buffer.extend(p)
        return self._log_buffer

    def _pack_log_header(self):
        names = [x.__name__.encode() for x in havoc_handlers]
        self._pack_into_log('B', len(names))
        for handler in names:
            self._pack_into_log(f'B{len(handler)}s', len(handler), handler)

    def _pack_log_entry(self, timestamp, models):
        self._pack_into_log('d', timestamp)
        self._pack_into_log('I', len(models))
        for label, model in models.items():
            self._pack_log_model(repr(label).encode(), model)

    def _pack_log_model(self, label, model):
        self._pack_into_log(f'B{len(label)}s', len(label), label)
        self._pack_into_log('II', model['features'], model['cum_features'])
        for handler in havoc_handlers:
            self._pack_into_log('ff', *model['actions'][handler])

    def _flush_log(self, file):
        file.write(self._log_buffer)
        self._log_buffer.clear()

    @classmethod
    def _calculate_reward(cls, source: AbstractState, destination: AbstractState, amplifier: float=10) -> float:
        bound = revenue = 0.
        if source != destination:
            bound += len(destination._feature_mask) * 8
            revenue += destination._feature_count
        if (local_state := destination.tracker._local_state) is not None:
            # now we rely on local state changes
            bound += len(local_state._feature_mask) * 8
            revenue += local_state._feature_count / amplifier
        if bound > 0.:
            revenue /= bound
        # WARN reward in exp3 must belong to [0, 1)
        reward = 1 - exp(-revenue / cls._estimate_cost(source) * amplifier * 5) # exp(-5) ~= 0
        return reward

    @staticmethod
    def _estimate_cost(state: AbstractState) -> float:
        if state is None:
            return .0
        bound = len(state._feature_mask) * 8
        cost = 0
        visited = set()
        while state is not None and state not in visited:
            visited.add(state)
            cost += state._feature_count
            # FIXME check other FIXME
            state = state._parent
        if cost == 0:
            # we cannot get an estimate, so we return highest cost
            return 1
        else:
            return cost / bound

    @staticmethod
    def _count_features(source: AbstractState, destination: AbstractState) -> int:
        return int(destination._feature_count)

    @staticmethod
    def _update_weights(model: dict, actions_taken: list, normalized_reward: float, gamma: float=0.1):
        amortized_reward = normalized_reward / len(actions_taken)
        # first we recalculate the weights
        for action in actions_taken:
            w_t, p_t = model[action]
            try:
                w_tp = w_t * exp(amortized_reward / p_t * gamma)
            except OverflowError:
                import ipdb; ipdb.set_trace()
                # in case of overflow, we just assign some higher weight value
                w_tp = w_t * 10
            # we keep p_t since actions may be repeated
            model[action] = (w_tp, p_t)

        # then we normalize weights and compute probabilities
        max_weight = max(map(lambda t: t[0], model.values()))
        model.update({
                # we make sure that no normalized weight goes to zero
                a: (max(t[0], HAVOC_MIN_WEIGHT * max_weight), None) # probability will be calculated later
                for a, t in model.items()
            })
        sum_weights = sum(map(lambda t: t[0], model.values()))
        model.update({
                a: (
                    t[0] / max_weight, # w_tp
                    (1 - gamma) * (t[0] / sum_weights) + gamma / len(model) # p_tp
                )
                for a, t in model.items()
            })

class StatelessReactiveInputGenerator(ReactiveInputGenerator):
    @classmethod
    def match_config(cls, config: dict) -> bool:
        return super(BaseInputGenerator, cls).match_config(config) and \
            config['generator'].get('type') == 'reactless'

    def update_state(self, state: AbstractState, /, *, input: AbstractInput,
            orig_input: AbstractInput, exc: Exception=None, **kwargs):
        state = state.tracker._entry_state
        super().update_state(state, input, orig_input=orig_input, exc=exc, **kwargs)

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, /, *,
            state_changed: bool, orig_input: AbstractInput, exc: Exception=None,
            **kwargs):
        source = source.tracker._entry_state
        if state_changed:
            assert source != destination, "No state change detected!"
            if (t := (source, destination)) in self._seen_transitions:
                # TODO handle case where mutator results in a seen transition
                return
            else:
                self._seen_transitions.add(t)

        if not orig_input.decorated:
            # input was not generated by us (e.g. seed input), ignore
            CountProfiler('undecorated_inputs')(1)
            return

        mut = orig_input.search_decorator_stack(lambda d: isinstance(d, ReactiveHavocMutator), max_depth=1)

        src_model = self._state_model[source]
        normalized_reward = self._calculate_reward(source, destination)
        if mut._actions_taken:
            self._update_weights(src_model['actions'], mut._actions, normalized_reward)

        if state_changed:
            # update feature counts
            fcount = self._count_features(source, destination)
            src_model['features'] += fcount

        if state_changed or mut._actions_taken:
            self._log_model(source)
