from . import debug, critical

from tango.core import (AbstractInstruction, TransmitInstruction,
    ReceiveInstruction, DelayInstruction, AbstractInput,
    BaseDecorator, AbstractState, BaseInputGenerator,
    CountProfiler, ValueProfiler)
from tango.havoc import havoc_handlers, HavocMutator, MutatedTransmitInstruction

from typing import Sequence, Iterable, Optional
from random import Random
from enum import Enum
from copy import deepcopy
from math import exp
from struct import pack
import datetime
import json
import os

__all__ = [
    'ReactiveInputGenerator', 'StatelessReactiveInputGenerator'
]

HAVOC_MIN_WEIGHT = 1e-3

class ReactiveInputGenerator(BaseInputGenerator,
        capture_paths=('generator.log_model_history', 'generator.log_time_step',
            'generator.log_flush_buffer')):
    def __init__(self, log_model_history: bool=False, log_time_step: float=60.,
            log_flush_buffer: int=256, **kwargs):
        super().__init__(**kwargs)
        self._seen_transitions = set()
        self._state_model = dict()

        self._log = log_model_history
        if self._log:
            self._log_counter = 0
            self._log_flush_at = log_flush_buffer
            self._log_timer = dict()
            self._log_time_step = log_time_step
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
        if (model := self._state_model.get(state)) is None:
            model = self._init_state_model(state)

        candidate = self.select_candidate(state)
        weights = map(lambda t: model['actions'][t][1], havoc_handlers)
        if hasattr(self, "_chunk_size"):
            mut = HavocMutator(candidate, weights=weights, entropy=self._entropy,
                chunk_size=self._chunk_size)
        else:
            mut = HavocMutator(candidate, weights=weights, entropy=self._entropy,
                chunk_size=None)
        return mut

    def update_state(self, state: AbstractState, /, *, input: AbstractInput,
            orig_input: AbstractInput, exc: Exception=None, **kwargs):
        if state not in self._state_model:
            self._init_state_model(state)

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, /, *,
            state_changed: bool, orig_input: AbstractInput, exc: Exception=None,
            **kwargs):
        if state_changed:
            if (t := (source, destination)) in self._seen_transitions:
                # TODO handle case where mutator results in a seen transition
                return
            else:
                self._seen_transitions.add(t)

        actions = set()
        for instruction in input:
            if isinstance(instruction, MutatedTransmitInstruction):
                actions.update(instruction.transforms)

        if not actions:
            return

        ancestors = set()
        src_model = self._state_model[source]
        normalized_reward = self._calculate_reward(source, destination)
        self._update_weights(src_model['actions'], actions, normalized_reward)

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

        if self._log:
            self._log_model(source, destination, *ancestors)

        return normalized_reward, actions

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
        now = datetime.datetime.now()
        models = dict()
        for state in states:
            last_logged = self._log_timer.get(
                state, now - datetime.timedelta(seconds=self._log_time_step+1))
            if (now - last_logged).total_seconds() > self._log_time_step:
                if not state in self._state_model:
                    continue
                models[state] = self._state_model[state]
                self._log_timer[state] = now
        self._pack_log_entry(now.timestamp(), models)

        self._log_counter += 1
        ValueProfiler('log_counter')(self._log_counter)
        if self._log_counter == self._log_flush_at:
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
        if not models:
            return
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
        super().update_state(state, input=input, orig_input=orig_input, exc=exc, **kwargs)

    def update_transition(self, source: AbstractState,
            destination: AbstractState, input: AbstractInput, /, *,
            state_changed: bool, orig_input: AbstractInput, exc: Exception=None,
            **kwargs):
        source = source.tracker._entry_state
        return super().update_transition(source, destination, input,
            state_changed=state_changed, orig_input=orig_input, exc=exc, **kwargs)