# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Action gradient descent
#   Simulate ahead and perform gradient descent on actions rather than a policy

from brax import envs
from brax.training import types

from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax
import functools


def make_policy() -> types.Policy:

    def policy(env : envs.Env, state : envs.State, num_steps : int) -> Tuple[types.Action, types.Extra]:
    # def policy(state, action_size : int, num_steps : int) -> Tuple[types.Action, types.Extra]:

        actions_arr = jnp.zeros((num_steps, env.action_size))

        # rewards = jax.lax.scan(run_ahead, state, actions_arr)

        step = jax.jit(env.step)

        def run_ahead(actions, state):
            reward = 0
            for i in range(num_steps):
                state = step(state, actions[i])
                reward = state.reward
                # print(i, reward)
            return reward

        def apply_gradient(actions, state):
            run_ahead_grad = jax.jit(jax.grad(run_ahead))
            g_a = run_ahead_grad(actions, state)
            return actions + g_a

        actions_arr = apply_gradient(actions_arr, state)
        print(actions_arr)


        return actions_arr

    return policy