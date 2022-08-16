from brax import envs
from brax.training.agents import act_grad
import jax

def main(unused_argv):
    env = envs.create('ant')
    state = env.reset(0)

    policy = act_grad.make_policy()
    act = policy(env, state, 20)
    

    # @jax.jit
    # def jit_next_state(env, state, action_size, key):
    #     new_key, tmp_key = jax.random.split(key)
    #     policy = act_grad.make_policy()

    #     act = policy(env, state, action_size, 20)
    #     return env.step(state, act), act, new_key

    # state = env.reset(0)
    # jit_next_state(env, state, jax.random.PRNGKey(0))