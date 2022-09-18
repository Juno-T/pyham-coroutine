# Python implementation of Hierarchies of Abstract Machines (HAMs) as a python's coroutine.

**:warning: Abandoned because of the limitation. [See new repo here](https://github.com/Juno-T/pyham)**

Hierarchies of Abstract Machines, also known as HAMs, is a framework for hierarchical reinforcement learning (HRL) which was first introduced in [this paper](https://proceedings.neurips.cc/paper/1997/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)(Parr & Russell, 1997). HAMs is essentially a finite state machine which could be easily implemented with coroutine in python. 

In the original paper, there are 4 types of machine, ACTION, CHOICE, CALL, STOP. Since we are using normal python's function to define machines, we don't need to have CALL or STOP machine. CALL machine becomes simply `ham.CALL(func, args=None)` and STOP machine becomes python's native `return` statement. Additionally, FUNCTIONAL machine is introduced for convenient purpose. Any thing that isn't ACTION or CHOICE can be registered as a FUNCTIONAL machine. More on that in the example.

**IMPORTANT: THIS IS NOT AN OFFICIAL IMPLEMENTATION OF THE PAPER**  


## Example usage

First, add this repo to your `$PYTHONPATH`.

Next, setup our HAMs. It requires primitive action executor (`env_exe`), a method that get called when a new transition is available (`transition_handler`) and internal reward discount value (`discount`)

``` python

from ham import HAM

replay_buffer = []

# Dummy environment. In real usecase, you probably need to use `gym.env.step(action)` instead.
def env_exe(action):
  return "observation", 0.5, False, "info"
  
# Handle transition by storing it in replay buffer
def transition_handler(transition):
  replay_buffer.append(transition)
  
# Internal discount
discount = 0.99

myham = HAM(env_exe, transition_handler, discount)
```

Now, design and register your machines as python functions and add decorators accordingly.

``` python

# Choice machine
@myham.learnable_choice_machine
def repetition_choice(ham: HAM, args:int):  
  return int(args)+1  # return a choice selected

# Action machine
@myham.action_machine
def action_machine(ham: HAM, args):
  return args["my action"]+"!!"  # return an action to be executed.

# Functional machine
@myham.functional_machine
def loop_machine(ham: HAM, args):
  for i in range(10):
    rep = ham.CALL(repetition_choice, i)
    for _ in range(rep):
      ham.CALL(action_machine, {'action': "my action"})
  return None  # Functional machine can return arbitrary info.

assert(myham.machine_count==3) # Registered three machines

# Initiate episode with initial observation, e.g. an output from gym.env.reset()
# Must be called before each episode.
myham.episodic_reset("initial observation")

# Start HAMs by calling your top level machine
myham.CALL(loop_machine)

assert(len(replay_buffer)==9) # 9 transitions, since there are ten choice points and env did not terminated yet
```

Notice that each machine can only takes two arguments, `ham` and `args`. `ham` is the current instance of HAM. You can access many useful info like `ham._current_observation` or `ham._machine_stack`. `args` is specified when you call the machine with `ham.CALL`. 

This is a simple example. You can put whatever you want in these machines. For example, inside choice machine, you can call an external value approximator like dqn given `ham._current_observation` and return the choice selected. You can also reduced your action machine if it's just returning trivial action by using `ham.CALL_action`. Details are in the code and doc string and some example in the test.


## Limitation
Although this implementation of HAMs is pretty concise and straight-forward for the user, the limitation of coroutine makes it hard for users to integrate HAMs with existing popular RL algorithm implementation. Their training function usually takes `gym.env`-like instance.
