# Create and reset a battle
# imports - bring in tensor, neural network operations, ActorCritic, battle simulator, opponent_policy
import torch
from hal_brains import ActorCritic
from HAL9000 import NUM_ACTIONS
import random
import cr_sim
from HAL9000 import (
    DECK, 
    encode_observation,
    action_mask,
    decode_action,
)
from random_agent import policy as opponent_policy

def main():
    # Create battle and player views; make battle with same deck for both sides
    battle = cr_sim.Battle()
    battle.reset(seed = 42, deck0 = DECK, deck1 = DECK)
    my_view = battle.view(0)
    opponent_view = battle.view(1)
    # Create and check network; encodes starting position and uses its length to determine networks input size
    features = encode_observation(my_view.observe())
    model = ActorCritic(len(features), NUM_ACTIONS)
    # Converts python list into PyTorch tensor w/ 32-bit floating point #s
    observation_tensor = torch.tensor(features, dtype=torch.float32)
    # .no_grad ensures data for backpropogation not stored for now; calculates models predicted values for each output, and estimated reward
    with torch.no_grad():
        logits, value = model(observation_tensor)

    # prints shape of estimated values
    print("Action scores shape:", logits.shape)
    # prints the estimated returned value
    print("Value estimate:", value.item())

    # choose and store actions during match
    transitions = []
    while not battle.done():
        obs = my_view.observe()
        # Encodes observation into neural net friendly format
        model_input = encode_observation(obs)
        # Determines allowed action ids
        mask = action_mask(obs)
        # randomly chooses a valid action
        allowed = [i for i, valid in enumerate(mask) if valid]
        # converts into valid action
        action_id = random.choice(allowed)
        action = decode_action(action_id)
        # Choose opponents move and advance
        opponent_action = opponent_policy(opponent_view.observe())
        if opponent_action is not None:
            opponent_action = tuple(opponent_action)
        battle.step(action, opponent_action)
        # Calculate reward and save experience
        done = battle.done()
        reward = 0.0
        if done:
            reward = {"win": 1, "draw": 0, "loss": -1}[my_view.result()]
        # Stores: state before action, selected action #, available actions for that move, feedback after move, state after move, whether step ended match
        transitions.append({
            "observations": model_input,
            "action_id": action_id,
            "action_mask": mask,
            "reward": reward,
            "next_observation": encode_observation(my_view.observe()),
            "done": done,
        })
    last = transitions[-1]
    obs_length = len(last["observations"])
    all_same = True
    done_count = 0
    for transition in transitions:
        if transition["done"]:
            done_count += 1
        if len(transition["observations"]) != obs_length:
            all_same = False
    last = transitions[-1]
    print("Last done?", last["done"])
    print("Done count", done_count)
    print("Final reward:", last["reward"])
    print("Result:", my_view.result()) 
    print("Observations have same len?", all_same)

# Ensures file is only run directly, not while being imported
if __name__ == "__main__":
    main()