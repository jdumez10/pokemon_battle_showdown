# Pokémon Battle AI using Deep Q-Learning

This repository contains a reinforcement learning agent trained to battle in Pokémon Showdown (gen8randombattle) using a Deep Q-Learning (DQN) approach. The goal is for the agent to learn optimal battle strategies through self-play and by challenging multiple baseline opponents such as RandomPlayer, MaxBasePowerPlayer, and SimpleHeuristicsPlayer.

--------------------------------------------------------------------------------
Table of Contents
--------------------------------------------------------------------------------
- Requirements
- Installation
- Project Structure
- Usage
   - Training
   - Evaluation
   - Interactive Mode
- Model Architecture
- Performance
- Troubleshooting
- Contributing
- License

--------------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------------
Ensure you have the following dependencies installed:

    python >= 3.8
    torch
    numpy
    matplotlib
    poke-env
    gymnasium
    tabulate

You can install them using pip if needed.

--------------------------------------------------------------------------------
Installation
--------------------------------------------------------------------------------
1. Clone the repository:

       git clone https://github.com/your-username/pokemon-battle-dqn.git
       cd pokemon-battle-dqn

2. Install dependencies:

   If you have a requirements.txt file:

       pip install -r requirements.txt

   Alternatively, install them manually:

       pip install torch numpy matplotlib poke-env gymnasium tabulate

3. (Optional) Create a virtual environment:

       python -m venv venv
       source venv/bin/activate   (On Windows: venv\Scripts\activate)

--------------------------------------------------------------------------------
Project Structure
--------------------------------------------------------------------------------
The project is organized as follows:

    pokemon-battle-dqn/
    │
    ├── agent.py              # DQNAgent class for training and action selection
    ├── environment.py        # Custom poke-env environment (SimpleRLPlayer)
    ├── eval.py               # Evaluation script vs. multiple opponents
    ├── main.py               # Example of loading/trial battles with the agent
    ├── max_damage.py         # Baseline player that picks max damage moves
    ├── memory.py             # ReplayMemory class for storing transitions
    ├── models.py             # Neural network architecture for DQN
    ├── rl_player.py          # Combined environment and agent code for training/testing
    ├── train.py              # Main training script that logs/plots metrics
    ├── utilities.py          # Helper functions for damage calculations, etc.
    ├── important.txt         # Additional references or constants
    │
    ├── training_plots/       # Generated plots
    │   ├── training_rewards.png
    │   └── ...
    │
    ├── requirements.txt      # Dependency list
    └── README.md             # Project documentation

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
Training:
-----------
To train the DQN agent against a RandomPlayer baseline, run:

       python train.py

During training:
- The agent interacts with the custom Pokémon environment.
- Experience (state, action, reward, next state, done) is stored in replay memory.
- The policy network is updated periodically.
- Training metrics (losses and rewards) are logged and saved to the "training_plots" folder.
- A trained model is saved in "saved_models/dqn_model.pth".

Evaluation:
-----------
After training, evaluate the model by running:

       python eval.py

This script loads the saved model and tests it against multiple opponents 
(e.g., RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer), printing win 
rates and battle results to the terminal.

Interactive Mode:
-----------------
To run a single test battle with your agent, execute:

       python main.py

This script performs one battle and prints the result, allowing you to test the 
agent’s performance interactively.

--------------------------------------------------------------------------------
Model Architecture
--------------------------------------------------------------------------------
The DQN model is defined in "models.py" and consists of:

- Input Layer: Processes the state representation (e.g., moves' base power, type 
  multipliers, and fainted Pokémon count).
- Hidden Layers: Two fully connected layers using ELU activation.
- Output Layer: Predicts Q-values for available actions (moves and/or switches).

Note: The architecture can be extended with a dueling DQN structure to improve 
stability and convergence.

--------------------------------------------------------------------------------
Performance
--------------------------------------------------------------------------------
- Training Metrics:
    Loss curves and average rewards are logged in the "training_plots" folder.

- Evaluation Metrics:
    During testing, win rates are calculated and printed, including cross-
    evaluation against baseline opponents.

Example Output:
---------------
    Step 1000/10000 | Avg Loss: 0.0234 | Avg Reward: 1.85
    Test Episode 10/100 | Avg Reward: 15.20 | Win Rate: 50.00%

--------------------------------------------------------------------------------
Troubleshooting
--------------------------------------------------------------------------------
Common Issues:

1. Authentication Errors:
   -------------------------
   If you see messages like:
       |nametaken|RandomPlayer 1|Your authentication token was invalid.
   it usually means the username is already in use or the token is invalid.
   Solution:
       Use a unique username or register an account with a valid password.
       For example:

           from poke_env.player_configuration import PlayerConfiguration, ServerConfiguration
           from poke_env.player import RandomPlayer

           my_config = PlayerConfiguration(username="UniqueUser123", password=None)
           server_config = ServerConfiguration(
               "sim.smogon.com:443",
               "https://play.pokemonshowdown.com/~~showdown/action.php?"
           )

           opponent = RandomPlayer(
               player_configuration=my_config,
               server_configuration=server_config,
               battle_format="gen8randombattle",
               start_challenging=True
           )

2. Agent Not Challenging:
   -------------------------
   If you receive a "RuntimeError: Agent is not challenging," it could be due 
   to environment checks forcing a server connection.
   Solution:
       - Remove or comment out the "check_env(test_env)" line in train.py if testing locally.
       - Use "start_challenging=False" if you only want local tests.
       - Alternatively, run a local Pokémon Showdown server and configure your client accordingly.

3. Network Connection Issues:
   ----------------------------
   If you see a "ConnectionRefusedError," ensure your network/firewall allows 
   connections to the Pokémon Showdown server, or set up a local server as described 
   in the poke-env documentation (https://github.com/hsahovic/poke-env).

--------------------------------------------------------------------------------
Contributing
--------------------------------------------------------------------------------
Contributions, bug reports, and feature requests are welcome. To contribute:

1. Fork the repository:

       git clone https://github.com/your-username/pokemon-battle-dqn.git

2. Create a new branch:

       git checkout -b feature/YourFeature

3. Make your changes and commit them:

       git commit -m "Add YourFeature"

4. Push the branch:

       git push origin feature/YourFeature

5. Open a pull request with a clear description of your changes.

Please ensure your code follows the project’s coding style and includes 
relevant documentation or tests.

--------------------------------------------------------------------------------
License
--------------------------------------------------------------------------------
This project is licensed under the MIT License. Feel free to modify and distribute 
it as long as the original license and credits are preserved.
