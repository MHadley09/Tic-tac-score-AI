## Instructions
<b>Dependencies required:</b>
- python 3.0
- tensorflow
- numpy
- copy

<b>Acknowledgements:</b>
- AlphaZero code was adapted from
https://adsp.ai/blog/how-to-build-your-own-alphazero-ai-using-python-and-keras/
- Also adapted from paper found at: https://arxiv.org/abs/2207.05991
- Original source found at: https://github.com/tanchongmin/bttt

<b>File Descriptions:</b>
- `main.py`: runs the Tic-tac-toe Scores tournament.
- `gym.py`: is the code for the Tic-tac-toe Scores environment, plus the in-built Minimax, MCTS and Random agents.
- `time.py`: runs the code to determine the runtime of each agent.
- `alphazero.py`: contains the code for the AlphaZero agent. Change the `MODEL_NAME` to the .h5 file which contains the trained model's weights.
- `config.py`: Contains the hyperparameters of the AlphaZero model. To select `AlphaZero NS`, `AlphaZero 100` and `AlphaZero 1000`, set `MCTS_SIMS` to 0, 100, 1000 respectively. Make sure `DIRICHLET` is set to *False* during the testing phase to let the model play optimally.
- `Tutorial.ipynb`: a tutorial on how to use the Tic-tac-toe Scores environment.
- `own_agent.py`: contains the code for your own agent. (Default: Random)
- Other files and `run` folder: used to train AlphaZero agent

<b>Tic-tac-toe Scores Tournament Setup Instructions:</b>
- To run the tournament, type `python main.py` into a command line interface
- Follow the prompts to select Player 1, Player 2, as well as the number of games
- To edit the version of AlphaZero you want to do it, edit the hyperparameters in `config.py`

<b>Runtime Analysis Instructions:</b>
- To do runtime analysis for each agent, type `python time.py` into a command line interface

<b>AlphaZero training instructions:</b>
- To train AlphaZero, open `alphazero.ipynb` and run the cells
- Hyperparameters can be edited in `config.py`. Important: Set `DIRICHLET` variable to *True* when training, and *False* when testing. This controls the amount of exploration in the first `TURNS_UNTIL_TAU0` steps through adding of dirichlet noise.
- Initial configurations can be edited in `initialize.py`
- Folder options can be edited in `settings.py`. Default model and memory storage folder is `./run`
