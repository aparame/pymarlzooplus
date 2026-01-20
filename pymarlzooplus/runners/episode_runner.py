import datetime
import time
import json
import os
from functools import partial
import numpy as np

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.utils.env_utils import check_env_installation


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class EpisodeRunner:

    def __init__(self, args, logger):

        # Check if the requirements for the selected environment are installed
        check_env_installation(args.env, env_REGISTRY, logger)

        self.batch = None
        self.new_batch = None
        self.mac = None
        self.explorer = None
        if (
            args.explorer == "maven"
        ):  # MAVEN uses a noise vector which augments the observation
            self.noise = None

        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # Initialize environment
        assert not (
            self.args.env == "pettingzoo"
            and self.args.env_args["centralized_image_encoding"] is True
        ), (
            "In 'episode_runner', the argument 'centralized_image_encoding' of pettingzoo should be False "
            "since there is only one environment, and thus the encoding can be considered as centralized."
        )
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        # Get info from environment to be printed
        print_info = self.env.get_print_info()
        if print_info != "None" and print_info is not None:
            # Simulate the message format of the logger defined in _logging.py
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.logger.console_logger.info(
                f"\n[INFO {current_time}] episode_runner {print_info}"
            )

        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, explorer):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.explorer = explorer

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        env_info = {}

        # --- DATA COLLECTION INIT ---
        save_data = getattr(self.args, "save_data", False)
        if save_data:
            # Try to get unwrapped env
            unwrapped_env = self.env.original_env.unwrapped

            # Enum Mappings
            try:
                from rware.warehouse import Action, Direction

                action_map = {a.value: a.name for a in Action}
                # RWARE Direction: UP=0, DOWN=1, LEFT=2, RIGHT=3
                # User requested "NORTH", "SOUTH" etc?
                # User example says "NORTH".
                # Assuming UP=NORTH, DOWN=SOUTH, LEFT=WEST, RIGHT=EAST
                dir_map = {
                    Direction.UP.value: "NORTH",
                    Direction.DOWN.value: "SOUTH",
                    Direction.LEFT.value: "WEST",
                    Direction.RIGHT.value: "EAST",
                }
            except ImportError:
                # Fallback if rware not importable (unlikely inside env)
                action_map = {}
                dir_map = {}

            # Metadata
            grid_size = getattr(unwrapped_env, "grid_size", [10, 20])
            total_steps = self.args.t_max

            # Initial State
            shelves = []
            if hasattr(unwrapped_env, "shelves"):
                for s in unwrapped_env.shelves:
                    shelves.append({"id": s.id, "x": s.x, "y": s.y})

            init_agents = []
            if hasattr(unwrapped_env, "agents"):
                for a in unwrapped_env.agents:
                    # Color is usually not in agent state, hardcode or randomize
                    colors = ["blue", "red", "green", "yellow"]
                    color = colors[
                        (a.id - 1) % len(colors)
                    ]  # Agent IDs usually 1-indexed in name, 0 in list?
                    init_agents.append(
                        {"id": a.id, "start_x": a.x, "start_y": a.y, "color": color}
                    )

            trajectory_data = {
                "metadata": {
                    "env": self.args.env_args.get("key", self.args.env),
                    "grid_size": list(grid_size),
                    "total_steps": total_steps,
                },
                "initial_state": {"shelves": shelves, "agents": init_agents},
                "trajectory": [],  # List of steps
            }
        # ----------------------------

        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            # --- DATA COLLECTION STEP (PRE-ACTION snapshot) ---
            # We record state BEFORE action is applied?
            # User format: "step": 1. "agents": [{action: "FORWARD", pos: [0,1]}]
            # Usually 'pos' is the position *before* the action or *result*?
            # "action" is what they *took* at this step.
            # "pos" is likely where they *were* when they took it.
            # So we capture state now, and append action later.

            if save_data:
                unwrapped_env = self.env.original_env.unwrapped

                # Requests
                request_queue = []
                if hasattr(unwrapped_env, "request_queue"):
                    # rware request queue is list of Request namedtuples (shelf_id, ...)
                    # request_queue = [r.shelf_id for r in unwrapped_env.request_queue]
                    # Correction: It seems they are Shelf objects with .id
                    request_queue = [r.id for r in unwrapped_env.request_queue]

                current_step_record = {
                    "step": self.t + 1,
                    "requests": request_queue,
                    "agents": [],
                }

                # Capture Agent Status (Position BEFORE action)
                if hasattr(unwrapped_env, "agents"):
                    for agent in unwrapped_env.agents:

                        agent_info = {
                            "id": agent.id,
                            # Action will be filled after selection
                            "action": "NOOP",
                            "pos": [agent.x, agent.y],
                            "dir": dir_map.get(
                                (
                                    agent.dir.value
                                    if hasattr(agent.dir, "value")
                                    else agent.dir
                                ),
                                "UNKNOWN",
                            ),
                            "carrying": (
                                agent.carrying_shelf.id
                                if agent.carrying_shelf
                                else None
                            ),
                        }
                        current_step_record["agents"].append(agent_info)

                # We need to hold this record until we have actions
            # -----------------------------------------

            self.batch.update(pre_transition_data, ts=self.t)

            # MAVEN noise ...
            if self.args.explorer == "maven" and self.t == 0:
                self.noise = self.explorer.sample(self.batch["state"][:, 0])
                self.batch.update({"noise": self.noise}, ts=0)

            # Select Actions
            actions, extra_returns = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            if self.args.explorer == "eoi":
                actions = self.explorer.select_actions(
                    actions, self.t, test_mode, pre_transition_data
                )

            # Step
            reward, terminated, env_info = self.env.step(actions[0])

            # --- DATA COLLECTION UPDATE (ACTION) ---
            if save_data:
                # Update actions in current_step_record
                chosen_actions = actions[0].cpu().numpy()  # indices

                for idx, ag_dict in enumerate(current_step_record["agents"]):
                    # Assuming agent order mirrors unwrapped.agents list
                    act_idx = chosen_actions[idx]
                    action_name = action_map.get(act_idx, str(act_idx))

                    if action_name == "TOGGLE_LOAD" and ag_dict["carrying"] is not None:
                        action_name = "TOGGLE_UNLOAD"

                    ag_dict["action"] = action_name

                trajectory_data["trajectory"].append(current_step_record)
            # ---------------------------------------

            # Render
            if test_mode and self.args.render:
                self.env.render()
                if hasattr(self.args, "fps") and self.args.fps is not None:
                    time.sleep(1.0 / self.args.fps)

            # Print info ...
            print_info = self.env.get_print_info()
            if print_info != "None" and print_info is not None:
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                self.logger.console_logger.info(
                    f"\n[INFO {current_time}] episode_runner {print_info}"
                )

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        actions, extra_returns = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # --- DATA COLLECTION SAVE ---
        if save_data:
            # Define success based on user req
            is_success = terminated and not env_info.get("TimeLimit.truncated", False)
            trajectory_data["metadata"]["success"] = is_success

            outcome = "success" if is_success else "fail"
            timestamp = datetime.datetime.now().strftime(
                "%H%M%S"
            )  # Time only for filename if date is in folder?
            # User asked for <date>/trajectories.json
            # Use date in folder, time in filename.

            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            algo_name = self.args.name
            env_name = self.args.env_args.get("key", self.args.env)

            # Construct User Requested Path
            # <results>/<algo-name>/<env-name>/<date>/
            results_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "results",
                algo_name,
                env_name,
                date_str,
            )
            os.makedirs(results_dir, exist_ok=True)

            filename = f"trajectory_{timestamp}_{outcome}.json"
            filepath = os.path.join(results_dir, filename)

            with open(filepath, "w") as f:
                json.dump(trajectory_data, f, indent=4, cls=NumpyEncoder)

            print(f"Saved trajectory to {filepath}")
        # ----------------------------

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch, [episode_return]

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
