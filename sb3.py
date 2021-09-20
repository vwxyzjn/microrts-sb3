import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from wandb.integration.sb3 import WandbCallback
from gym_microrts import microrts_ai
from gym_microrts.envs.new_vec_env import MicroRTSGridModeVecEnv
import numpy as np
import gym


def mask_fn(env: gym.Env) -> np.ndarray:
    # Uncomment to make masking a no-op
    # return np.ones_like(env.action_mask)
    return env.get_action_mask()


def get_wrapper(env: gym.Env) -> gym.Env:
    return ActionMasker(env, mask_fn)


config = {
    "total_timesteps": int(100e6),
    "num_envs": 8,
    "env_name": "BreakoutNoFrameskip-v4",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

num_envs = 24
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=num_envs,
    partial_obs=False,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(num_envs - 6)]
    + [microrts_ai.randomBiasedAI for _ in range(min(num_envs, 2))]
    + [microrts_ai.lightRushAI for _ in range(min(num_envs, 2))]
    + [microrts_ai.workerRushAI for _ in range(min(num_envs, 2))],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
envs = VecMonitor(envs)
envs = VecVideoRecorder(envs, "videos", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)  # record videos
model = MaskablePPO(
    "CnnPolicy",
    envs,
    n_steps=128,
    n_epochs=4,
    learning_rate=lambda progression: 2.5e-4 * progression,
    ent_coef=0.01,
    clip_range=0.1,
    batch_size=256,
    verbose=1,
    tensorboard_log=f"runs",
)
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
    ),
)