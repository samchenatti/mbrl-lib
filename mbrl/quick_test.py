from algorithms.mbpo import evaluate
import mbrl
import gymnasium
from mysac.envs.nao import WalkingNao
from mbrl.third_party.pytorch_sac import VideoRecorder

gymnasium.register(id='WalkingNao-v0', entry_point=WalkingNao)

cfg = mbrl.util.common.load_hydra_cfg("/tmp/2023.07.24/183346")

handler = mbrl.util.create_handler(cfg)
env, *_ = handler.make_env(cfg)

agent = mbrl.planning.load_agent("/tmp/2023.07.24/183346/", env)

avg = evaluate(
    env=env,
    agent=agent,
    num_episodes=10,
    max_steps=500,
    video_recorder=VideoRecorder(None)
)

print(avg)