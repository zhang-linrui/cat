import argparse
import numpy as np
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from advgen.adv_generator import AdvGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_raw',action='store_true')   # visualize the raw/adv scene
    parser.add_argument('--OV_traj_num', type=int,default=32)
    parser.add_argument('--AV_traj_num', type=int,default=1)
    adv_generator = AdvGenerator(parser)

    args = parser.parse_args()

    extra_args = dict(mode="top_down", film_size=(2200, 2200))

    env = WaymoEnv(
            {
                "agent_policy": ReplayEgoCarPolicy,
                "reactive_traffic": False,
                #"use_render": True,
                "data_directory": './raw_scenes_100',
                "num_scenarios": 100,
                "force_reuse_object_name" :True,
                "sequential_seed": True,
                "vehicle_config":dict(show_navi_mark=False,show_dest_mark=False,)
            }
        )

    for i in range(100):

      env.reset(force_seed=i)
      env.render(**extra_args)

      done = False
      ep_timestep = 0
      adv_generator.before_episode(env)
      adv_generator.generate()
      env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,adv_generator.adv_traj)
      env.engine._top_down_renderer.set_adv(adv_generator.adv_agent)
      
      while not done:
        # keep the 1s history fixed and modify the following 8s
        # if ep_timestep == 11 and not args.viz_raw:
        #   env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,adv_generator.adv_traj)
        
        o, r, done, info = env.step([1.0, 0.])
        env.render(**extra_args)
        #env.engine._top_down_renderer.set_adv(adv_generator.adv_agent)
        ep_timestep += 1

        if done:
          adv_generator.after_episode()
          break

    env.close()