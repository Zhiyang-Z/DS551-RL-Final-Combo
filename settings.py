from diambra.arena import SpaceTypes, Roles

all_settings = {
    'basic': {
        'step_ratio': 1,
        'difficulty': 8,
        'frame_shape': (112, 192, 1),
        'role': Roles.P1,
        'characters': 'Ken',
        'action_space': SpaceTypes.MULTI_DISCRETE,
        'continue_game': 0.0
    },
    'wrapper': {
        'normalize_reward': True,
        # 'normalization_factor': 0.5,
        'stack_frames': 5,
        'add_last_action': True,
        'stack_actions': 12,
        'scale': True,
        'exclude_image_scaling': True,
        'flatten': True,
        'filter_keys': [],
        'role_relative': True,
    },
    'agent': {
        'model_folder': '/data/programs_data/RL_final_project/',
        'log_dir': '/data/programs_data/RL_final_project/logs/',
        'model_checkpoint': 'PPO',
        'autosave_freq': 30000,
        'time_steps': int(5e6)
    }
}