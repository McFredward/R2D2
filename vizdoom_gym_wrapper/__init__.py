from gym.envs.registration import register

register(
    id="VizdoomBasic-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "basic.cfg"}
)

register(
    id="VizdoomCorridor-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "deadly_corridor.cfg"}
)

register(
    id="VizdoomDefendCenter-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_center.cfg"}
)

register(
    id="VizdoomDefendLine-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_line.cfg"}
)

register(
    id="VizdoomHealthGathering-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "health_gathering.cfg"}
)

register(
    id="VizdoomMyWayHome-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "my_way_home.cfg"}
)

register(
    id="VizdoomPredictPosition-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "predict_position.cfg"}
)

register(
    id="VizdoomTakeCover-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "take_cover.cfg"}
)

register(
    id="VizdoomDeathmatch-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "deathmatch.cfg"}
)

register(
    id="VizdoomHealthGatheringSupreme-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "health_gathering_supreme.cfg"}
)

register(
    id="VizdoomBasicWithAttack-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "basic_with_attack.cfg"}
)

register(
    id="VizdoomBasicWithAttackLessActions-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "basic_with_attack_less_actions.cfg"}
)

register(
    id="VizdoomBasicDeathmatch-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "multi.cfg"}
)

register(
    id="VizdoomSingleDeathmatch-v0",
    entry_point="vizdoom_gym_wrapper.gym_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "multi_single.cfg"}
)