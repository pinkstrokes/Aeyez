from src.seeingeye.tools.action_motion_scan import ACTION_MOTION_SCAN_SYSTEM


def test_action_motion_prompt_models_egocentric_action_and_motion_paths() -> None:
    assert (
        "Action = hand pose + active object + contact target + temporal motion + scene context"
        in ACTION_MOTION_SCAN_SYSTEM
    )
    assert "movable_entities" in ACTION_MOTION_SCAN_SYSTEM
    assert "predicted_motion_paths" in ACTION_MOTION_SCAN_SYSTEM
    assert "contact_or_collision_targets" in ACTION_MOTION_SCAN_SYSTEM
    assert "route_conflicts" in ACTION_MOTION_SCAN_SYSTEM
    assert "line_of_fire_or_release_paths" in ACTION_MOTION_SCAN_SYSTEM
