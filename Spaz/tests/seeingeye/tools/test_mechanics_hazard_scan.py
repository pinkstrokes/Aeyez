from src.seeingeye.tools.mechanics_hazard_scan import MECHANICS_HAZARD_SCAN_SYSTEM


def test_mechanics_hazard_prompt_models_failure_and_line_of_fire() -> None:
    assert "mechanics hazard verifier" in MECHANICS_HAZARD_SCAN_SYSTEM
    assert "load_path" in MECHANICS_HAZARD_SCAN_SYSTEM
    assert "stored_energy_sources" in MECHANICS_HAZARD_SCAN_SYSTEM
    assert "likely_failure_modes" in MECHANICS_HAZARD_SCAN_SYSTEM
    assert "predicted_motion_paths" in MECHANICS_HAZARD_SCAN_SYSTEM
    assert "line_of_fire_zones" in MECHANICS_HAZARD_SCAN_SYSTEM
    assert "pinch_crush_shear_zones" in MECHANICS_HAZARD_SCAN_SYSTEM
    assert "falling_or_swinging_object_zones" in MECHANICS_HAZARD_SCAN_SYSTEM
