from src.seeingeye.tools.route_surface_scan import (
    DYNAMIC_CLEAR_VERIFIER_SYSTEM,
    ROUTE_SURFACE_SCAN_SYSTEM,
)


def test_route_surface_scan_prompt_prevents_route_fixation() -> None:
    assert "Do not lock onto the first apparently safe lane" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "side_openings_before_or_beside_blockage" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "kept_alternatives" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "candidate_ranking_not_final" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "dynamic_object_before_after" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "space_revealed_after_dynamic_clear" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "ego_centered_route_map" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "predicted_motion_paths_by_entity" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "current_position, forward/left/right candidates" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "overhead_hazards_by_route" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "struck_by_or_falling_object_zones" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "Scan overhead hazards with the same priority as floor hazards" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "before/after scene-change hypothesis" in ROUTE_SURFACE_SCAN_SYSTEM
    assert "before, beside, or just beyond a blockage" in ROUTE_SURFACE_SCAN_SYSTEM


def test_dynamic_clear_verifier_prompt_checks_revealed_routes() -> None:
    assert "dynamic-clear route verifier" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
    assert "mentally remove it" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
    assert "revealed_after_clear" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
    assert "side_openings_after_clear" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
    assert "overhead_hazards_after_clear" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
    assert "after_clear_step_sequence" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
    assert "move forward slightly, turn right/left" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
    assert "wide on the floor but unsafe" in DYNAMIC_CLEAR_VERIFIER_SYSTEM
