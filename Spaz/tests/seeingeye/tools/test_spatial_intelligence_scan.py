from src.seeingeye.tools.spatial_intelligence_scan import (
    PROJECTION_SECTION_VERIFIER_SYSTEM,
    SPATIAL_INTELLIGENCE_SYSTEM,
    SPATIAL_REGION_SELECTION_SYSTEM,
    _looks_like_projection_task,
    _parse_regions,
)


def test_spatial_region_selection_prompt_uses_guided_visual_search() -> None:
    assert "guided visual-search planner" in SPATIAL_REGION_SELECTION_SYSTEM
    assert "4x4 grid" in SPATIAL_REGION_SELECTION_SYSTEM
    assert "Select regions that clarify spatial relations" in SPATIAL_REGION_SELECTION_SYSTEM
    assert "could disprove the obvious first answer" in SPATIAL_REGION_SELECTION_SYSTEM


def test_spatial_intelligence_prompt_requires_candidate_verification() -> None:
    assert "viewpoint_frame" in SPATIAL_INTELLIGENCE_SYSTEM
    assert "dynamic_before_after_hypotheses" in SPATIAL_INTELLIGENCE_SYSTEM
    assert "candidate_verification_table" in SPATIAL_INTELLIGENCE_SYSTEM
    assert "geometric_consistency_checks" in SPATIAL_INTELLIGENCE_SYSTEM
    assert "Do not fixate on the first visually salient path" in SPATIAL_INTELLIGENCE_SYSTEM
    assert "For math/diagram questions" in SPATIAL_INTELLIGENCE_SYSTEM


def test_projection_section_verifier_prompt_compares_options() -> None:
    assert "projection and cross-section verifier" in PROJECTION_SECTION_VERIFIER_SYSTEM
    assert "option_alignment_table" in PROJECTION_SECTION_VERIFIER_SYSTEM
    assert "Eliminate options one by one" in PROJECTION_SECTION_VERIFIER_SYSTEM
    assert "outer silhouette" in PROJECTION_SECTION_VERIFIER_SYSTEM
    assert "cut-plane A-A arrows" in PROJECTION_SECTION_VERIFIER_SYSTEM


def test_projection_task_detector_catches_engineering_graphics() -> None:
    assert _looks_like_projection_task("Select the correct left view()", "")
    assert _looks_like_projection_task("From the A-A section, select the correct section.", "")
    assert _looks_like_projection_task("ordinary route question", "") is False


def test_parse_regions_accepts_json_and_caps_regions() -> None:
    text = '{"regions": [[1, 2], [5, -1], [8, 11], [12, 15], [0, 15]]}'
    assert _parse_regions(text) == [[1, 2], [5, -1], [8, 11], [12, 15]]


def test_parse_regions_falls_back_to_whole_image() -> None:
    assert _parse_regions("no useful regions") == [[0, 15]]
