from prism.core.modes import MODES, get_mode, list_modes, CognitiveMode


def test_all_modes_are_cognitive_mode():
    for name, mode in MODES.items():
        assert isinstance(mode, CognitiveMode)


def test_get_mode_valid():
    mode = get_mode("synthesis")
    assert mode.optimal_alpha == 1.5
    assert mode.task_class == "synthesis"
    assert "structural" in mode.creative_system.lower()


def test_get_mode_invalid():
    try:
        get_mode("nonexistent_mode")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_mode" in str(e)


def test_list_modes():
    modes = list_modes()
    assert len(modes) >= 8
    assert "synthesis" in modes
    assert "forensic" in modes
    assert "optimal_alpha" in modes["synthesis"]
    assert "task_class" in modes["synthesis"]


def test_alpha_ranges_valid():
    for name, mode in MODES.items():
        lo, hi = mode.alpha_range
        assert lo <= mode.optimal_alpha <= hi, (
            f"{name}: optimal_alpha {mode.optimal_alpha} not in range {mode.alpha_range}"
        )


def test_all_modes_have_prompts():
    for name, mode in MODES.items():
        assert len(mode.creative_system) > 20, f"{name} missing creative_system"
        assert len(mode.conservative_system) > 20, f"{name} missing conservative_system"


def test_security_modes_exist():
    for name in ["security_injection", "security_authorization", "security_threading"]:
        mode = get_mode(name)
        assert mode.task_class == "security"
        assert mode.optimal_alpha == 1.0
