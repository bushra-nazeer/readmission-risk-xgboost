from readmission.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.random_state == 42
    assert cfg.target_positive_label == "<30"
    assert cfg.paths.model_path.endswith("model.pkl")
    assert "race" in cfg.sensitive_features
    assert 0 < cfg.test_size < 1
