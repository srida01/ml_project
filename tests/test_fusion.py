from ids.pipeline.fuse import FusionPolicy, fuse_decision


def test_fusion_distinguishes_anomaly_only_from_known_attack():
    policy = FusionPolicy(known_confidence_threshold=0.6)
    assert fuse_decision(svm_suspicious=False, known_attack_label=None, known_attack_confidence=None, anomaly_flag=True, policy=policy)["final_alert_label"] == "anomaly_review"
    assert fuse_decision(svm_suspicious=True, known_attack_label="DoS", known_attack_confidence=0.9, anomaly_flag=True, policy=policy)["final_alert_level"] == "high"
    assert fuse_decision(svm_suspicious=True, known_attack_label="DoS", known_attack_confidence=0.2, anomaly_flag=False, policy=policy)["final_alert_label"] == "suspicious_unclassified"
