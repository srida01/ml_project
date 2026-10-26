from ids.data.labels import LabelMapping


def test_label_mapping_preserves_raw_and_binary_targets():
    mapping = LabelMapping("BENIGN", {"BENIGN": "BENIGN", "DoS Hulk": "DoS", "SSH-Patator": "Patator"})
    result = mapping.transform([" BENIGN ", "DoS Hulk", "SSH-Patator"])
    assert result["raw_label"].tolist() == ["BENIGN", "DoS Hulk", "SSH-Patator"]
    assert result["binary_label"].tolist() == [0, 1, 1]
    assert result["multiclass_label"].tolist() == ["BENIGN", "DoS", "Patator"]
