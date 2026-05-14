import pandas as pd

from gco_hpif.interpretation.association_rules import (
    OUTCOME_HARD,
    OUTCOME_NOT_HARD,
    bin_features_by_dataset_percentile,
    evaluate_rule_as_classifier,
    percentile_edges_for_bin_count,
    pretty_item,
    resolve_min_support,
    select_non_overlapping_rules,
    to_hard_binary,
)


def test_percentile_edges_for_requested_bin_counts():
    assert percentile_edges_for_bin_count(3) == [0, 33, 66, 100]
    assert percentile_edges_for_bin_count(4) == [0, 25, 50, 75, 100]
    assert percentile_edges_for_bin_count(5) == [0, 20, 40, 60, 80, 100]


def test_to_hard_binary_handles_strings_and_numbers():
    labels = pd.Series([1, 0, "Hard", "Not Hard", "true", "false"])
    assert to_hard_binary(labels).tolist() == [1, 0, 1, 0, 1, 0]


def test_datasetwise_percentile_binning_uses_each_dataset_separately():
    df = pd.DataFrame(
        {
            "dataset": ["A", "A", "A", "B", "B", "B"],
            "feature_Number_of_Nodes": [1, 2, 3, 100, 200, 300],
        }
    )
    binned, cols = bin_features_by_dataset_percentile(df, ["feature_Number_of_Nodes"], 3)
    assert len(cols) == 3
    # Smallest value in each dataset should land in the first 0-33 bin.
    assert binned.loc[0, "BIN::feature_Number_of_Nodes::P0_33"]
    assert binned.loc[3, "BIN::feature_Number_of_Nodes::P0_33"]
    # Largest value in each dataset should land in the final 66-100 bin.
    assert binned.loc[2, "BIN::feature_Number_of_Nodes::P66_100"]
    assert binned.loc[5, "BIN::feature_Number_of_Nodes::P66_100"]


def test_rule_evaluation_reports_requested_four_cases():
    bin_matrix = pd.DataFrame(
        {
            "BIN::feature_Density::P66_100": [True, True, False, False],
        }
    )
    y = pd.Series([1, 0, 1, 0])
    stats = evaluate_rule_as_classifier(
        ["BIN::feature_Density::P66_100"],
        OUTCOME_HARD,
        bin_matrix,
        y,
    )
    assert stats["satisfy_rule_and_hard"] == 1
    assert stats["satisfy_rule_and_not_hard"] == 1
    assert stats["not_satisfy_rule_and_hard"] == 1
    assert stats["not_satisfy_rule_and_not_hard"] == 1
    assert "weighted_f1" in stats
    assert "minority_class_f1" in stats


def test_pretty_item_formats_bins_and_outcomes():
    assert pretty_item(OUTCOME_HARD) == "Hard"
    assert pretty_item(OUTCOME_NOT_HARD) == "Not Hard"
    assert pretty_item("BIN::feature_Number_of_Nodes::P75_100") == "Number of Nodes in [75%, 100%]"


def test_auto_min_support_does_not_exceed_minority_prevalence():
    y = pd.Series([1] * 5 + [0] * 95)
    value = resolve_min_support("auto", y)
    assert 0 < value <= 0.05


def test_non_overlapping_selection_uses_covered_rows_not_just_item_names():
    rules = pd.DataFrame(
        [
            {
                "antecedents": frozenset(["BIN::feature_A::P0_50"]),
                "consequents": frozenset([OUTCOME_HARD]),
                "support": 0.50,
                "lift": 2.0,
                "confidence": 0.8,
            },
            {
                "antecedents": frozenset(["BIN::feature_B::P0_50"]),
                "consequents": frozenset([OUTCOME_HARD]),
                "support": 0.40,
                "lift": 3.0,
                "confidence": 0.9,
            },
        ]
    )
    bin_matrix = pd.DataFrame(
        {
            "BIN::feature_A::P0_50": [True, True, False, False],
            "BIN::feature_B::P0_50": [False, False, True, True],
        }
    )
    selected = select_non_overlapping_rules(rules, bin_matrix, max_rules=10, min_matched_rows=1)
    assert len(selected) == 2

from pathlib import Path
import pytest

from gco_hpif.interpretation.association_rules import (
    DEFAULT_TARGETS,
    discover_selected_features_path,
    validate_selected_features,
)


def test_selected_feature_discovery_is_target_specific_and_no_primary_fallback(tmp_path):
    root = tmp_path / "ml_hardness_part_f"
    primary = root / DEFAULT_TARGETS[0] / "tables"
    primary.mkdir(parents=True)
    (primary / "selected_features_anova_first_peak.csv").write_text("feature\nfeature_A\n")

    # Consensus-5 exists and is found.
    found = discover_selected_features_path(
        root,
        DEFAULT_TARGETS[0],
        "selected_features_anova_first_peak.csv",
    )
    assert found == primary / "selected_features_anova_first_peak.csv"

    # Consensus-4 does not exist. The function must not silently reuse Consensus-5.
    with pytest.raises(FileNotFoundError):
        discover_selected_features_path(
            root,
            DEFAULT_TARGETS[1],
            "selected_features_anova_first_peak.csv",
        )


def test_validate_selected_features_rejects_all_feature_like_large_set(tmp_path):
    selected_file = tmp_path / "selected_features_anova_first_peak.csv"
    features = [f"feature_{i}" for i in range(23)]
    with pytest.raises(ValueError, match="Too many selected features"):
        validate_selected_features(
            features,
            selected_features_path=selected_file,
            max_selected_features_per_target=10,
        )


def test_validate_selected_features_accepts_small_first_peak_set(tmp_path):
    selected_file = tmp_path / "selected_features_anova_first_peak.csv"
    features = ["feature_Number_of_Nodes", "feature_Density", "feature_Clustering"]
    assert validate_selected_features(features, selected_file, 10) == features
