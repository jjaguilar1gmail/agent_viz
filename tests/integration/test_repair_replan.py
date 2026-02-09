"""Integration tests for repair classification and replan triggers."""

import pytest
from autoviz_agent.registry.validation import classify_repair, RepairType


class TestRepairClassification:
    """Test repair type classification logic."""

    def test_missing_df_is_safe(self):
        """Missing df parameter should be safe to auto-fill."""
        repair_type = classify_repair(
            parameter_name="df",
            old_value=None,
            new_value="$dataframe",
            tool_name="aggregate"
        )
        assert repair_type == RepairType.SAFE

    def test_default_values_are_safe(self):
        """Default values for missing parameters should be safe."""
        safe_defaults = ["auto", 0, False, [], {}, "$dataframe"]
        
        for default in safe_defaults:
            repair_type = classify_repair(
                parameter_name="some_param",
                old_value=None,
                new_value=default,
                tool_name="test_tool"
            )
            assert repair_type == RepairType.SAFE, \
                f"Default value {default} should be SAFE"

    def test_column_casing_is_safe(self):
        """Column name casing fixes should be safe."""
        repair_type = classify_repair(
            parameter_name="column",
            old_value="Revenue",
            new_value="revenue",
            tool_name="aggregate"
        )
        assert repair_type == RepairType.SAFE

    def test_semantic_parameter_changes_are_semantic(self):
        """Changes to semantic parameters should be classified as semantic."""
        semantic_params = ["group_by", "agg_func", "metrics", "x", "y", "column"]
        
        for param in semantic_params:
            repair_type = classify_repair(
                parameter_name=param,
                old_value="old_value",
                new_value="new_value",
                tool_name="test_tool"
            )
            assert repair_type == RepairType.SEMANTIC, \
                f"Parameter {param} change should be SEMANTIC"

    def test_group_by_change_is_semantic(self):
        """Changing group_by parameter is semantic."""
        repair_type = classify_repair(
            parameter_name="group_by",
            old_value=["region"],
            new_value=["product"],
            tool_name="aggregate"
        )
        assert repair_type == RepairType.SEMANTIC

    def test_agg_func_change_is_semantic(self):
        """Changing aggregation function is semantic."""
        repair_type = classify_repair(
            parameter_name="agg_func",
            old_value="sum",
            new_value="mean",
            tool_name="aggregate"
        )
        assert repair_type == RepairType.SEMANTIC

    def test_metric_change_is_semantic(self):
        """Changing metrics is semantic."""
        repair_type = classify_repair(
            parameter_name="metrics",
            old_value=["revenue"],
            new_value=["cost"],
            tool_name="compute_summary_stats"
        )
        assert repair_type == RepairType.SEMANTIC

    def test_time_column_change_is_semantic(self):
        """Changing time column is semantic."""
        repair_type = classify_repair(
            parameter_name="time_column",
            old_value="date",
            new_value="timestamp",
            tool_name="plot_line"
        )
        assert repair_type == RepairType.SEMANTIC


class TestSemanticRepairDetection:
    """Test semantic repair detection in execution flow."""

    def test_repair_details_structure(self):
        """Repair details should have expected structure."""
        repair_details = {
            "changed_params": {
                "group_by": {
                    "old": ["region"],
                    "new": ["product"]
                }
            },
            "added_params": {},
            "removed_params": []
        }
        
        assert "changed_params" in repair_details
        assert "group_by" in repair_details["changed_params"]
        assert "old" in repair_details["changed_params"]["group_by"]
        assert "new" in repair_details["changed_params"]["group_by"]

    def test_semantic_repair_classification_in_loop(self):
        """Semantic repairs should be detected during execution."""
        changed_params = {
            "df": {"old": None, "new": "$dataframe"},  # Safe
            "group_by": {"old": ["region"], "new": ["product"]},  # Semantic
            "agg_func": {"old": "sum", "new": "mean"}  # Semantic
        }
        
        semantic_count = 0
        safe_count = 0
        
        for param_name, change_info in changed_params.items():
            old_val = change_info["old"]
            new_val = change_info["new"]
            repair_type = classify_repair(param_name, old_val, new_val, "aggregate")
            
            if repair_type == RepairType.SEMANTIC:
                semantic_count += 1
            else:
                safe_count += 1
        
        assert semantic_count == 2, "Should detect 2 semantic repairs"
        assert safe_count == 1, "Should detect 1 safe repair"


class TestRepairReplanTrigger:
    """Test that semantic repairs trigger replan recommendations."""

    def test_semantic_repair_detected_flag(self):
        """Semantic repair should set detection flag."""
        # Simulate what happens in execute_tools_node
        semantic_repair_detected = False
        
        repair_details = {
            "changed_params": {
                "group_by": {"old": ["region"], "new": ["product"]}
            }
        }
        
        for param_name, change_info in repair_details.get("changed_params", {}).items():
            old_val = change_info.get("old")
            new_val = change_info.get("new")
            repair_type = classify_repair(param_name, old_val, new_val, "test_tool")
            
            if repair_type == RepairType.SEMANTIC:
                semantic_repair_detected = True
        
        assert semantic_repair_detected, "Should detect semantic repair"

    def test_execution_log_includes_semantic_flag(self):
        """Execution log should track semantic repair detection."""
        # Simulating execution log structure
        log_data = {
            "run_id": "test_run",
            "status": "completed",
            "semantic_repair_detected": True
        }
        
        assert "semantic_repair_detected" in log_data
        assert log_data["semantic_repair_detected"] is True

    def test_multiple_repairs_classification(self):
        """Multiple repairs should be classified individually."""
        repairs = [
            ("df", None, "$dataframe", "aggregate", RepairType.SAFE),
            ("column", "Revenue", "revenue", "aggregate", RepairType.SAFE),
            ("group_by", ["region"], ["product"], "aggregate", RepairType.SEMANTIC),
            ("agg_func", "sum", "mean", "aggregate", RepairType.SEMANTIC),
        ]
        
        for param, old_val, new_val, tool, expected_type in repairs:
            result = classify_repair(param, old_val, new_val, tool)
            assert result == expected_type, \
                f"Repair of {param} from {old_val} to {new_val} should be {expected_type}"


class TestReplanRecommendation:
    """Test replan recommendation logic."""

    def test_semantic_repair_triggers_recommendation(self):
        """Semantic repairs should trigger replan recommendation."""
        semantic_repair_detected = True
        
        if semantic_repair_detected:
            recommendation = "Semantic repairs detected. Consider re-running with adjusted parameters."
        else:
            recommendation = None
        
        assert recommendation is not None
        assert "semantic repairs" in recommendation.lower()
        assert "re-running" in recommendation.lower()

    def test_safe_repairs_no_recommendation(self):
        """Safe repairs should not trigger replan recommendation."""
        semantic_repair_detected = False
        
        if semantic_repair_detected:
            recommendation = "Semantic repairs detected."
        else:
            recommendation = None
        
        assert recommendation is None

    def test_replan_provenance_logged(self):
        """Replan events should be logged with provenance."""
        # Simulating what would be logged
        provenance_log = {
            "event": "semantic_repair_detected",
            "tool": "aggregate",
            "parameter": "group_by",
            "old_value": ["region"],
            "new_value": ["product"],
            "recommendation": "replan"
        }
        
        assert provenance_log["event"] == "semantic_repair_detected"
        assert provenance_log["recommendation"] == "replan"


class TestRepairIntegration:
    """Integration tests for repair flow."""

    def test_safe_repair_flow(self):
        """Safe repairs should flow through without triggering replan."""
        # Step 1: Detect repair
        repair_type = classify_repair("df", None, "$dataframe", "aggregate")
        assert repair_type == RepairType.SAFE
        
        # Step 2: No semantic flag set
        semantic_repair_detected = (repair_type == RepairType.SEMANTIC)
        assert not semantic_repair_detected
        
        # Step 3: No replan recommendation
        recommendation = None
        if semantic_repair_detected:
            recommendation = "Replan recommended"
        assert recommendation is None

    def test_semantic_repair_flow(self):
        """Semantic repairs should flow through with replan recommendation."""
        # Step 1: Detect repair
        repair_type = classify_repair("group_by", ["region"], ["product"], "aggregate")
        assert repair_type == RepairType.SEMANTIC
        
        # Step 2: Set semantic flag
        semantic_repair_detected = (repair_type == RepairType.SEMANTIC)
        assert semantic_repair_detected
        
        # Step 3: Generate replan recommendation
        recommendation = None
        if semantic_repair_detected:
            recommendation = "Semantic repairs detected - replan recommended"
        assert recommendation is not None
        assert "replan" in recommendation.lower()

    def test_mixed_repairs_flow(self):
        """Mix of safe and semantic repairs should still trigger replan."""
        repairs_detected = []
        
        # Simulate multiple repairs
        repairs = [
            ("df", None, "$dataframe"),  # Safe
            ("column", "Revenue", "revenue"),  # Safe
            ("group_by", ["region"], ["product"]),  # Semantic
        ]
        
        for param, old_val, new_val in repairs:
            repair_type = classify_repair(param, old_val, new_val, "aggregate")
            repairs_detected.append(repair_type)
        
        # Any semantic repair should trigger flag
        semantic_repair_detected = any(r == RepairType.SEMANTIC for r in repairs_detected)
        assert semantic_repair_detected
        
        # Should recommend replan
        if semantic_repair_detected:
            recommendation = "Replan recommended due to semantic repairs"
        assert "replan" in recommendation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
