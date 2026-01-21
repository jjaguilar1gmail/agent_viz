"""Basic unit tests for schema inference."""

import pandas as pd
import pytest

from autoviz_agent.tools.schema import infer_schema


def test_infer_schema_basic():
    """Test basic schema inference."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'value': [10.5, 20.3, 30.1]
    })
    
    schema = infer_schema(df)
    
    assert schema.row_count == 3
    assert len(schema.columns) == 3
    assert schema.data_shape in ['wide', 'long', 'unknown']


def test_infer_schema_with_datetime():
    """Test schema inference with datetime column."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'value': [1, 2, 3, 4, 5]
    })
    
    schema = infer_schema(df)
    
    assert schema.row_count == 5
    assert schema.data_shape == 'time_series'
    
    # Check for datetime role
    date_col = [c for c in schema.columns if c.name == 'date'][0]
    assert 'datetime' in date_col.roles


def test_infer_schema_with_missing_values():
    """Test schema inference with missing values."""
    df = pd.DataFrame({
        'col1': [1, 2, None, 4, 5],
        'col2': ['a', None, 'c', 'd', None]
    })
    
    schema = infer_schema(df)
    
    col1 = [c for c in schema.columns if c.name == 'col1'][0]
    col2 = [c for c in schema.columns if c.name == 'col2'][0]
    
    assert col1.missing_rate == 0.2  # 1 out of 5
    assert col2.missing_rate == 0.4  # 2 out of 5
