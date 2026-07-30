"""Unit tests for Data Lake query engine."""

from src.data.lake import DataLake


class TestLake:
    def test_build_where(self):
        lake = DataLake()
        assert "WHERE timestamp >= 100 AND timestamp <= 200" == lake._build_where(100, 200)
        assert "WHERE timestamp >= 100" == lake._build_where(100, None)
        assert "WHERE timestamp <= 200" == lake._build_where(None, 200)
        assert "" == lake._build_where(None, None)

    def test_build_where_filters(self):
        lake = DataLake()
        where = lake._build_where(1700000000000, 1700086400000)
        assert "1700000000000" in where
        assert "1700086400000" in where
