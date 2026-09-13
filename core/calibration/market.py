"""Market calibration (§2 layout).

Derived-market calibration (margin/spread/run-line, totals, push/tie
mass) consumes the coherent joint score distribution produced by each
sport's market model. The implementation lands with the Phase-r7 market
contract completion (spec §13/§20); this module is declared now so the
§2 tree is complete and so market-calibration consumers have a single
canonical import path.
"""
