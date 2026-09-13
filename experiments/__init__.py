"""Experiment scaffolding (§15: the durable, allowlisted entries only).

Production code never imports this package; this package may import
production modules (core/, sports/). Experiment outputs are printed to
the run log — never written to data_delivery/, never committed.
"""
