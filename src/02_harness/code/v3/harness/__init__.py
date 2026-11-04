"""Harness v3 (research/log/2026-10-31-0150.md): the agent loop, the environment kinds and the reference policy as one
package. Every module imports the others as harness.<module>, so each is loaded once; the entry scripts next to this
package (run.py, golden.py, collect.py, envtool.py) only call into it."""
