"""
Debug pass: prints current process RSS. Inserted at multiple positions
in the pipeline to track memory usage pass-by-pass.
"""
import os
import subprocess

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _current_rss_mb() -> float:
    try:
        r = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            capture_output=True, text=True,
        )
        return int(r.stdout.strip()) / 1024
    except Exception:
        return 0.0


@register_pass(namespace="common")
class debug_rss(AbstractGraphPass):
    """No-op pass that prints current process RSS to stdout."""

    _pass_counter: int = 0

    def apply(self, prog):
        debug_rss._pass_counter += 1
        rss = _current_rss_mb()
        print(f"  [debug_rss #{debug_rss._pass_counter}] RSS={rss:.0f} MB",
              flush=True)
