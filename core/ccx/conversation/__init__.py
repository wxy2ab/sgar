"""Reserved namespace for ccx-side conversation extensions.

Currently empty — ccx does not yet add anything on top of
``core.cc.conversation``. The package exists so future ccx-specific
session/conversation shims have a natural home without colliding
with cc's own module path.

Per the C2 module-boundary convention, exports are gated by
``__all__``. Leaving it empty means "no public surface today" — a
deliberate choice, not an oversight; the boundary scanner in
``tests/test_module_boundaries.py`` treats absent ``__all__`` as a
violation and would flag this file if the convention were skipped.
"""

__all__: list[str] = []
