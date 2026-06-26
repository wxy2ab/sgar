"""Path-token extraction — thin re-export.

The implementation lives in
``core.cc.conversation.strategy_common`` so cc and ccx share one
source of truth. cc uses it to assemble its "Paths in this task" prompt
block; ccx uses it for the same purpose plus focused-subtree
expansion in doc / ask mode runners.
"""

from __future__ import annotations

from core.cc.conversation.strategy_common import extract_path_tokens

__all__ = ["extract_path_tokens"]
