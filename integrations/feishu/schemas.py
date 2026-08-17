"""Feishu-facing schemas.

The canonical models live in models.schemas so REST and the integration use
the same wire format. This module keeps imports local to the integration.
"""

from models.schemas import FeishuBinding, FeishuBindingRequest, FeishuInboundMessage

__all__ = ["FeishuBinding", "FeishuBindingRequest", "FeishuInboundMessage"]
