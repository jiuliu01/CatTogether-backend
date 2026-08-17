"""Feishu transport for CatTogether.

Modules are intentionally not imported eagerly here because the event handler
and task queue reference each other through their singleton instances.
"""
