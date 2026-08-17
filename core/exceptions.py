class RecoverableRunError(RuntimeError):
    """The current agent process stopped, but the persisted Run may resume."""

