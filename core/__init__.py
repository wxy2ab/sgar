def str_to_obj(*args, **kwargs):
    from .utils.common import str_to_obj as _str_to_obj

    return _str_to_obj(*args, **kwargs)


__all__ = ["str_to_obj"]
