"""options - global obsarray options"""

from typing import Any, Dict

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = ["set_options", "get_options"]


# None means "not chosen explicitly" - obsarray then uses its default behaviour
OPTIONS: Dict[str, Any] = {"unc_conventions": None}

_VALID_VALUES = {"unc_conventions": ("v0.1",)}


class set_options:
    """
    Sets obsarray global options, either for the rest of the session or within a ``with`` block

    Available options:

    * ``unc_conventions`` - version of the `UNC specification
      <https://comet-toolkit.github.io/unc_website/>`_ obsarray follows when writing uncertainty metadata.
      Currently the only supported value is ``"v0.1"`` (obsarray's current behaviour). Setting it explicitly
      also silences the :py:class:`~obsarray.conventions.UNCConventionsWarning` about the upcoming change of
      default to draft v0.2.

    .. code-block:: python

        import obsarray

        # for the rest of the session
        obsarray.set_options(unc_conventions="v0.1")

        # or only within a block
        with obsarray.set_options(unc_conventions="v0.1"):
            ds.unc["temperature"]["u_noise"] = (["x"], u_noise, {"units": "K"})

    :param kwargs: option names and their new values
    """

    def __init__(self, **kwargs: Any):
        self._old: Dict[str, Any] = {}

        for name, value in kwargs.items():
            if name not in OPTIONS:
                raise ValueError(
                    f"{name!r} is not an obsarray option, must be one of {sorted(OPTIONS)}"
                )

            if value not in _VALID_VALUES[name]:
                raise ValueError(
                    f"{value!r} is not a valid value for {name!r}, must be one of "
                    f"{list(_VALID_VALUES[name])}"
                )

            self._old[name] = OPTIONS[name]

        OPTIONS.update(kwargs)

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> None:
        OPTIONS.update(self._old)


def get_options() -> Dict[str, Any]:
    """
    Returns a copy of the current obsarray global options

    :return: option names and their values (``None`` if not set explicitly)
    """

    return dict(OPTIONS)


if __name__ == "__main__":
    pass
