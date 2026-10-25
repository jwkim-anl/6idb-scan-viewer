"""A window's settings as a dict, and the same dict put back.

Every viewer here is a panel of controls around a view, and what is on
screen is whatever those controls say -- which colormap, what intensity
range, which ROIs, where the slice sits. None of it can be read or
written from outside the window: the methods are parameterless by
design, because each one reads the widgets.

``state()`` is that panel as a flat dict and ``set_state()`` takes the
same dict back. Between them a display can be written down, carried to
a second window on another file, restored next week, or driven from
somewhere with no hands at all -- a script, a test, or a tool at the
other end of a socket.

Three rules hold for all five windows, and this module is what keeps
them honest:

* ``w.set_state(w.state())`` changes nothing. So ``state()`` reports
  everything settable and nothing else; what the window worked out for
  itself -- the file open, the frame count, the bounds of the map --
  goes under the read-only ``"about"`` key, which :func:`apply` skips.
* A dict may be partial. What it does not mention is left alone, so
  ``set_state({"colormap": "magma"})`` is a one-line way to say one
  thing.
* A key no window knows is an error naming the ones it does. A setting
  silently dropped because of a typo is a long afternoon.
"""

from __future__ import annotations

#: What :meth:`state` calls the part of itself that cannot be set back.
ABOUT = "about"


def apply(state, setters, known=None):
    """Call *setters* for the keys *state* holds, in the setters' order.

    *setters* maps a setting's name to a one-argument callable, and its
    order is the order they are applied in -- which matters more than it
    looks: a range has to be typed before the "auto" tick that would
    overwrite it, a plane's normal before the offset measured along it.
    Write the mapping in the order the window wants and the caller never
    has to think about it, whatever order their dict happens to be in.

    *known* names every setting the window has, for a window that
    applies its dict in two goes with work in between -- a conversion to
    re-run between the settings that decide it and the ones that only
    say how to draw the result. Each go then checks the dict against the
    whole list rather than rejecting the other half of it.

    Returns the names actually set, which is how a caller tells a
    setting that was applied from one that was not in the dict.
    """
    if not isinstance(state, dict):
        raise TypeError(
            f"a state is a dict of settings, not {type(state).__name__}")
    names = setters if known is None else known
    unknown = [name for name in state if name not in names and name != ABOUT]
    if unknown:
        raise ValueError(
            f"no such setting: {', '.join(sorted(unknown))}. "
            f"Settings: {', '.join(names)}"
        )
    done = []
    for name, setter in setters.items():
        if name in state:
            setter(state[name])
            done.append(name)
    return done


def choose(combo, value, what):
    """Put *combo* on *value*, or say what it could have been instead.

    ``setCurrentText`` on a non-editable combo box is a no-op for a name
    that is not in it, which would make a misspelt colormap look like it
    worked.
    """
    text = str(value)
    if combo.findText(text) < 0:
        choices = ", ".join(combo.itemText(i) for i in range(combo.count()))
        raise ValueError(f"no {what} called {text!r}. Choices: {choices}")
    combo.setCurrentIndex(combo.findText(text))


def within(spin, value, what):
    """Set a spin box to a whole number, or say so rather than clamp.

    ``setValue`` outside the range quietly gives the nearest end, which
    for a frame number or a point budget is a wrong answer that looks
    like it worked.
    """
    number = int(value)
    low, high = int(spin.minimum()), int(spin.maximum())
    if not low <= number <= high:
        raise ValueError(f"{what} must be {low:,}..{high:,}; got {number:,}")
    spin.setValue(number)


def triple(value, what):
    """Three floats from whatever a caller wrote them as."""
    values = [float(v) for v in value]
    if len(values) != 3:
        raise ValueError(f"{what} takes three numbers, H K L; got {len(values)}")
    return values


def pair(value, what):
    """A low, high pair -- the shape every range in these windows takes."""
    values = [float(v) for v in value]
    if len(values) != 2:
        raise ValueError(f"{what} takes two numbers, low and high; "
                         f"got {len(values)}")
    return values
