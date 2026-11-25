"""Compatibility alias for the shared equipment handle-fitting utilities."""

import sys

from .. import handlefit as _equipment_handlefit

sys.modules[__name__] = _equipment_handlefit
