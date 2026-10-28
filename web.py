#!/usr/bin/env python3
"""Compatibility entry point for the portable Web debugger module."""

import sys

from modules.vision_tools.web_debug import app as _web_app

# Existing scripts and tests import ``web`` and set paths on that module.
# Expose the implementation object so those assignments keep working.
sys.modules[__name__] = _web_app

if __name__ == "__main__":
    _web_app.main()
