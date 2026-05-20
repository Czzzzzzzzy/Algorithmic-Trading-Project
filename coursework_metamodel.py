#!/usr/bin/env python3
"""Compatibility wrapper for the refactored BUSI70575 coursework pipeline."""

from coursework.coursework_metamodel import *  # noqa: F401,F403
from coursework.src.config import *  # noqa: F401,F403
from coursework.src.evaluation import *  # noqa: F401,F403
from coursework.src.importance import *  # noqa: F401,F403
from coursework.src.models import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
