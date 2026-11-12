"""Pin the output language of the test run to English.

Import this module before anything else that reads the environment (``conftest.py`` does).
It acts at import time, not in a fixture, so it also holds for other runners such as
``python -m unittest`` and for a developer shell with ``LANG=ko_KR.UTF-8``.
Tests of other languages pass ``--lang`` or ``monkeypatch`` explicitly.
"""

import os

os.environ["EMBED_AB_LANG"] = "en"  # highest-priority environment variable; beats LC_ALL, LC_MESSAGES, LANG
