
import logging
import sys
import time
from pathlib import Path

from _pytest.capture import CaptureFixture

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from helpers import DLOG, PrjPaths


def test_dlog_init_on_import_helpers():
    logLevel = logging.DEBUG

    assert DLOG.name == "LTNApp"
    assert DLOG.level == logLevel
    assert DLOG.hasHandlers()

    file_handlers = [h for h in DLOG.handlers if isinstance(h, logging.FileHandler)]
    stream_handlers = [h for h in DLOG.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]

    assert len(file_handlers) >= 1
    assert len(stream_handlers) >= 1

    fileHandler = file_handlers[0]
    assert Path(fileHandler.baseFilename).resolve() == (PrjPaths.logs / Path("debug.log")).resolve()
    assert fileHandler.formatter is not None
    assert fileHandler.formatter._fmt == "%(asctime)s | %(levelname)s | %(message)s"
    assert fileHandler.encoding == "utf-8"
    assert fileHandler.level == logLevel

    streamHandler = stream_handlers[0]
    assert streamHandler.stream is sys.stderr
    assert streamHandler.formatter is not None
    assert streamHandler.formatter._fmt == "%(asctime)s | %(levelname)s | %(message)s"
    assert streamHandler.level == logLevel


def test_dlog_levels(capsys: CaptureFixture[str]):
    capsys.readouterr()  # Clear any initial output before test loop

    log_levels = [
        ("debug", logging.DEBUG, "DEBUG"),
        ("info", logging.INFO, "INFO"),
        ("warning", logging.WARNING, "WARNING"),
        ("error", logging.ERROR, "ERROR"),
    ]

    for method, level_int, level_str in log_levels:
        logMessage = f"pytest: test DLOG {method} log"

        log_func = getattr(DLOG, method)
        log_func(logMessage)

        # Allow time for writing to file
        time.sleep(0.05)
        debug_log_path = PrjPaths.logs / Path("debug.log")
        assert debug_log_path.exists()
        with open(debug_log_path, "r", encoding="utf-8") as f:
            logContent = f.read()
            assert f"{level_str} | {logMessage}" in logContent

        captured = capsys.readouterr()
        assert logMessage in captured.err
