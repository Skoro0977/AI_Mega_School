import logging

logger = logging.getLogger("uvicorn")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)