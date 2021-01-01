import argparse
import logging

from config import LOGGING_LEVEL, LOGGING_FORMAT

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

if __name__ == '__main__':
    pass


