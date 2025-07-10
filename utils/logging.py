import logging
import time


def get_standard_streamhandler():
    logging_fmt = '%(message)s'
    formatter = logging.Formatter(logging_fmt)
    formatter.converter = time.gmtime

    standard_streamhandler = logging.StreamHandler()
    standard_streamhandler.setFormatter(formatter)
    standard_streamhandler.setLevel(logging.INFO)
    return standard_streamhandler



if __name__ == "__main__":
    # standard_streamhandler = get_standard_streamhandler()
    # Logging commands
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_standard_streamhandler())
