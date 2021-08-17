import logging

def getLogger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    logger.addHandler(console)
    logger.addHandler(handler)
    return logger
