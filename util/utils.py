import os
import logging
from util.base_logger import logger

root = None

def set_root(_root):
    global root
    root = _root

def get_root():
    global root
    return root

def create_folder(path):

    path_list = str(path).split("/")
    logger.debug(path_list)

    concat_path = "."

    for dir in path_list:
        concat_path = f"{concat_path}/{dir}"
        if (os.path.isdir(f"{concat_path}")):
            logger.warning(f"{concat_path} already exists")
        else:
            os.mkdir(f'{concat_path}')
            logger.info(f"Created folder {concat_path}")