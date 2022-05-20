import os
import shutil

from mdutils.mdutils import MdUtils

import util.utils

mdFile = None
mdFilename = None
mdPath = None


def set_mdPath(root_path):
    global mdPath
    mdPath = root_path


def get_mdPath():
    global mdPath
    return mdPath


def create_report(filename, title=""):
    global mdPath
    global mdFilename
    mdFilename = filename
    global mdFile
    mdFile = MdUtils(file_name=mdFilename, title=title)

    mdFile.new_header(level=1, title='Overview')


def write_to_report(string):
    global mdFile
    mdFile.new_line(string)
    pass


def header2(headertext: str):
    global mdFile
    mdFile.new_header(level=2, title=headertext)


def header1(headertext: str):
    global mdFile
    mdFile.new_header(level=1, title=headertext)


def image_to_report(path, title, text=""):
    global mdFile
    mdFile.new_header(level=2, title=title)
    mdFile.new_line(f"![Alt text]({path}?raw=true \"Title\")")
    mdFile.new_paragraph(text)
    pass


def save_report():
    global mdFile
    global mdFilename
    global mdPath

    mdFile.create_md_file()
    shutil.copyfile(f"./{mdFilename}.md", f"./{mdPath}/{mdFilename}.md", follow_symlinks=True)
    os.remove(f"./{mdFilename}.md")


if __name__ == '__main__':
    print("testing to report file")

    test_path = "out/test_report"

    util.utils.create_folder(test_path)

    create_report("test_report_markdown")
    write_to_report("test")
    image_to_report("tsne_otsu.png", "some image", text="this is some example text for the paragraph")
    save_report()
