import re
from typing import Dict, List, Optional, Union
from langchain.text_splitter import CharacterTextSplitter

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


def regex_processing(line):
    endswith = line.endswith("\n")
    line = line.strip()
    line = re.sub(r'第 \d+ 頁，', " ", line)
    line = re.sub(r'共 \d+ 頁', " ", line)
    #line = re.sub(r'「」', "", line)
    line = re.sub(r'()', "，", line)
    if endswith:
        line += " \n"
    return line
    