from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import unescape

import time
import emoji
import mojimoji
from src.arai_model import make_arai_reply
from src.t5_model import make_t5_reply


def make_reply(input_text: str) -> List[Dict[str, str]]:
    """
    replyを生成する。

    Parameters
    ----------
    input_text : str

    Returns
    -------
    outputs : List[Dict[str, str]]
    """
    input_text = replace_func(input_text)
    reply1 = make_t5_reply(input_text)
    #time.sleep(3)
    #reply2 = make_arai_reply([[input_text]])
    return reply1
    #return reply1 + reply2


def replace_func(text: str) -> str:
    text = unescape(text)
    text = replace_emoji(text)
    text = mojimoji.zen_to_han(text, kana=False)
    return text


def replace_emoji(text: str) -> str:
    return ''.join(["" if c in emoji.UNICODE_EMOJI else c for c in text])