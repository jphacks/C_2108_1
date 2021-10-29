from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import unescape

import emoji
import mojimoji
from transformers import T5TokenizerFast, T5ForConditionalGeneration


model_path = Path(__file__).parent / 'model/'
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


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
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(
        input_ids, 
        max_length=20,
        num_beams=4,
        repetition_penalty=1.2, 
        num_return_sequences=3, 
        num_beam_groups=4,
        diversity_penalty=3.0
    )
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [{"reply_text": output} for output in outputs]


def replace_func(text: str) -> str:
    text = unescape(text)
    text = replace_emoji(text)
    text = mojimoji.zen_to_han(text, kana=False)
    return text


def replace_emoji(text: str) -> str:
    return ''.join(["" if c in emoji.UNICODE_EMOJI else c for c in text])