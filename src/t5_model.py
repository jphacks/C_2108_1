from pathlib import Path
from typing import Dict, List

from transformers import T5TokenizerFast, T5ForConditionalGeneration


model_path = Path(__file__).parent.parent / 'model/t5/'
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


def make_t5_reply(input_text):
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


if __name__ == "__main__":
    reply = make_t5_reply("今日も良い天気ですね")
    print(reply)