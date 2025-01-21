import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    PreTrainedTokenizerFast,
)


def unused_tokens_to_task_tokens(tokenizer: PreTrainedTokenizerFast, model_name: str):
    situation_ls = [
        "가족관계",
        "학업 및 진로",
        "학교폭력/따돌림",
        "대인관계",
        "연애,결혼,출산",
        "진로,취업,직장",
        "대인관계(부부, 자녀)",
        "재정,은퇴,노후준비",
        "건강",
        "직장, 업무 스트레스",
        "건강,죽음",
        "대인관계(노년)",
        "재정",
    ]
    disease_ls = ["만성질환 유", "만성질환 무"]
    age_ls = ["청소년", "청년", "중년", "노년"]
    gender_ls = ["남성", "여성"]
    task_token_ls = situation_ls + disease_ls + age_ls + gender_ls

    tokenizer_config = json.loads(Path("/root/tokenizer_config.json").read_text())
    tokenizer = json.loads(Path("/root/tokenizer.json").read_text())

    added_tokens = tokenizer["added_tokens"]

    unused_prefix = "unused"
    task_token_idx = 0
    for token_id, added_token in tokenizer_config["added_tokens_decoder"].items():
        if unused_prefix not in added_token["content"] or task_token_idx >= len(task_token_ls):
            continue

        task_token = f"[{task_token_ls[task_token_idx]}]"
        for idx, x in enumerate(added_tokens):
            if added_token["content"] == x["content"]:
                added_tokens[idx]["content"] = task_token
                break

        tokenizer_config["added_tokens_decoder"][token_id]["content"] = task_token
        added_token["content"] = task_token
        task_token_idx += 1
    tokenizer_config = Path("/root/tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=4)
    )
    tokenizer = Path("/root/tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False, indent=4))


def EmotionalDialogueCorpus_model_build() -> None:
    EMOTION_LS = ["분노", "슬픔", "불안", "기쁨"]
    model_name = "answerdotai/ModernBERT-base"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("/root")

    model.config.id2label = {ids: v for ids, v in enumerate(EMOTION_LS)}  # noqa: C416
    model.config.label2id = {v: ids for ids, v in enumerate(EMOTION_LS)}

    # new
    model.config.label2id = {
        **{v: ids for ids, v in enumerate(EMOTION_LS)},
        "상처": model.config.label2id["슬픔"],
        "당황": model.config.label2id["불안"],
    }

    model.config.num_labels = len(EMOTION_LS)

    model.config.save_pretrained("/root/output_dir/EmotionalModernBERT-base")
    model.save_pretrained("/root/output_dir/EmotionalModernBERT-base")
    tokenizer.save_pretrained("/root/output_dir/EmotionalModernBERT-base")

    model_name

    # unused_tokens_to_task_tokens(tokenizer, model_name)


# def EmotionalDialogueCorpus_model_build() -> None:
#     EMOTION_DICT = {
#         "E10": "분노",
#         "E11": "툴툴대는",
#         "E12": "좌절한",
#         "E13": "짜증내는",
#         "E14": "방어적인",
#         "E15": "악의적인",
#         "E16": "안달하는",
#         "E17": "구역질 나는",
#         "E18": "노여워하는",
#         "E19": "성가신",
#         "E20": "슬픔",
#         "E21": "실망한",
#         "E22": "비통한",
#         "E23": "후회되는",
#         "E24": "우울한",
#         "E25": "마비된",
#         "E26": "염세적인",
#         "E27": "눈물이 나는",
#         "E28": "낙담한",
#         "E29": "환멸을 느끼는",
#         "E30": "불안",
#         "E31": "두려운",
#         "E32": "스트레스 받는",
#         "E33": "취약한",
#         "E34": "혼란스러운",
#         "E35": "당혹스러운",
#         "E36": "회의적인",
#         "E37": "걱정스러운",
#         "E38": "조심스러운",
#         "E39": "초조한",
#         "E40": "상처",
#         "E41": "질투하는",
#         "E42": "배신당한",
#         "E43": "고립된",
#         "E44": "충격 받은",
#         "E45": "가난한, 불우한",
#         "E46": "희생된",
#         "E47": "억울한",
#         "E48": "괴로워하는",
#         "E49": "버려진",
#         "E50": "당황",
#         "E51": "고립된(당황한)",
#         "E52": "남의 시선을 의식하는",
#         "E53": "외로운",
#         "E54": "열등감",
#         "E55": "죄책감의",
#         "E56": "부끄러운",
#         "E57": "혐오스러운",
#         "E58": "한심한",
#         "E59": "혼란스러운(당황한)",
#         "E60": "기쁨",
#         "E61": "감사하는",
#         "E62": "신뢰하는",
#         "E63": "편안한",
#         "E64": "만족스러운",
#         "E65": "흥분",
#         "E66": "느긋",
#         "E67": "안도",
#         "E68": "신이 난",
#         "E69": "자신하는",
#     }
#     model_name = "answerdotai/ModernBERT-base"

#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained("/root")

#     model.config.id2label = {ids: v for ids, v in enumerate(EMOTION_DICT.values())}  # noqa: C416
#     model.config.label2id = {v: ids for ids, v in enumerate(EMOTION_DICT.values())}

#     model.config.num_labels = len(EMOTION_DICT.values())

#     model.save_pretrained("EmotionalModernBERT-base")
#     tokenizer.save_pretrained("EmotionalModernBERT-base")
#     model.push_to_hub("EmotionalModernBERT-base", private=True)
#     tokenizer.push_to_hub("EmotionalModernBERT-base", private=True)

#     # unused_tokens_to_task_tokens(tokenizer, model_name)


if "__main__" in __name__:
    EmotionalDialogueCorpus_model_build()
EMOTION_DICT = {
    "E10": "분노",
    "E11": "분노",
    "E12": "슬픔",
    "E13": "분노",
    "E14": "분노",
    "E15": "분노",
    "E16": "분노",
    "E17": "분노",
    "E18": "분노",
    "E19": "분노",
    "E20": "슬픔",
    "E21": "슬픔",
    "E22": "슬픔",
    "E23": "슬픔",
    "E24": "슬픔",
    "E25": "슬픔",
    "E26": "슬픔",
    "E27": "슬픔",
    "E28": "슬픔",
    "E29": "슬픔",
    "E30": "불안",
    "E31": "불안",
    "E32": "불안",
    "E33": "불안",
    "E34": "불안",
    "E35": "불안",
    "E36": "불안",
    "E37": "불안",
    "E38": "불안",
    "E39": "불안",
    "E40": "슬픔",
    "E41": "불안",
    "E42": "슬픔",
    "E43": "슬픔",
    "E44": "불안",
    "E45": "슬픔",
    "E46": "슬픔",
    "E47": "분노",
    "E48": "슬픔",
    "E49": "슬픔",
    "E50": "불안",
    "E51": "불안",
    "E52": "불안",
    "E53": "슬픔",
    "E54": "불안",
    "E55": "불안",
    "E56": "불안",
    "E57": "불안",
    "E58": "불안",
    "E59": "불안",
    "E60": "기쁨",
    "E61": "기쁨",
    "E62": "기쁨",
    "E63": "기쁨",
    "E64": "기쁨",
    "E65": "기쁨",
    "E66": "기쁨",
    "E67": "기쁨",
    "E68": "기쁨",
    "E69": "기쁨",
}
