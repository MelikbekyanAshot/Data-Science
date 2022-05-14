import torch
from transformers import BertTokenizer
from bert_classifier import BertClassifier
from functools import lru_cache
import os.path


@lru_cache(maxsize=1)
def init():
    if os.path.isfile('bert.pt'):
        global model, tokenizer
        model = BertClassifier()
        model.load_state_dict(torch.load('bert.pt', map_location=torch.device('cpu')))
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    else:
        print('Для начала работы необходимо обучить модель! Воспользуйтесь функцией train из train.py')


def single_test(text):
    test_str = [tokenizer(text,
                          padding='max_length', max_length=512, truncation=True,
                          return_tensors="pt") for text in [text]]
    mask = test_str[0]['attention_mask']
    input_id = test_str[0]['input_ids'].squeeze(1)
    output = model(input_id, mask)
    print(output)
    return 'educational' if output.argmax().item() == 0 else 'not educational'


if __name__ == '__main__':
    init()
    while True:
        print(single_test(input('Введите текст:')))
