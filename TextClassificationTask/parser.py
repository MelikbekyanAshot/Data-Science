import pandas as pd
import vk_api
from tqdm import tqdm
import re
from functools import lru_cache


@lru_cache(maxsize=1)
def get_api():
    """Gives access to vk api.
    Return:
        vk api"""
    login = "" # write here login
    password = "" # write here password
    token = "" # write here token
    vk_session = vk_api.VkApi(login=login,
                              password=password,
                              token=token)
    vk_session.auth()
    return vk_session.get_api()


def get_text(wall):
    text = []
    for post in wall:
        try:
            text.append(post['text'])
        except Exception:
            pass
    return text


def process_text(text):
    """Text processing. Strip, replace special symbols and links.
    Args:
        text (list) - text.

    Returns:
        text (list) - processed text."""
    text = list(map(str.strip, text))
    text = ''.join(text)
    text = text.replace('\n', '')
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", text)
    return text.split('.')


def get_data(wall, label):
    """Convert text from vk wall to pd.DataFrame.

    Returns:
        """
    text = get_text(wall)
    processed_text = process_text(text)
    data = zip(processed_text, [label] * len(processed_text))
    df = pd.DataFrame(data=data, columns=['text', 'category'])
    df = df.dropna()
    return df


if __name__ == '__main__':
    api = get_api()

    edu_wall = api.wall.get(owner_id=-84793390, count=100)['items'] + \
               api.wall.get(domain='compscicenter', count=100)['items'] + \
               api.wall.get(domain='hexlet', count=100)['items'] + \
               api.wall.get(domain='datascience', count=100)['items']

    edu_df = get_data(edu_wall, 'edu_data')

    not_edu_wall = api.wall.get(domain='overhear', count=100)['items'] + \
                   api.wall.get(owner_id=-148881515, count=100)['items'] + \
                   api.wall.get(domain='vk', count=100)['items'] + \
                   api.wall.get(domain='psy.people', count=100)['items']

    not_edu_df = get_data(not_edu_wall, 'not_edu_data')

    full_df = edu_df.append(not_edu_df).sample(frac=1)
    full_df.to_csv('data.csv', index=False)
