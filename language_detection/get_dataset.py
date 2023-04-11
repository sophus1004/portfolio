# pip install wikipedia
# pip install nltk

import re
import pandas as pd
import random
import wikipedia
import warnings
import nltk
nltk.download('punkt')  # 문장 분리 라이브러리입니다.
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

get_p = 5 # 한 언어에서 불러오는 위키피디아 페이지 수입니다.
get_s = 10 # 한 언어에서 불러오는 문장 수입니다.
dt_save = 'data/train_set.xlsx' # 데이터셋 저장 결로 입니다.

# wikipedia 라이브러리에서 램덤으로 페이지 텍스트를 가져오는 함수입니다.
def get_wikipedia_text(lang, num_pages):
    txt = ''
    wikipedia.set_lang(lang)  # 가져올 페이지의 언어를 지정합니다.
    for count in range(num_pages):  # 가져올 페이지의 장수를 지정합니다.
        while True:
            try:
                add_txt = wikipedia.page(wikipedia.random(pages=1))
                break

            # 랜덤으로 불러오는 타이틀이 위키에 없는 경우 찾을 때까지 계속합니다.
            except:
                pass
        txt = txt + ' ' + add_txt.content.replace("\n", "").replace("  ", "")  # 문자열로 처리한 뒤에 출력합니다.
    return txt


# 데이터 프레임을 선언합니다.
data_set = pd.DataFrame(columns=['text', 'lang_code'])

# 데이터셋에 필요한 언어
language = ['id', 'ja', 'ko', 'tl', 'zh']
## language = ['ar', 'de', 'en', 'es', 'fr', 'id', 'ja', 'ko', 'ru', 'tl', 'uz', 'vi', 'zh']

for lang in language:
    print('처리 중인 언어:   ', lang.replace('zh', 'zh-cn'))

    wiki_sen = get_wikipedia_text(lang=lang, num_pages=get_p)  # for에서 지정되는 언어와 몇 페이지에서 텍스트를 추출할지 정합니다.

    # 일본어와 중국어 문장장 분리 정규분포를 사용합니다.
    if lang == 'ja' or lang == 'zh':
        sentences = re.findall('.+?[。！？]', wiki_sen.replace(' ', ''))

    # 나머지 언어들은 nltk.sent_tokenize()로 문장을 분리합니다.
    else:
        sentences = nltk.sent_tokenize(wiki_sen)

    # 어떤 문장을 가져올지 랜덤으로 정합니다.
    # 페이지에서 불러온 문장이 충분하지 않으면 랜덤 선택에서 에러가 발생합니다.
    random_sentences = random.sample(sentences, k=get_s)

    # 가져온 문장은 데이터 프레임에 입력합니다.
    for sen in random_sentences:
        add_row = pd.DataFrame({'text': [str(sen)], 'label': [lang.replace('zh', 'zh-cn')]})
        data_set = pd.concat([data_set, add_row], axis=0)

data_set.index = range(len(data_set))
data_set.to_excel(dt_save)