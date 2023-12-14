import re

unknown_unicode_list = ["﻿", '□', '▪', 'О', 'н', 'л', 'Л', 'ю', 'б', 'В', '这', '么', 'з', 'Ω', 'Я', 'К', 'я', '≥', 'ж', 
                        '\ue6f0', '打', '电', '话', '叫', '医', '生', 'ซ', 'ฤ', 'ะ', 'ล', 'บ', 'ร', 'ด', 'น', 'จ', 'ม', 
                        'เ', 'ย', '💡', 'ุ', 'โ', 'ิ', 'ต', 'ป', 'ก', 'Р', 'ว', 'ฟ', 'ช', 'х', 'ท', '♣', '\ue607', '\ue608', 
                        'ง', 'พ', '⌛', 'М', 'С', 'Х', 'р', 'ц', 'Ж', 'З', 'И', 'र', 'ो', 'ट', 'ी', '\uf4c5', '오', '스', 
                        '어', '♡', '≧', '\ue609', '₩', '♴', 'ไ', '中', '文', '►', '國', '傳', '統', 'А', '∥', '⊥', '\uf107', 
                        '\ue80c', 'Λ', 'ά', 'ο', 'ς', '㎜', 'Ⅱ', 'س', 'ͧ', 'ǹ', 'ŧ', 'շ', 'ͺ', '֧', 'प', 'Ф', 'Φ', '➜', '金', 
                        '融', '经', '济', '司', '法', '组', '版', '权', '申', '明', '隐', '私', '政', '策', '\uf1f1', '\uf1e6', 
                        '国', '简', '体', 'แ', '⬇', '⬆', 'ъ', 'ข', 'า', '⇓', 'ラ', 'ー', 'オ', '語', '่', 'อ', 'ف', 'م', 'ك', 
                        'ب', 'ي', 'ر', 'ั', 'ธ', 'ส', 'ี', 'ำ', 'Г', '▲', 'ج', 'ن', 'و', 'ه', 'ا', '佳', '作', '汇', '编', 'П', 
                        'щ', '้', 'أ', 'ت', 'ل', 'ス', '王', 'ค']

def remove_unknown_unicode(s):
    for unknown_char in unknown_unicode_list:
        s = s.replace(unknown_char, "")
    return s

def remove_quot(s):
    s = s.replace("&quot;", "")
    s = s.replace("& quot;", "")
    return s

def remove_url(s):
    url_clean = re.compile(r"https?://(?:www\.)?[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})+(?:[/?].*)?")
    s = url_clean.sub(r'',s)
    return s

def clean_data(path):
    # Cleaning Lao file
    with open(path + ".lo", mode="r", encoding="utf8") as lao_read:
        with open(path + "_clean.lo", mode="w+", encoding="utf8") as lao_write:
            for line in lao_read:
                line = line[:-1]
                line = remove_unknown_unicode(line)
                line = remove_quot(line)
                line = remove_url(line)
                lao_write.writelines(line + "\n")
    
    # Cleaning Viet file
    with open(path + ".vi", mode="r", encoding="utf8") as viet_read:
        with open(path + "_clean.vi", mode="w+", encoding="utf8") as viet_write:
            for line in viet_read:
                line = line[:-1]
                line = remove_unknown_unicode(line)
                line = remove_quot(line)
                line = remove_url(line)
                viet_write.writelines(line + "\n")

train_path = "data_processing/data/Train/train2023"
valid_path = "data_processing/data/Dev/dev2023"
test_path = "data_processing/data/Test/test2023"

clean_data(train_path)
clean_data(valid_path)
clean_data(test_path)



