# This file splits the dataset based on the existence of Roman characters in the sentence,
# and combines Laotian and Vietnamese sentences into a single file

# Only run this after clean data has been generated

# Building the Roman alphabet
roman_alphabet = ['Ă', 'Ằ', 'Ắ', 'Ẳ', 'Ẵ', 'Ặ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 
                  'Â', 'Ầ', 'Ấ', 'Ẩ', 'Ẫ', 'Ậ', 'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ',
                  'Ê', 'Ề', 'Ế', 'Ể', 'Ễ', 'Ệ', 'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ',
                  'Ô', 'Ồ', 'Ố', 'Ổ', 'Ỗ', 'Ộ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
                  'Ơ', 'Ờ', 'Ớ', 'Ở', 'Ỡ', 'Ợ', 'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ',
                  'Ư', 'Ừ', 'Ứ', 'Ử', 'Ữ', 'Ự', 'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự',
                  'À', 'Á', 'Ả', 'Ã', 'Ạ', 'à', 'á', 'ả', 'ã', 'ạ',
                  'È', 'É', 'Ẻ', 'Ẽ', 'Ẹ', 'è', 'é', 'ẻ', 'ẽ', 'ẹ',
                  'Ì', 'Í', 'Ỉ', 'Ĩ', 'Ị', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
                  'Ò', 'Ó', 'Ỏ', 'Õ', 'Ọ', 'ò', 'ó', 'ỏ', 'õ', 'ọ',
                  'ù', 'ú', 'ủ', 'ũ', 'ụ', 'Ù', 'Ú', 'Ủ', 'Ũ', 'Ụ',
                  'đ', 'Đ']

for i in range(26):
    roman_alphabet.insert(0, chr(ord('a') + i))
    roman_alphabet.insert(0, chr(ord('A') + i))

def roman_split(path):
    count = 0
    with open(path + "_clean.lo", mode="r", encoding="utf8") as lao_read:
        with open(path + "_clean.vi", mode="r", encoding="utf8") as vie_read:
            with open(path + "_roman.dat", mode="w+", encoding="utf8") as roman_write:
                with open(path + "_laotian.dat", mode="w+", encoding="utf8") as laotian_write:
                    for lao_line in lao_read:
                        count += 1
                        if count % 500 == 250:
                            print(f"Line {count} reached!")

                        lao_line = lao_line[:-1]
                        vie_line = vie_read.readline()[:-1]

                        for char in roman_alphabet:
                            if char in lao_line:
                                line = lao_line + "\t" + vie_line + "\n"
                                roman_write.writelines(line)
                                break
                        else:
                            line = lao_line + "\t" + vie_line + "\n"
                            laotian_write.writelines(line)

train_path = "data_processing/data/Train/train2023"
dev_path = "data_processing/data/Dev/dev2023"

roman_split(train_path)
roman_split(dev_path)



