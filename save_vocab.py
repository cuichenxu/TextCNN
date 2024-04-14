FILE_PATHS = {
    "class": "path of class.txt",
    "dev": "path of dev.txt",
    "train": "path of train.txt",
    "test": "path of test.txt",
}

# 读取类别
def get_class(file_path=FILE_PATHS["class"]):
    with open(file_path, 'r') as file:
        classes = file.read()
    classes = classes.splitlines()
    class_to_idx = {}
    for i, cla in enumerate(classes):
        class_to_idx[cla] = i
    return classes, class_to_idx

# 读取数据，返回x和y
def get_x_and_y(file_type, file_paths = FILE_PATHS):
    with open(file_paths[file_type], 'r') as file:
        datas = file.read()
    datas = datas.splitlines()
    x = []
    y = []
    for data in datas:
        data = data.split('\t')
        x.append(list(data[0].replace(' ', '')))
        y.append(data[1])
    return x, y

# 读取所有的数据，统计词频，保存词表
from collections import Counter
def save_vocab(vocab_path):
    contents = []
    for file_type in ["train", "dev", "test"]:
        x, _ = get_x_and_y(file_type)
        for i in x:
            contents += list(i)
    counter = Counter(contents)
    count_pairs = counter.most_common(9999)
    words, _ = list(zip(*count_pairs))
    words = ['<pad>'] + list(words)
    with open(vocab_path, 'w') as file:
        file.write('\n'.join(words))

