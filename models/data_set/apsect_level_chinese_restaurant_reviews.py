import torch.utils.data as D
import json
from torch.utils.data import Dataset


class TextDataSet(Dataset):

    def __init__(self, train=True, columns=["content_indexes"]):
        json_file = "/home/sqzhang/SA-Models/models/word_segmentation/ltp/train.json" if train \
            else "/home/sqzhang/SA-Models/models/word_segmentation/ltp/validationset.json"
        # json_file = "/Users/zhixuan/project/learn/sentiment-analysis/SA-Models/models/word_segmentation/ltp/train.json" if train \
        #     else "/Users/zhixuan/project/learn/sentiment-analysis/SA-Models/models/word_segmentation/ltp/validationset.json"
        print(json_file)

        dict_datas = json.loads(open(json_file).readline())
        self.datas = []
        for i in range(len(dict_datas)):
            line = []
            for column in columns:
                line.append(dict_datas[str(i)][column])

            if int(line[1]) == -2:
                continue
            self.datas.append(line)

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    train_loader = D.DataLoader(TextDataSet(train=False),
                                batch_size=2, shuffle=True, num_workers=1
                                )

    for i, data in enumerate(train_loader):
        print(data)
