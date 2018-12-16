import os

from pyltp import Segmentor
import pandas as pd
from pandas._libs import json

from src.const import DATA_SET_DIR, LTP_DATA_DIR
import numpy as np

train_file_path = "trainingset/sentiment_analysis_trainingset.csv"
validationset_file_path = "validationset/sentiment_analysis_validationset.csv"


def main():
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    # train_file = os.path.join(DATA_SET_DIR, train_file_path)
    words_dict_json = open("words.json", "r")
    words_dict = json.loads(words_dict_json.readline())
    print(words_dict["味道"])

    validationset_file = os.path.join(DATA_SET_DIR, validationset_file_path)

    lines = pd.read_csv(validationset_file,
                        header=0,
                        dtype={"id": np.int, "content": np.str, "location_traffic_convenience": np.int,
                               "location_distance_from_business_district": np.int,
                               "location_easy_to_find": np.int, "service_wait_time": np.int,
                               "service_waiters_attitude": np.int, "service_parking_convenience": np.int,
                               "service_serving_speed": np.int, "price_level": np.int,
                               "price_cost_effective": np.int, "price_discount": np.int,
                               "environment_decoration": np.int, "environment_noise": np.int,
                               "environment_space": np.int, "environment_cleaness": np.int,
                               "dish_portion": np.int, "dish_taste": np.int, "dish_look": np.int,
                               "dish_recommendation": np.int, "others_overall_experience": np.int,
                               "others_willing_to_consume_again": np.int
                               },
                        index_col="id")

    lines_dict = {}
    for index in lines.index:
        line = lines.loc[index]
        line_dict = {}
        for column in line.index:
            line_dict[column] = line[column]

        line_dict["content"] = line["content"][1 if line["content"][0] == "\"" else 0:
                                               -1 if line["content"][-1] == "\"" else len(line["content"])]

        line_dict["content_indexes"] = " ".join(
            [str(words_dict[word]["index"]) for word in segmentor.segment(line_dict["content"])])

        lines_dict[index] = line_dict

    line_dict_json = json.dumps(lines_dict)

    json_file = open("validationset.json", "w")
    json_file.write(line_dict_json)
    json_file.flush()
    json_file.close()


def regist_words(word_dict, words):
    for word in words:
        if word in word_dict:
            word_dict[word]["num"] += 1
        else:
            word_dict[word] = {
                "num": 1,
                "index": word_dict[""]["index"]
            }
            word_dict[""]["index"] += 1


if __name__ == '__main__':
    main()

