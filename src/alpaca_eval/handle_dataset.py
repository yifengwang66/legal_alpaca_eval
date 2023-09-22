import pandas as pd
import json


def get_reference_outputs():
    # 读取CSV文件
    df = pd.read_csv('legal_dataset.csv')
    df.fillna('', inplace=True)

    # 通过列名获取数据
    questions = df['题目']
    answer = df['标准答案']
    glm_answer = df["glm"]
    baichan_answer = df["百川"]
    beidachatlaw_answer = df["北大chatlaw"]
    tsz_360_answer = df["360"]
    tongyi_answer = df["通义"]

    outputs = []

    legal_model_outputs_noApi = []

    for i in range(len(questions)):
        outputs.append({
            "instruction": questions[i],
            "output": answer[i]
        })
        legal_model_outputs_noApi.append({
            "generator": 'glm',
            "instruction": questions[i],
            "output": glm_answer[i]
        })
        legal_model_outputs_noApi.append({
            "generator": '百川',
            "instruction": questions[i],
            "output": baichan_answer[i]
        })
        # legal_model_outputs_noApi.append({
        #     "generator": "Peking_chatlaw",
        #     "instruction": questions[i],
        #     "output": beidachatlaw_answer[i]
        # })

        legal_model_outputs_noApi.append({
            "generator": '360',
            "instruction": questions[i],
            "output": tsz_360_answer[i]
        })

        legal_model_outputs_noApi.append({
            "generator": '通义',
            "instruction": questions[i],
            "output": tongyi_answer[i]
        })

    print("questions:", questions)
    print("answer:", answer)
    print("outputs:", outputs)
    print("legal_model_outputs_noApi:", legal_model_outputs_noApi)

    with open('legal_reference_outputs.json', 'w') as f:
        json.dump(outputs, f)

    with open('legal_model_outputs_noApi.json', 'w') as f:
        json.dump(legal_model_outputs_noApi, f)


if __name__ == '__main__':
    get_reference_outputs()