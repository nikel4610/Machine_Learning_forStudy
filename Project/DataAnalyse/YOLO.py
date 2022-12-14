import cv2
import numpy as np
import pandas as pd
import os
import pickle

from sklearn.preprocessing import LabelEncoder
ingre_list = []

def yolo(frame, size, score_threshold, nms_threshold):
    """YOLO 시작"""
    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet("D:/vsc_project/machinelearning_study/yolofiles/yolov4-obj_final.weights",
                          "D:/vsc_project/machinelearning_study/yolofiles/yolov4-obj.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # 각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)

    # 후보 박스 중 선택된 박스의 인덱스 출력
    print(f"indexes: ", end='')
    for index in indexes:
        print(index, end=' ')
    print("\n\n============================== classes ==============================")

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]

            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            # 탐지된 객체의 정보 출력
            print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")
            globals()['ingre_list'].append(class_name)

    return frame


classes = ['Enemy',
           'Soy sauce',
           'Potato',
           'Eggs',
           'sweet potato',
           'chili',
           'Kimchi',
           'Green Onion',
           'Pork',
           'Garlic',
           'Radish',
           'Soybean paste',
           'Pear',
           'Cabbage',
           'Peach',
           'Pimento',
           'apple',
           'Lettuce',
           'Spam',
           'Onion',
           'Cucumber',
           'Rice']

# 이미지 경로
office = "D:/vsc_project/machinelearning_study/yolofiles/test.jpg"

# 이미지 읽어오기
frame = cv2.imread(office)

# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
size_list = [320, 416, 608]

frame = yolo(frame=frame, size=size_list[2], score_threshold=0.4, nms_threshold=0.4)
cv2.imshow("Output_Yolo", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

words = {'Soy sauce': '간장', 'Potato': '감자', 'Eggs': '계란', 'sweet potato': '고구마', 'chili': '고추', 'Kimchi': '김치', 'Green Onion': '대파', 'Pork': '돼지고기', 'Garlic': '마늘', 'Radish': '무', 'Soybean paste': '된장', 'Pear': '배', 'Cabbage': '양배추', 'Peach': '복숭아', 'Pimento': '피망', 'apple': '사과', 'Lettuce': '양상추', 'Spam': '스팸', 'Onion': '양파', 'Cucumber': '오이', 'Rice': '햇반'}

path = 'D:/vsc_project/machinelearning_study/Project/searchData'
df1 = pd.read_csv(os.path.join(path, 'rcp.csv'), encoding='cp949')

le = LabelEncoder()
df1['CKG_NM2'] = le.fit_transform(df1['CKG_NM'])
df1_le = df1[['CKG_NM2', 'CKG_MTRL_CN']]

# print(df1)
# CKG_MTRL_CN

for i in range(len(ingre_list)):
    for key, value in words.items():
        if ingre_list[i] == key:
            ingre_list[i] = value

# ingre_list 중복 제거
ingre_list = list(set(ingre_list))
# print(ingre_list)

# ingre_list가 포함된 CKG_MTRL_CN 추출
df2 = df1_le[df1_le['CKG_MTRL_CN'].str.contains('|'.join(ingre_list))]
df2.rename(columns={'CKG_NM2': 'CKG_NM'}, inplace=True)
print(df2)

# lgbm_t.pkl 파일 불러오기
with open('D:/vsc_project/machinelearning_study/Project/lgbm_t.pkl', 'rb') as f:
    lgbm_t = pickle.load(f)

# print(lgbm_t)

Gender = 1
Age = 20
Temperature = 20.0
Precipitation = 0.0
Humidity = 0.0
Cloud = 0.0
Month = 12
Season = 4
Weekday = 5

# print(df2)

model = lgbm_t

df2 = df2['CKG_NM']
df2 = df2.values.reshape(-1, 1)
# print(df2)

df3 = pd.DataFrame(columns=['RCP_NM', 'score'])

for i in range(len(df2)):
    input_data = [df2[i][0], Gender, Age, Temperature, Precipitation, Humidity, Cloud, Month, Season, Weekday]
    # print(input_data)
    # print(model.predict([input_data]))
    # df3에 df2의 CKG_NM과 model.predict([input_data])의 예측값을 전부 저장
    df3 = df3.append({'RCP_NM': df2[i][0], 'score': model.predict([input_data])}, ignore_index=True)

# df3의 RCP_NM이 df1의 CKG_NM2가 같으면 CKG_NM을 RCP_NM으로 저장
df3['RCP_NM'] = df3['RCP_NM'].map(df1.set_index('CKG_NM2')['CKG_NM'])
# df3의 score를 내림차순으로 정렬
df3 = df3.sort_values(by='score', ascending=False)
print(df3.head())


