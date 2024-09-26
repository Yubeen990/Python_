import cv2
import numpy as np
import pytesseract
from common.image import img2gray, resize, img_show, find_bounding_rects, visualize_rects
from deep_convnet import DeepConvNet
from neuralnet_mnist import predict

# Tesseract OCR 설치 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

network = DeepConvNet()
network.load_params("params.pkl")

# 웹캠 열기
cap = cv2.VideoCapture(0)
#rects = find_bounding_rects(frame, min_size=(10, 10)) # 이미지에서 숫자 영역(contour)들을 찾아내기


while True:
    # 웹캠 프레임 읽기
    ret, frame = cap.read()
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 이미지 이진화
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 숫자 인식
    numbers = pytesseract.image_to_string(threshold, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    
    # 숫자 영역 검출
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # for rect in rects
    for contour in contours:
        # 윤곽선 영역 크기 필터링
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue
            rect_img = gray[y:y+h, x:x+w]
            resized_img = resize(rect_img, dsize=(28, 28)).reshape(784)
            prediction = np.argmax(predict(network, resized_img / 255.0))
        # 숫자 영역에 사각형 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 인식된 숫자 표시
            cv2.putText(frame, numbers, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 프레임 출력
    cv2.imshow('Webcam', frame)
    
    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠과 윈도우 창 종료
cap.release()
cv2.destroyAllWindows()
