import cv2

def process_camera_and_display():
    # 카메라 열기
    camera = cv2.VideoCapture(0)

    # 카메라 설정 (해상도 설정)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 원본 프레임 표시
        cv2.imshow('Original Frame', frame)

        # 이미지 처리
        crop_img = frame[60:120, 0:160]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)

        mask = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('Processed Frame', mask)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제 및 윈도우 종료
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_camera_and_display()
