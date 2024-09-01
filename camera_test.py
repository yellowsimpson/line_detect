import cv2

def test_camera():
    # 카메라 열기
    camera = cv2.VideoCapture(0)

    # 카메라 열기 실패 확인
    if not camera.isOpened():
        print("Camera not found or not accessible.")
        return

    # 카메라 설정 (해상도 설정)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    print("Press 'q' to quit")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 원본 프레임 표시
        cv2.imshow('Camera Test', frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제 및 윈도우 종료
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_camera()
