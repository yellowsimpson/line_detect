import cv2
import serial
import threading
import time

class UartCommunication:
    def __init__(self, com_port="/dev/serial0", baud_rate=115200):  # com_port를 /dev/serial0로 변경
        try:
            self.ser = serial.Serial(com_port, baud_rate, timeout=2)
            if self.ser.isOpen():
                print('Serial port open!')
            else:
                print("Failed to open serial port.")
            time.sleep(1)
        except Exception as e:
            print(f"Error initializing serial port: {e}")

    def receive_data(self):
        if not hasattr(self, 'ser') or self.ser is None:
            print("Serial port is not initialized.")
            return
        
        self.ser.flushInput()
        while True:
            try:
                data = self.ser.read(1)
                if data:
                    print(f"Received data: {data}")
            except Exception as e:
                print(f"Error receiving data: {e}")

    def send_data(self, data):
        if not hasattr(self, 'ser') or self.ser is None:
            print("Serial port is not initialized.")
            return
        
        try:
            self.ser.write(data.encode('ascii'))
        except Exception as e:
            print(f"Error sending data: {e}")

    def start_receive_thread(self):
        try:
            rx_thread = threading.Thread(target=self.receive_data, name="SerialReceiveThread")
            rx_thread.daemon = True  # setDaemon(True) 대신 daemon 속성 사용
            rx_thread.start()
            print("Started RX Thread")
            time.sleep(0.05)
        except Exception as e:
            print(f"Error starting serial thread: {e}")

def process_camera_and_send_data(uart_comm):
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

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                if 95 <= cx <= 125:
                    command = "L"
                elif 39 <= cx <= 65:
                    command = "R"
                else:
                    command = "G"

                print(f"Sending command: {command}")
                uart_comm.send_data(command)
        
        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    uart_comm = UartCommunication(com_port="/dev/serial0", baud_rate=115200)  # com_port를 /dev/serial0로 변경
    uart_comm.start_receive_thread()

    process_camera_and_send_data(uart_comm)
