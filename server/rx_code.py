'''
(카메라수신)
송신하고 있는 데이터를 수신하기 위한 코드
앞으로 라즈베리파이와 컴퓨터에서 실시간 정보 공유를 하기위한 코드
이 코드는 한명밖에 접속 못함
'''
import asyncio
import websockets
import cv2
import numpy as np

async def receive_data():
    async with websockets.connect('ws://172.30.1.59:8765') as websocket: #송신하는 컴퓨터의 ip주소를 적어줘
        while True:
            data = await websocket.recv()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imshow('Received Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

asyncio.get_event_loop().run_until_complete(receive_data())