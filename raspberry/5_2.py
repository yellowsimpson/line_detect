import cv2

def main():
    camera = cv2.VideoCaputre(-1)
    camera.set(3, 640)
    camera.set(4, 480)

    while(camera.isOpened()):
        _, image = camera.read()
        cv2.imshow('cmaera test', image)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
       