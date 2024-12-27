import cv2 as cv
web_cam = cv.VideoCapture(0)

while True:
    isTrue, frame = web_cam.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20)&0xFF == ord('f'):
        break
web_cam.release()
cv.destroyAllWindows()