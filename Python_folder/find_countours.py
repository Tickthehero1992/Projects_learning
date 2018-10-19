import cv2

def Show_image(filename):
    img = cv2.imread(filename)
    cv2.imshow('Name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Countours_of_image(filename):
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.destroyAllWindows()
    th1= cv2.threshold(img_gray,127,255,0)
    th2 =cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    im2, contours, her= cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow('Source', img)
    cv2.imshow('Gray', img_gray)
    cv2.waitKey(0)
    print(her)