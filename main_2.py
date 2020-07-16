import cv2
import imutils
from imutils import contours
import numpy as np

import mnist_nn


def set_template(temp_path):
    img = cv2.imread(temp_path)
    # img = imutils.resize(img, 600)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)[1]

    # cv2.imshow("temp image", img)

    # find the contours and sort
    img_cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    img_cnts = imutils.grab_contours(img_cnts)
    img_cnts = contours.sort_contours(img_cnts, method="left-to-right")[0]

    digits = {}
    for (i, c) in enumerate(img_cnts):
        # find rectangle around the number for separation
        (x, y, w, h) = cv2.boundingRect(c)
        roi = img[y:y + h, x:x + w]
        # standard size
        roi = cv2.resize(roi, (42, 60))
        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi
        # cv2.imshow("2", roi)
    cv2.waitKey()

    return digits


def find_id(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, height=600)

    # cv2.imshow("INPUT", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (36, 9))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # show image
    # cv2.imshow("GRAY", gray)

    # th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("threshold gaussian", th3)

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    # cv2.imshow("black hat", blackhat)

    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))
    grad_x = (255 * ((grad_x - minVal) / (maxVal - minVal)))
    grad_x = grad_x.astype("uint8")

    # cv2.imshow("Sobel", grad_x)

    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
    thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)

    # cv2.imshow("threshold", thresh)
    # cv2.imwrite("th.png", thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    locs = []

    for (i, c) in enumerate(cnts):
        # find the area of digits
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # ar: aspect ratio
        if 2.0 < ar < 6.0:
            if (148 < w < 158) and (50 < h < 65):
                locs.append((x, y, w, h))

    locs = sorted(locs, key=lambda z: z[0])

    output_template = []
    output_mnist = []
    # init template

    digits = set_template("images/ref/1.png")
    # loop over the 4 groupings of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        gp_output_template = []
        # extract the group ROI of 4 digits from the gray-scale image,
        # then apply threshold to segment the digits from the
        # print(gY, gH, gX, gW)
        group = gray[gY-2:gY + gH + 2, gX:gX + gW+1]
        # group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # cv2.imshow("gp", group)
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2.imshow("gp", group)
        cv2.waitKey()
        # detect the contours of each individual digit in the group
        digit_cnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_cnts = imutils.grab_contours(digit_cnts)
        digit_cnts = contours.sort_contours(digit_cnts, method="left-to-right")[0]

        # loop over the digit contours
        for c in digit_cnts:
            # compute the bounding box of the individual digit
            (x, y, w, h) = cv2.boundingRect(c)

            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (42, 60))
            black_n = cv2.resize(roi, (12, 16))

            # black_n = cv2.bitwise_not(black_n)

            n = mnist_nn.predict(black_n)
            # print("NUMBER: ", n)
            output_mnist.append(str(int(n)))
            # cv2.imshow("black", black_n)
            # cv2.imwrite("5.png", black_n)
            # cv2.waitKey()
            # initialize a list of template matching scores
            scores = []
            # loop over the reference digit name and digit ROI
            for (digit, digitROI) in digits.items():
                # apply correlation-based template matching
                # cv2.imshow("DROI", digitROI)
                # cv2.waitKey()
                result = cv2.matchTemplate(roi, digitROI,
                                           cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            gp_output_template.append(str(np.argmax(scores)))

        cv2.rectangle(image, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (255, 255, 0), 4)
        # cv2.putText(image, "".join(gp_output_template), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        output_template.extend(gp_output_template)
    print("numbers from neural network: ", "".join(output_mnist))
    # print(len(output_template))
    print("numbers from template:       ", "".join(output_template))
    cv2.imshow("output", image)
    cv2.waitKey()


# set_template("images/ref/1.png")


find_id("images/1.png")
# find_id("images/2.png")
# find_id("images/3.jpg")
# find_id("images/4.jpg")
