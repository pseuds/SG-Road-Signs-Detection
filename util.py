import cv2 
import time
import matplotlib.pyplot as plt
import numpy as np

def get_coordinates_interactive(path: str) -> tuple[int,int,int,int]:
    """
    Opens up a GUI to get bbox coordinates of image.

    Parameters
    ----------
    path : str
        The path to image or video. If video, it takes the first image of the video.

    Returns
    -------
    tuple
        x, y, w, h, where x,y are the centre coordinates of the bbox, and w,h are the width and height of the bbox respectively.
    """

    if 'mp4' in path:
        # get first img of video
        cap = cv2.VideoCapture(path)
        ret, img = cap.read()
    elif '.jpg' in path:
        # singular img
        img = cv2.imread('test_uncropped/014.png')  # Replace with your image path
    else:
        print("ERROR: This current version only supports .mp4 or .jpg format.")
        return None

    clone = img.copy()

    # init list of reference pt, and cropping state
    ref_pt = []
    display_done = False

    def click_crop(event, x, y, flags, params):
        nonlocal ref_pt, display_done

        # when L mouse is clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # clear all drawings
            ref_pt = []

            # show coordinates of clicked point
            print(f'First ref pt: {(x,y)}', end=", ")
            ref_pt.append((x,y))
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, str((x,y)), (x, y), font, 1, (255, 0, 0), 1)

        # when L mouse is released
        elif event == cv2.EVENT_LBUTTONUP:
            # show coordinates of released point
            print(f'Second ref pt: ({x}, {y})')
            ref_pt.append((x,y))
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, str((x,y)), (x, y), font, 1, (255, 0, 0), 1)

            # draw rectangle of crop
            cv2.rectangle(img, ref_pt[0], ref_pt[1], (0,255,0), 2)

            # # draw centre point, get w and h
            xr,yr,wr,hr = get_xywh_from_refpts(ref_pt)
            cv2.circle(img, (xr,yr), radius=0, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, str((xr,yr)), (xr,yr), font, 1, (0, 0, 255), 1)
            print(f'x,y,w,h: {(xr,yr)}, {wr},{hr}')

            
            
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_crop)

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF

        # when 'r' pressed, reset to original image
        if key == ord("r"):
            img = clone.copy()
            ref_pt = []
        
        # when enter pressed, exit if 2 ref pts obtained
        elif key == 13:
            if len(ref_pt) == 2:
                print("Bbox selected.")
                break
            else:
                print("ERROR: No bbox selected.")
                continue

        # when 'q' pressed, exit
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    return get_xywh_from_refpts(ref_pt)




def get_xywh_from_refpts(refpts: list) -> tuple[int,int,int,int]:
    """
    Returns values x, y, w, h, where x,y are the centre coordinates of the bbox, and w,h are the width and height of the bbox respectively.

    Parameters
    ----------
    refpts : list
        A list of two refpts, which are opposite corners of a bbox.

    Returns
    -------
    tuple
        x, y, w, h, where x,y are the centre coordinates of the bbox, and w,h are the width and height of the bbox respectively.
    """
    # unpack the two reference points 
    pt1, pt2 = refpts[0], refpts[1]

    # calculate w, h
    w = abs(pt2[0]-pt1[0])
    h = abs(pt2[1]-pt1[1])

    # calculate centre
    x = (pt2[0]+pt1[0])//2
    y = (pt2[1]+pt1[1])//2

    return x,y,w,h

    
def plot_imgs(img1:np.ndarray, img2:np.ndarray):
    plt.figure(figsize=(12,6))
    plt.subplot(121),plt.imshow(img1,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2,cmap = 'gray')
    plt.title('Preprocessed Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()