# converter_core.py  – YENİ
import cv2, numpy as np

# -------------------------------------------------
#  Ana API – image_to_lines
# -------------------------------------------------
# converter_core.py 

def image_to_lines(
        gray_img      : np.ndarray,
        thr           : int = 60,
        min_len       : int = 30,
        canny_low     : int = 50,
        canny_high    : int = 150,
        hough_thresh  : int = 50,
        max_line_gap  : int = 50,
        min_pixels    : int = 50,
        morph_kernel  : tuple = (2, 2)
    ):
    """
    Kat planı → {"lines":[…]}  (her segment default 'wall')
    """
    # 1) Binary + morfolojik temizlik
    _, bin_img = cv2.threshold(gray_img, thr, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    opened  = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,  k, iterations=1)
    closed  = cv2.morphologyEx(opened , cv2.MORPH_CLOSE, k, iterations=2)

    # 2) Skeleton + küçük parçaları at
    skel = skeletonize_morphological(closed)
    skel = remove_small_components(skel, min_pixels)

    # 3) Canny → HoughLinesP
    
    edges  = cv2.Canny(skel, canny_low, canny_high)

    linesP = cv2.HoughLinesP(edges, 
                             1, 
                             np.pi/180,
                             hough_thresh,
                             minLineLength=min_len,
                             maxLineGap=max_line_gap)

    # 4) JSON hazırlığı
    out, seg_id = {"lines": []}, 0
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:, 0]:
            out["lines"].append({
                "id"  : int(seg_id),
                "type": "wall",                 # kapı/pencere henüz bulunamıyor
                "p1"  : {"x": float(x1), "y": float(y1)},
                "p2"  : {"x": float(x2), "y": float(y2)}
            })
            seg_id += 1
    return out



# ––––––––––––––  Yardımcılar  ––––––––––––––
def skeletonize_morphological(bin_img):
    img, skel = bin_img.copy(), np.zeros_like(bin_img)
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        er = cv2.erode(img, ker) # ––––––nesne küçültme
        temp = cv2.dilate(er, ker) # ––––––kalan alan çekirdeği
        temp = cv2.subtract(img, temp) 
        skel = cv2.bitwise_or(skel, temp) # ––––––ana iskelete ekleme
        img = er
        if cv2.countNonZero(img) == 0:
            break
    return skel


def remove_small_components(skeleton, min_pixels=50):
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(skeleton, 8)
    clean = np.zeros_like(skeleton)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            clean[lbl == i] = 255
    return clean
