import face_recognition
import numpy as np
import cv2

mask_ori = cv2.imread('cln.png',-1) 
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FPS, 30)
 
 
 
def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  
    rows, cols, _ = src.shape  
    y, x = pos[0], pos[1]  
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src
 
 
 
while 1:
    ret, img = cap.read()
    rgb_frame = img[:, :, ::-1]
    facer = face_recognition.face_locations(rgb_frame)
    faces=[(0,0,0,0)]
    if facer != []:
        faces=[[facer[0][3],facer[0][0],abs(facer[0][3]-facer[0][1])+150,abs(facer[0][0]-facer[0][2])]]
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:

            mask_symin = int(y- 3 * h / 5)
            mask_symax = int(y + 8 * h / 5)
            sh_mask = mask_symax - mask_symin
            face_mask_roi_color = img[mask_symin:mask_symax, x:x+w]

            mask= cv2.resize(mask_ori, (w, sh_mask),interpolation=cv2.INTER_CUBIC)
           
            transparentOverlay(face_mask_roi_color,mask)
    cv2.imshow('Frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
