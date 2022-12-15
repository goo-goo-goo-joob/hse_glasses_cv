import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

class YUNET_CROP(object):
    def __init__(self, face_detection_model: str, 
                 score_threshold: float = 0.9, 
                 nms_threshold: float = 0.3,
                 top_k: int = 100,
                 size: tuple[int] = [320, 320], 
                 scale:float = 1, 
                 padding: tuple[int] = (0,0),
                 draw_image: bool = False, 
                 thickness: int=2):
        """
        face_detection_model - path to model
        score_threshold - threshold for detecting images
        top_k - top k images by threshold
        size - resize (for crop model)
        scale - scake for image
        padding - paddind for cropping images, original borders from crop model
        draw_image - draw full image with faces
        thickness - thickness of borders for draw_image
        """
        self.scale = scale
        self.w_padding = padding[0]
        self.h_padding = padding[1]
        self.draw_images = draw_image
        self.thickness = 2
        self.detector = cv.FaceDetectorYN.create(
            face_detection_model,
            "",
            size,
            score_threshold,
            nms_threshold,
            top_k
        )
    def crop(self, img1):
        img1Width = int(img1.shape[1]*args["scale"])
        img1Height = int(img1.shape[0]*args['scale'])
        img1 = cv.resize(img1, (img1Width, img1Height))
        self.detector.setInputSize((img1Width, img1Height))
        faces = self.detector.detect(img1)
        imgs = []
        for i in faces[1]:
            coords = i.astype(np.int32)
            nk = (int(coords[0] - self.h_padding ),int(coords[1] - self.w_padding))
            wh = (int(coords[0]+  self.h_padding + coords[2]), int(coords[1]+self.w_padding + coords[3]))

            imgs.append(img1[nk[1]:wh[1], nk[0]:wh[0]])
            if self.draw_images:
                copy_img = img1.copy()
                cv.rectangle(copy_img, nk, wh, (0, 255, 0), self.thickness)
        if self.draw_images:
            cv2_imshow(img1)
        return faces[1], imgs