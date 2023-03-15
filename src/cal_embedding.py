import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from .models.inception_resnet_v1 import InceptionResnetV1
from .utils.utils import TensorrtBase


class cal_face_embedding(object):
    def __init__(self,
                 model_weights="/workspace/huangniu_demo/facenet/src/weights/20180402-114759-vggface2.pt",
                 device=None,
                 speed_up=False,
                 speed_up_weights="",
                 max_face_num=20,
                 rebuild_engine=False,
                 ):

        self.device = device
        self.speed_up = speed_up
        self.speed_up_weights = speed_up_weights
        self.max_face_num = max_face_num
        self.rebuild_engine = rebuild_engine

        # create feature extractor and load weights
        print("creating face feature extactor")
        self.model = InceptionResnetV1(device=self.device).eval()
        print("loading face feature extactor weights")
        extractor_weights_state_dict = torch.load(model_weights)
        incompatible = self.model.load_state_dict(extractor_weights_state_dict, strict=False)
        if incompatible.missing_keys:
            print("missing_keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("unexpected_keys:", incompatible.unexpected_keys)
        print("face feature extactor weights loaded")

        # speed up by tensorrt
        if self.speed_up:
            os.makedirs(self.speed_up_weights, exist_ok=True)
            onnx_filename = os.path.join(self.speed_up_weights, model_weights.split("/")[-1].split(".")[0] + ".onnx")
            trt_filename = os.path.join(self.speed_up_weights, model_weights.split("/")[-1].split(".")[0] + ".engine")
            self.face_cog_trt = TensorrtBase(
                model=self.model,
                if_dynamic=True,
                input_shape=(160, 160),
                speed_up_weights_root=self.speed_up_weights,
                onnx_filename=onnx_filename,
                trt_filename=trt_filename,
                gpu_id=str(self.device.index),
                dynamic_factor=1,
                max_face_num=self.max_face_num,
                rebuild_engine=self.rebuild_engine,
            )

    def get_size(self, img):
        if isinstance(img, (np.ndarray, torch.Tensor)):
            return img.shape[1::-1]
        else:
            return img.size

    def imresample(self, img, sz):
        im_data = interpolate(img, size=sz, mode="area")
        return im_data

    def crop_resize(self, img, box, image_size):
        if isinstance(img, np.ndarray):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = cv2.resize(
                img,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        elif isinstance(img, torch.Tensor):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = self.imresample(img.permute(2, 0, 1).unsqueeze(0).float(), (image_size, image_size)).byte().squeeze(
                0).permute(1, 2, 0)
        else:
            out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
        return out

    def extract_face(self, img, box, image_size=160, margin=0):
        """Extract face + margin from PIL Image given bounding box.

        Arguments:
            img {PIL.Image} -- A PIL Image.
            box {numpy.ndarray} -- Four-element bounding box.
            image_size {int} -- Output image size in pixels. The image will be square.
            margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
                Note that the application of the margin differs slightly from the davidsandberg/facenet
                repo, which applies the margin to the original image before resizing, making the margin
                dependent on the original image size.
            save_path {str} -- Save path for extracted face image. (default: {None})

        Returns:
            torch.tensor -- tensor representing the extracted face.
        """

        margin = [
            margin * (box[2] - box[0]) / (image_size - margin),
            margin * (box[3] - box[1]) / (image_size - margin),
        ]
        raw_image_size = self.get_size(img)
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]  # margin==0，无外扩
        face = self.crop_resize(img, box, image_size)
        if not self.speed_up:
            face = F.to_tensor(np.float32(face))
        else:
            face = np.ascontiguousarray(np.transpose(np.float32(face), axes=(2, 0, 1)))
        return face

    def fixed_image_standardization(self, image_tensor):
        processed_tensor = (image_tensor - 127.5) / 128.0
        return processed_tensor

    def get_crop_processed_faces(self, img, boxes, image_size=160, margin=0):
        # img: shape=(H,W,3), BGR, ndarray
        # boxes: shape=(N, 4), ndarray
        im = Image.fromarray(img[:, :, ::-1])  # RGB {PIL.Image}
        faces = []
        for i, box in enumerate(boxes):
            face = self.extract_face(im, box, image_size, margin)
            face = self.fixed_image_standardization(face)
            faces.append(face)
        if not self.speed_up:
            faces = torch.stack(faces)
        else:
            faces = np.stack(faces, axis=0)
        return faces

    @torch.no_grad()
    def cal_embedding(self, inputs):
        # inputs, tensor.float32, (N, 3, H, W)
        inputs = inputs.to(self.device)
        embedding = self.model(inputs)
        return embedding

    def get_img_faces_embedding(self, img, boxes):
        # img: shape=(H,W,3), BGR, ndarray
        # boxes: shape=(N, 4), ndarray
        inputs = self.get_crop_processed_faces(img, boxes)
        if not self.speed_up:
            faces_emb = self.cal_embedding(inputs).cpu().numpy()
            return faces_emb
        else:
            binding_shape_map = {
                "input": inputs.shape
            }
            trt_outs = self.face_cog_trt.do_inference([inputs], binding_shape_map)[0]
            return trt_outs


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    c_e = cal_face_embedding(device=device)
    x = torch.randn((4, 3, 160, 160))
    embedding_x = c_e.cal_embedding(x)
