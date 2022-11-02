import sys
import yaml
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)
       
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    
    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, kp_detector, dense_motion_network, avd_network


def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode = 'relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device= 'cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode = 'relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


class Animator:
    
    def __init__(self, source_image, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode='relative'):
        assert mode in ['standard', 'relative', 'avd']
        self.mode = mode
        self.device = device
        
        with torch.no_grad():
            self.source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            self.source = self.source.to(self.device)
            self.kp_detector = kp_detector
            self.kp_source = self.kp_detector(self.source)
            self.kp_driving_initial = None
            
            self.inpainting_network = inpainting_network
            self.dense_motion_network = dense_motion_network
            self.avd_network = avd_network

    def next_frame(self, frame):
        with torch.no_grad():
            frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            driving_frame = frame.to(self.device)
        
            if self.kp_driving_initial is None:
                self.kp_driving_initial = self.kp_detector(driving_frame)
                
            kp_driving = self.kp_detector(driving_frame)
            if self.mode == 'standard':
                kp_norm = self.kp_driving
            elif self.mode=='relative':
                kp_norm = relative_kp(kp_source=self.kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=self.kp_driving_initial)
            elif self.mode == 'avd':
                kp_norm = self.avd_network(self.kp_source, kp_driving)
            dense_motion = self.dense_motion_network(source_image=self.source, kp_driving=kp_norm,
                                                    kp_source=self.kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = self.inpainting_network(self.source, dense_motion)

            prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            
            return prediction


class opt:
    source_image = 'G:/Images/Emma_watson_asian_empress.jpg'
    config = 'config/vox-256.yaml'
    checkpoint = 'checkpoints/vox.pth.tar'
    img_shape = (256, 256)  # 'Shape of image, that the model was trained on.'
    mode = 'relative'  # "Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result"
    find_best_frame = False  # "Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)"
    cpu = False


from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

model_name = 'RealESRGAN_x2plus'
model_path = 'G:/Python Scripts/Real-ESRGAN/models/RealESRGAN_x2plus.pth'

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
netscale = 2

tile = 0
tile_pad = False
pre_pad = False
fp32 = False
gpu_id = 0

# restorer
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    model=model,
    tile=tile,
    tile_pad=tile_pad,
    pre_pad=pre_pad,
    half=not fp32,
    gpu_id=gpu_id)

def upsample(image):
    output, _ = upsampler.enhance(image, outscale=2)
    return output

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return image
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        y = y - 40
        x = x - 40
        h = h + 80
        w = w + 80
        face = image[y:y + h, x:x + w]
    return face
    
def main():
    source_image = imageio.imread(opt.source_image)

    if opt.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    source_image = resize(source_image, opt.img_shape)[..., :3]
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
        config_path = opt.config, checkpoint_path = opt.checkpoint, device = device)

    animator = Animator(source_image, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
    cap = cv2.VideoCapture(0)

    try:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break
            # frame = extract_face(frame)
            h, w = frame.shape[:2]
            nh = int(h * 0.7)
            nh = h
            frame = frame[(h - nh) // 2 : h - (h - nh) // 2, (w - nh) // 2 : w - (w - nh) // 2]
            cv2.imshow('frame', frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = resize(frame, opt.img_shape)
            res = animator.next_frame(frame)
            
            # Image.fromarray((res * 255).astype(np.uint8)).save(f'frame_{i}.jpg')
            res = (res * 255).astype(np.uint8)
            out_img = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            # out_img = upsample(out_img)
            cv2.imshow('image', out_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
