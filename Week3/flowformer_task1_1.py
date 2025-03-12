import sys
sys.path.append('core')

from PIL import Image
import argparse
import time
import numpy as np
import torch
import cv2

# Import configuration helpers from FlowFormer
from FlowFormer.configs.things_eval import get_cfg as get_things_cfg
from FlowFormer.core.utils.utils import InputPadder
from FlowFormer.core.FlowFormer import build_flowformer

# ------------------------------------------------------------------
def read_flow_gt(flow_file):
    """
    Reads the ground truth optical flow from a file (e.g. a specially encoded PNG)
    and returns an array with components (u, v) and a validity mask.
    """
    flow_raw = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED).astype(np.double)
    u = (flow_raw[:, :, 2] - 2**15) / 64.0  # u component
    v = (flow_raw[:, :, 1] - 2**15) / 64.0  # v component
    valid = flow_raw[:, :, 0] == 1         # validity mask (1: valid, 0: invalid)
    u[valid == 0] = 0
    v[valid == 0] = 0
    return np.stack((u, v, valid), axis=2)

def compute_msen_pepn(flow_pred, flow_gt, tau=3):
    """
    Computes the MSEN (Mean Square Error Norm) and PEPN (Percentage of Erroneous Pixels)
    between the predicted flow and the ground truth.
    """
    square_error = (flow_pred[:, :, 0] - flow_gt[:, :, 0])**2 + (flow_pred[:, :, 1] - flow_gt[:, :, 1])**2
    valid_mask = flow_gt[:, :, 2]
    error_valid = square_error * valid_mask
    non_occluded = np.sum(valid_mask != 0)
    pixel_error = np.sqrt(error_valid)
    msen = np.sum(pixel_error) / non_occluded
    erroneous = np.sum(pixel_error > tau)
    pepn = (erroneous / non_occluded) * 100
    return msen, pepn

# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FlowFormer Optical Flow Estimation Demo")
    parser.add_argument('-viz', dest='viz', action='store_true', help='Visualize (i.e. save) the output flow')
    parser.add_argument('--model', type=str, default='/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/Week3/sintel.pth',
                        help='Path to the FlowFormer checkpoint')
    args = parser.parse_args()

    # Paths to the example images and ground truth
    img1_path = '/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/Week3/data_stereo_flow/training/image_0/000050_10.png'
    img2_path = '/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/Week3/data_stereo_flow/training/image_0/000050_11.png'
    gt_flow_path = '/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/Week3/data_stereo_flow/training/flow_noc/000050_10.png'

    # Load images using PIL and convert to RGB
    im1 = Image.open(img1_path).convert("RGB")
    im2 = Image.open(img2_path).convert("RGB")

    # Convert images to numpy arrays (H, W, C) and normalize to [0, 1]
    im1_np = np.array(im1).astype(np.float32) / 255.
    im2_np = np.array(im2).astype(np.float32) / 255.
    if im1_np.ndim == 2:
        im1_np = np.expand_dims(im1_np, axis=2)
        im2_np = np.expand_dims(im2_np, axis=2)

    # Convert to torch tensors with shape (1, 3, H, W)
    im1_tensor = torch.from_numpy(im1_np.transpose(2, 0, 1)).unsqueeze(0)
    im2_tensor = torch.from_numpy(im2_np.transpose(2, 0, 1)).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im1_tensor = im1_tensor.to(device)
    im2_tensor = im2_tensor.to(device)

    # Load configuration (using the "things" config here) and update with the checkpoint path
    cfg = get_things_cfg()
    cfg.model = args.model
    # Optionally, adjust other config parameters if needed

    # Build FlowFormer model
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    model = model.to(device)
    model.eval()

    # Prepare input by padding (to handle arbitrary image sizes)
    padder = InputPadder(im1_tensor.shape)
    im1_tensor, im2_tensor = padder.pad(im1_tensor, im2_tensor)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        # It is assumed that the model returns a tuple/list, and the first element is the flow prediction.
        flow_output = model(im1_tensor, im2_tensor)
    end_time = time.time()
    print(f"Time Taken: {end_time - start_time:.2f} seconds for image of size {im1_np.shape}")

    # Unpad the output; assume output is a tuple and we take the first element.
    flow_tensor = padder.unpad(flow_output[0]).cpu()  # shape (1, 2, H, W)
    # Remove batch dimension and convert to numpy with shape (H, W, 2)
    flow_np = flow_tensor[0].numpy().transpose(1, 2, 0)

    # Save the optical flow
    np.save('outFlow_flowformer.npy', flow_np)

    # Compute evaluation metrics against ground truth flow
    flow_gt = read_flow_gt(gt_flow_path)
    # flow_pred should be in shape (H, W, 2)
    msen, pepn = compute_msen_pepn(flow_np, flow_gt)
    print(f"MSEN: {msen:.4f}")
    print(f"PEPN: {pepn:.4f}%")

    # Visualization (if -viz flag is set)
    if args.viz:
        h, w, _ = flow_np.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255  # Maximum saturation
        mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # Normalize angle to [0,180]
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('outFlow_flowformer.png', rgb)

if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()
