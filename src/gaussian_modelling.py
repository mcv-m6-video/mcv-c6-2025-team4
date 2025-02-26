import cv2
import numpy as np

class NonRecursiveGaussianModel:
    def __init__(self):
        super().__init__()

    def compute_gaussian_background(self, frames):
        """
        Computes the pixel-wise Gaussian model from a list of frames.
        Returns the background mean and variance images
        :param frames:
        :return:
        """

        # Stack frames to form a 3D array: shape (num_frames, height, width)
        frames_stack = np.stack(frames, axis = 0)

        # Compute the mean and variance for each pixel over time
        background_mean = np.mean(frames_stack, axis = 0)
        background_var = np.var(frames_stack, axis = 0)

        return background_mean, background_var

    def classify_frame(self, frame, background_mean, background_variance, threshold_factor=0.5):
        """
        Classifies each pixel of an RGB frame as background if its intensity in each channel
        lies within [mean - threshold_factor * sigma, mean + threshold_factor * sigma].
        Returns a binary mask with values 0 (foreground) or 255 (background).
        """
        # Convert to float for accurate computation
        frame_float = frame.astype(np.float32)
        bg_mean_float = background_mean.astype(np.float32)
        sigma = background_variance.astype(np.float32)
        sigma = background_variance.astype(np.float32)

        # Compute the absolute difference for each channel
        diff = np.abs(frame_float - bg_mean_float)

        # Check per channel if the difference is within the threshold.
        within_threshold = diff <= (threshold_factor * sigma)

        # A pixel is background only if all three channels are within the threshold.
        background_mask = np.all(within_threshold, axis=2).astype(np.uint8) * 255
        return background_mask

class AdaptiveGaussianModel:
    def __init__(self, rho=0.01, threshold_factor=8):
        """
        Initialize the adaptive background model.
        :param rho: Update rate (e.g., 0.01 for slow adaptation)
        :param threshold_factor: Threshold factor to decide if a pixel is background
        """
        self.rho = rho
        self.threshold_factor = threshold_factor
        self.background_mean = None  # will be a (H, W, 3) array for RGB
        self.background_variance = None  # same shape as mean

    def compute_gaussian_background(self, frames):
        """
        Computes the pixel-wise Gaussian model (mean and variance) from a list of RGB frames.
        :param frames: List of frames (each as an RGB numpy array)
        :return: background_mean, background_variance
        """
        # Stack frames into a 4D array: shape (num_frames, height, width, channels)
        frames_stack = np.stack(frames, axis=0)
        background_mean = np.mean(frames_stack, axis=0)
        background_variance = np.var(frames_stack, axis=0)
        return background_mean, background_variance

    def initialize(self, training_frames):
        """
        Initializes the background model using a list of training frames.
        """
        self.background_mean, self.background_variance = self.compute_gaussian_background(training_frames)

    def classify_frame(self, frame):
        """
        Classifies each pixel of an RGB frame as background if its intensity in every channel lies
        within [mean - threshold_factor * sigma, mean + threshold_factor * sigma].
        Returns a binary mask with 255 indicating background.
        :param frame: Input RGB frame
        :return: Binary mask (uint8) of the same height and width
        """
        frame_float = frame.astype(np.float32)
        sigma = np.sqrt(self.background_variance.astype(np.float32) + 1e-6)
        diff = np.abs(frame_float - self.background_mean.astype(np.float32))
        # Check per channel if the difference is within the threshold
        within_threshold = diff <= (self.threshold_factor * sigma)
        # A pixel is considered background only if all channels are within the threshold
        background_mask = np.all(within_threshold, axis=2).astype(np.uint8) * 255
        return background_mask

    def update_background(self, frame, background_mask):
        """
        Updates the background model recursively. Only pixels classified as background (mask==255)
        are used to update the mean and variance.
        :param frame: Current RGB frame
        :param background_mask: Binary mask indicating background pixels
        """
        frame_float = frame.astype(np.float32)
        # Create a boolean mask for background pixels
        mask = (background_mask == 255)
        # Copy the current background parameters
        new_mean = self.background_mean.copy()
        new_variance = self.background_variance.copy()

        # For pixels classified as background, update the mean and variance
        new_mean[mask] = (1 - self.rho) * self.background_mean[mask] + self.rho * frame_float[mask]
        new_variance[mask] = (1 - self.rho) * self.background_variance[mask] + self.rho * ((frame_float[mask] - new_mean[mask]) ** 2)

        self.background_mean = new_mean
        self.background_variance = new_variance

    def process_frame(self, frame):
        """
        Processes a frame: classifies it using the current background model, then updates
        the model for pixels classified as background.
        :param frame: Input RGB frame
        :return: Background mask (binary image)
        """
        mask = self.classify_frame(frame)
        self.update_background(frame, mask)
        return mask

import cv2

class GMMBackgroundSubtractor:
    def __init__(self, history=500, varThreshold=16, detectShadows=True):
        """
        Initializes the Gaussian Mixture Model (GMM) background subtractor.

        Parameters:
        - history: int, optional
            The number of frames used to learn the background model.
        - varThreshold: float, optional
            Threshold on the squared Mahalanobis distance to decide whether
            a pixel is well described by the background model.
        - detectShadows: bool, optional
            If True, the algorithm detects and marks shadows in the output.
        """
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=varThreshold, detectShadows=detectShadows
        )

    def apply(self, frame, learningRate=-1):
        """
        Applies background subtraction to a given frame.

        Parameters:
        - frame: np.ndarray
            The input image frame in which foreground objects need to be detected.
        - learningRate: float, optional
            The learning rate that controls how fast the background model adapts
            to changes. If set to -1, an automatic learning rate is used.

        Returns:
        - fg_mask: np.ndarray
            A binary mask where foreground pixels are white (255) and
            background pixels are black (0).
        """
        fg_mask = self.backSub.apply(frame, learningRate=learningRate)
        return fg_mask

    def get_background(self):
        """
        Retrieves the estimated background image.

        Returns:
        - background: np.ndarray
            The current background model as an image. Returns None if
            the background model has not been initialized yet.
        """
        return self.backSub.getBackgroundImage()
