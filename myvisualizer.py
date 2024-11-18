import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import numpy as np
from typing import Sequence

import cv2

import aria.sdk as aria
from projectaria_tools.core.sensor_data import (
    BarometerData,
    ImageDataRecord,
    MotionData,
    AudioData,
    AudioDataRecord
)

NANOSECOND = 1e-9  # Conversion factor from nanoseconds to seconds
IMAGE_MODES = ["RGB", "RIGHTSLAM", "LEFTSLAM", "EYETRACKER"]
SENSORS = ["IMU1", "IMU2", "MAGNETOMETER", "BAROMETER"]

class TemporalWindowPlot:
    """
    Manage a Matplotlib plot with streaming data, showing the most recent values.
    """

    def __init__(self, axes, title: str, dim: int, window_duration_sec: float = 4):
        self.axes = axes
        self.title = title
        self.window_duration = window_duration_sec
        self.timestamps = deque()
        self.samples = [deque() for _ in range(dim)]
        self.lines = [self.axes.plot([], [], label=f"Channel {i}")[0] for i in range(dim)]
        self.count = 0

        # Set up the plot appearance
        self.axes.set_title(self.title)
        self.axes.legend()
        # self.axes.set_ylim(-1, 1)  # Adjust as needed

    def add_samples(self, timestamp_ns: float, samples: Sequence[float]):
        # Convert timestamp to seconds
        timestamp = timestamp_ns * NANOSECOND

        # Remove old data outside of the window
        while self.timestamps and (timestamp - self.timestamps[0]) > self.window_duration:
            self.timestamps.popleft()
            for sample in self.samples:
                sample.popleft()

        # Add new data
        self.timestamps.append(timestamp)
        for i, sample in enumerate(samples):
            self.samples[i].append(sample)

    def update(self, frame):
        if not self.timestamps:
            return

        current_time = self.timestamps[-1]
        self.axes.set_xlim(current_time - self.window_duration, current_time)

         # Determine y-axis limits based on the current data with a buffer
        all_samples = [sample for channel in self.samples for sample in channel]
        min_y, max_y = min(all_samples), max(all_samples)
        y_buffer = 0.1 * (max_y - min_y) if max_y > min_y else 1  # Add a 10% buffer
        self.axes.set_ylim(min_y - y_buffer, max_y + y_buffer)
        
        for i, line in enumerate(self.lines):
            line.set_data(self.timestamps, list(self.samples[i]))

class MyVisualizer:
    """
    Example Aria visualizer class with separate windows for images and sensor plots.
    """

    def __init__(self):
        # Data logs
        self.image_log = {}
        for mode in IMAGE_MODES:
            self.image_log[mode] = []
        print(self.image_log)
        self.sensor_log = {}
        for mode in SENSORS:
            self.sensor_log[mode] = []
        self.audio_log = []

        self.image_figures = {
            "Front RGB": plt.figure("Aria RGB"),
            "Left SLAM": plt.figure("Aria SLAM Left"),
            "Right SLAM": plt.figure("Aria SLAM Right"),
            "Eye Track": plt.figure("Aria Eye Tracker")
        }

        # Create individual image plots in each figure
        self.image_axes = {
            "Front RGB": self.image_figures["Front RGB"].add_subplot(1, 1, 1),
            "Left SLAM": self.image_figures["Left SLAM"].add_subplot(1, 1, 1),
            "Right SLAM": self.image_figures["Right SLAM"].add_subplot(1, 1, 1),
            "Eye Track": self.image_figures["Eye Track"].add_subplot(1, 1, 1),
        }

        # Initialize the images with a placeholder
        self.image_plot = {
            "Front RGB": self.image_axes["Front RGB"].imshow(np.zeros((1408, 1408, 3), dtype="uint8")),
            "Left SLAM": self.image_axes["Left SLAM"].imshow(np.zeros((640, 480), dtype="uint8"), cmap="gray", vmin=0, vmax=255),
            "Right SLAM": self.image_axes["Right SLAM"].imshow(np.zeros((640, 480), dtype="uint8"), cmap="gray", vmin=0, vmax=255),
            "Eye Track": self.image_axes["Eye Track"].imshow(np.zeros((240, 640), dtype="uint8"), cmap="gray", vmin=0, vmax=255)
        }

        # Set titles and turn off axes for each image plot
        for title, ax in self.image_axes.items():
            ax.set_title(title)
            ax.axis("off")

        # Create the sensor plot figure
        self.sensor_fig, self.sensor_axes = plt.subplots(3, 2, figsize=(16, 8))
        self.sensor_fig.subplots_adjust(hspace=0.5, wspace=0.5)

        # Create the sensor plots dictionary
        self.sensor_plot = {
            "accel": [
                TemporalWindowPlot(self.sensor_axes[0, idx], f"IMU{idx} accel", 3)
                for idx in range(2)
            ],
            "gyro": [
                TemporalWindowPlot(self.sensor_axes[1, idx], f"IMU{idx} gyro", 3)
                for idx in range(2)
            ],
            "magneto": TemporalWindowPlot(self.sensor_axes[2, 0], "Magnetometer", 3),
            "baro": TemporalWindowPlot(self.sensor_axes[2, 1], "Barometer", 1),
        }
        axs = self.sensor_axes.flat
        for a in axs:
            a.get_xaxis().set_visible(False)

    def render_loop(self):
        # Initialize animation for each sensor plot
        self.animations = []
        for sensor_plots in self.sensor_plot.values():
            if isinstance(sensor_plots, list):
                for plot in sensor_plots:
                    anim = FuncAnimation(self.sensor_fig, plot.update, blit=False, interval=100)
                    self.animations.append(anim)
            else:
                anim = FuncAnimation(self.sensor_fig, sensor_plots.update, blit=False, interval=100)
                self.animations.append(anim)

        # Show both figures
        plt.show()

    def update_image(self, title: str, image_data: np.ndarray):
        # Update the data for the specific image plot
        if title in self.image_plot:
            self.image_plot[title].set_data(image_data)
            self.image_figures[title].canvas.draw_idle()

    def stop(self):
        plt.close(self.sensor_fig)
        for i in self.image_figures:
            i.close()

    def print_logs(self):
        for log in self.image_log:
            print(log)


class BaseStreamingClientObserver:
    """
    Streaming client observer class. Describes all available callbacks that are invoked by the
    streaming client.
    """

    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        pass

    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        pass

    def on_magneto_received(self, sample: MotionData) -> None:
        pass

    def on_baro_received(self, sample: BarometerData) -> None:
        pass

    def on_audio_received(self, sample: AudioData) -> None:
        pass

    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        pass

class MyVisualizerStreamingClientObserver(BaseStreamingClientObserver):
    """
    Example implementation of the streaming client observer class.
    Set an instance of this class as the observer of the streaming client using
    set_streaming_client_observer().
    """

    def __init__(self, visualizer: MyVisualizer):
        self.visualizer = visualizer

    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        # Rotate images to match the orientation of the camera
        if record.camera_id != aria.CameraId.EyeTrack:
            image = np.rot90(image, -1)
        else:
            image = np.rot90(image, 2)

        # Map camera ID to figure title
        camera_titles = {
            aria.CameraId.Rgb: "Front RGB",
            aria.CameraId.Slam1: "Left SLAM",
            aria.CameraId.Slam2: "Right SLAM",
            aria.CameraId.EyeTrack: "Eye Track"
        }
        title = camera_titles.get(record.camera_id)
        if title:
            self.visualizer.update_image(title, image)
            self.visualizer.image_log[IMAGE_MODES[record.camera_id]].append(image)

    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        # Only plot the first IMU sample per batch
        sample = samples[0]
        self.visualizer.sensor_plot["accel"][imu_idx].add_samples(
            sample.capture_timestamp_ns, sample.accel_msec2
        )
        self.visualizer.sensor_plot["gyro"][imu_idx].add_samples(
            sample.capture_timestamp_ns, sample.gyro_radsec
        )

    def on_magneto_received(self, sample: MotionData) -> None:
        self.visualizer.sensor_plot["magneto"].add_samples(
            sample.capture_timestamp_ns, sample.mag_tesla
        )

    def on_baro_received(self, sample: BarometerData) -> None:
        self.visualizer.sensor_plot["baro"].add_samples(
            sample.capture_timestamp_ns, [sample.pressure]
        )

    def on_audio_received(self, sample: AudioData, record: AudioDataRecord) -> None:
        print(sample.data)
        pass

    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        print(f"Streaming Client Failure: {reason}: {message}")
