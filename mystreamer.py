# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import os

import aria.sdk as aria
from common import quit_keypress, update_iptables

from myvisualizer import *

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile12",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    parser.add_argument(
        "--log", 
        help="Log sensor and image to designated directory",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--log_dir", help="Directory to log output"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # Check if log directory already exists, blocks if already exists
    if args.log_dir:
        output_dir = f"data_collection/{args.log_dir}"
        if os.path.isdir(output_dir) and args.log_dir:
            exit("Directory already exists, please retry with a new directory")

    # Set SDK's log level
    aria.set_log_level(aria.Level.Info)

    # Create DeviceClient to begin streaming from code and not CLI
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # Connect to the device
    device = device_client.connect()

    # Retrieve the streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # Configurate StreamingManager
    # Set custom profile or default to profile18
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name

    # By default streaming uses Wifi, set usb if specified
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb

    # Use ephemeral streaming certificates for StreamingManager
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # Start streaming
    streaming_manager.start_streaming()
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    # -------------------------------------------------------------
    # End config for StreamingManager, begin StreamingClient config
    # -------------------------------------------------------------

    # Configure what to listen to, look at StreamingDataType for more options
    config = streaming_client.subscription_config
    # config.subscriber_data_type = (
    #     aria.StreamingDataType.Rgb | aria.StreamingDataType.Slam
    # )
    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.Slam] = 1

    # Set the security options for StreamingClient
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # observer = StreamingClientObserver()
    visualizer = MyVisualizer()
    observer = MyVisualizerStreamingClientObserver(visualizer)
    streaming_client.set_streaming_client_observer(observer)

    # Start listening
    print("Start listening to image data")
    streaming_client.subscribe()
    print(f"Subscribed to stream: {streaming_client.is_subscribed()}")

    visualizer.render_loop()

    # Unsubscribe to clean up resources, stop stream and disconnect from device
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
    if args.log and args.log_dir:
        visualizer.write_logs(args.log_dir)


if __name__ == "__main__":
    main()
