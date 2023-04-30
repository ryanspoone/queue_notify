import os
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import pychromecast
import screeninfo
import mss
import sys
import subprocess
from dotenv import load_dotenv
import chump

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model = load_model(os.environ["MODEL_PATH"])

# Set up Pushover notifications
pushover_user_key = os.environ["PUSHOVER_USER_KEY"]
pushover_app_token = os.environ["PUSHOVER_APP_TOKEN"]

google_home_name = os.environ["GOOGLE_HOME_NAME"]

cast_device = None

last_notification_time = time.time()
max_retries = 5
retries = 0
app = chump.Application(pushover_app_token)
user = app.get_user(pushover_user_key)

# Create a directory to store queue popped screenshots
screenshot_folder = "screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)


while retries < max_retries:
    chromecasts, browser = pychromecast.get_listed_chromecasts(
        friendly_names=[google_home_name]
    )
    if chromecasts:
        cast_device = chromecasts[0]
        break
    else:
        retries += 1
        print(
            f"No device named '{google_home_name}' found. Retrying in 5 seconds..."
        )
        time.sleep(5)

if not cast_device:
    print(
        f"Could not find '{google_home_name}' after {max_retries} retries. Exiting."
    )
    sys.exit(1)
else:
    print(f"Found '{google_home_name}'!")
    # Start socket client's worker thread and wait for initial status update
    cast_device.wait()

# set the maximum volume
max_volume = 1.0  # this sets the max volume to 100%
cast_device.set_volume(max_volume)


def play_tts_on_google_home(url):
    has_played = False
    player_state = None
    t = 30
    cast_device.media_controller.play_media(url, "audio/mp3")
    while has_played is False:
        try:
            if (
                player_state
                != cast_device.media_controller.status.player_state
            ):
                player_state = cast_device.media_controller.status.player_state
            if player_state == "PLAYING":
                has_played = True
            if (
                cast_device.socket_client.is_connected
                and has_played
                and player_state != "PLAYING"
            ):
                has_played = False
                cast_device.media_controller.play_media(url, "audio/mp3")

            time.sleep(0.1)
            t = t - 0.1
        except Exception:
            break
    else:
        print(f"The file {url} does not exist. Please check the file path.")


def send_pushover_notification(message):
    # Create an emergency message with the given message
    # and send it to the user with emergency priority
    user.send_message(message, priority=chump.NORMAL)


def save_queue_popped_screenshot(img, pred):
    screenshot_filename = (
        f"queue_popped_{int(time.time())}_pred_{pred:.2f}.jpg"
    )
    screenshot_filepath = os.path.join(screenshot_folder, screenshot_filename)
    cv2.imwrite(screenshot_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Queue popped screenshot saved to {screenshot_filepath}")


# Play "Setup complete" MP3
play_tts_on_google_home(
    "https://drive.google.com/uc?export=download&id=1aBcBldLNUscpeDRg6ddFT9PflRWA_ZU4"
)
send_pushover_notification("Setup complete. Ready to receive notifications.")

# Continuously check for a queue pop every 5 seconds in a loop:
while True:
    try:
        # Take a screenshot
        monitor = screeninfo.get_monitors()[0]
        left, top, width, height = (
            monitor.x,
            monitor.y,
            monitor.width,
            monitor.height,
        )
        # Take a screenshot with 50% image quality
        with mss.mss() as sct:
            monitor = {
                "top": top,
                "left": left,
                "width": width,
                "height": height,
                "capture_every": 1,
                "quality": 50,
            }
            screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        # Preprocess the screenshot
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0

        # Make a prediction
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]

        # Notify if queue has popped
        if pred >= 0.95 and time.time() - last_notification_time >= 15:
            save_queue_popped_screenshot(img, pred)
            send_pushover_notification("The queue has popped!")

            # Play "Queue popped" WAV
            play_tts_on_google_home(
                "https://drive.google.com/uc?export=download&id=1J2JillCdrW-_ulNi1HdNyXodj20RDUqu"
            )

    except KeyboardInterrupt:
        # Disconnect from the Chromecast
        try:
            browser.stop_discovery()
            cast_device.disconnect()
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"Exception: {e}")
        try:
            browser.stop_discovery()
            cast_device.disconnect()
        except Exception:
            pass
        sys.exit(1)

    # Wait 1 second before the next iteration
    time.sleep(1)
