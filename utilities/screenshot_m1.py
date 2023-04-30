import time
import os
from PIL import ImageGrab, Image
from screeninfo import get_monitors
from datetime import datetime


def take_screenshot(screenshot_folder, monitor):
    left, top, right, bottom = (
        monitor.x,
        monitor.y,
        monitor.x + monitor.width,
        monitor.y + monitor.height,
    )
    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
    screenshot = screenshot.resize(
        (screenshot.width // 2, screenshot.height // 2), Image.ANTIALIAS
    )
    file_name = f"{screenshot_folder}/screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    screenshot.save(file_name, "JPEG", quality=50)
    print(f"Screenshot saved as {file_name}")


def main():
    screenshot_folder = "screenshots"

    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    monitors = get_monitors()
    if len(monitors) < 1:
        print("No monitors found.")
        return

    monitor1 = monitors[0]

    interval = 5  # Set the interval between screenshots, in seconds
    while True:
        take_screenshot(screenshot_folder, monitor1)
        time.sleep(interval)


if __name__ == "__main__":
    main()
