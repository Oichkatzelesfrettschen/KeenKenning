import subprocess
import time
import xml.etree.ElementTree as ET
import re
import os

ADB = "/opt/android-sdk/platform-tools/adb"
PACKAGE = "org.yegie.keenkenning.kenning"  # Keen Kenning flavor

def run_adb(cmd):
    full_cmd = f"{ADB} {cmd}"
    # print(f"Executing: {full_cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def dump_ui():
    run_adb("shell uiautomator dump /sdcard/dump.xml")
    xml_content = run_adb("shell cat /sdcard/dump.xml")
    if not xml_content.startswith("<?xml"):
        return None
    return ET.fromstring(xml_content)

def find_node(root, **kwargs):
    for node in root.iter("node"):
        match = True
        for key, value in kwargs.items():
            if node.get(key) != value:
                match = False
                break
        if match:
            return node
    return None

def get_coords(node):
    bounds = node.get("bounds") # "[x1,y1][x2,y2]"
    match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1 + x2) // 2, (y1 + y2) // 2
    return None

def main():
    print("Waiting for boot...")
    while True:
        status = run_adb("shell getprop sys.boot_completed")
        if status == "1":
            break
        time.sleep(5)
    
    print("Installing APK...")
    run_adb("install -r app/build/outputs/apk/debug/app-debug.apk")
    
    print("Launching Keen...")
    run_adb(f"shell monkey -p {PACKAGE} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)
    
    print("Analyzing UI...")
    root = dump_ui()
    if root is None:
        print("Failed to dump UI")
        return

    # Find START GAME button
    # Based on activity_menu.xml, it's a Button with text "START GAME" (or @string/play)
    # Since I updated it to "START GAME" in strings.xml
    start_btn = find_node(root, text="START GAME")
    if start_btn is None:
        print("START GAME button not found. Dumping nodes...")
        for node in root.iter("node"):
            if node.get("text"): print(f"Found text: {node.get('text')}")
        return

    x, y = get_coords(start_btn)
    print(f"Clicking START GAME at {x}, {y}")
    run_adb(f"shell input tap {x} {y}")
    time.sleep(5) # Wait for level gen
    
    print("Analyzing Game Grid...")
    root = dump_ui()
    # In KeenActivity, the game view is a custom view.
    # Uiautomator might see it as a single node or a container.
    # I'll check if the progress bar is gone.
    prog = find_node(root, resource_id=f"{PACKAGE}:id/progress_bar")
    if prog:
        print("Still loading...")
        time.sleep(10)
        root = dump_ui()

    # Let's try to click in the middle of the screen to select a cell
    # and then click one of the numbers.
    # Grid is usually at the top.
    print("Attempting to interact with the game grid...")
    run_adb("shell input tap 500 500") # Center-ish
    time.sleep(1)
    
    # Numbers panel is at the bottom.
    # Let's try to find a node with text "1"
    num_1 = find_node(root, text="1")
    if num_1:
        nx, ny = get_coords(num_1)
        print(f"Clicking number 1 at {nx}, {ny}")
        run_adb(f"shell input tap {nx} {ny}")
    else:
        # Fallback to hardcoded bottom area click
        print("Number button not found by text, trying coordinate-based click for '1'...")
        run_adb("shell input tap 100 1500") 

    print("Verification: Dumping UI again...")
    time.sleep(2)
    root = dump_ui()
    # Check if '1' appeared in the grid area or if the undo button is now visible
    undo_btn = find_node(root, text="undo")
    if undo_btn:
        print("SUCCESS: 'undo' button detected. Interaction confirmed.")
    else:
        print("Interaction could not be confirmed via UI dump, but commands were sent.")

if __name__ == "__main__":
    main()
