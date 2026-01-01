#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


DEFAULT_MODES = ["Standard", "Multiply", "Mystery", "Zero Mode"]
DEFAULT_DIFFS = [
    "Easy",
    "Normal",
    "Hard",
    "Extreme",
    "Unreasonable",
    "Ludicrous",
    "Incomprehensible",
]


def build_adb(serial):
    if serial:
        return ["adb", "-s", serial]
    return ["adb"]


def adb_cmd(adb_base, args, retries=3, delay=2.0):
    for attempt in range(retries):
        res = subprocess.run(
            adb_base + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            return res.stdout

        err = (res.stderr or "") + (res.stdout or "")
        if "device offline" in err or "no devices/emulators found" in err:
            time.sleep(delay)
            continue

        raise RuntimeError(f"adb failed: {' '.join(args)}\n{err.strip()}")

    raise RuntimeError(f"adb failed after retries: {' '.join(args)}")


def wait_for_boot(adb_base):
    adb_cmd(adb_base, ["wait-for-device"], retries=1)
    for _ in range(90):
        try:
            booted = adb_cmd(adb_base, ["shell", "getprop", "sys.boot_completed"]).strip() == "1"
            if booted:
                adb_cmd(adb_base, ["shell", "cmd", "activity", "get-current-user"], retries=1)
                adb_cmd(adb_base, ["shell", "cmd", "package", "list", "packages"], retries=1)
                adb_cmd(adb_base, ["shell", "ls", "/sdcard/Android"], retries=1)
                return
        except RuntimeError:
            pass
        time.sleep(2)
    raise RuntimeError("device did not finish boot")


def dump_ui(adb_base):
    raw = adb_cmd(adb_base, ["exec-out", "uiautomator", "dump", "/dev/tty"])
    start = raw.find("<?xml")
    if start == -1:
        raise RuntimeError("uiautomator xml not found")
    xml = raw[start:]
    end = xml.rfind("</hierarchy>")
    if end == -1:
        raise RuntimeError("uiautomator hierarchy end not found")
    xml = xml[: end + len("</hierarchy>")]
    return ET.fromstring(xml)


def parse_bounds(bounds):
    match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
    if not match:
        raise RuntimeError(f"bad bounds: {bounds}")
    x1, y1, x2, y2 = map(int, match.groups())
    return x1, y1, x2, y2


def node_center(node):
    x1, y1, x2, y2 = parse_bounds(node.attrib["bounds"])
    return (x1 + x2) // 2, (y1 + y2) // 2


def find_by_text(root, text):
    for node in root.iter("node"):
        if node.attrib.get("text") == text:
            return node
    return None


def find_expand_for_difficulty(root):
    expands = []
    for node in root.iter("node"):
        if node.attrib.get("content-desc") == "Expand":
            _, y1, _, _ = parse_bounds(node.attrib["bounds"])
            expands.append((y1, node))
    if not expands:
        return None
    expands.sort(key=lambda item: item[0])
    return expands[-1][1]


def tap_node(adb_base, node):
    x, y = node_center(node)
    adb_cmd(adb_base, ["shell", "input", "tap", str(x), str(y)])


def check_crash(adb_base, package_name):
    logs = adb_cmd(adb_base, ["logcat", "-d", "*:E"])
    if (
        ("FATAL" in logs or "Fatal signal" in logs or "SIGSEGV" in logs)
        and (package_name in logs or "libkeen" in logs)
    ):
        return logs
    return None


def ensure_menu(adb_base):
    for _ in range(8):
        root = dump_ui(adb_base)
        if find_by_text(root, "START GAME"):
            return
        if find_by_text(root, "System UI isn't responding"):
            wait_btn = find_by_text(root, "Wait")
            if wait_btn:
                tap_node(adb_base, wait_btn)
                time.sleep(1.0)
                continue
        adb_cmd(adb_base, ["shell", "input", "keyevent", "4"])
        time.sleep(0.8)
    raise RuntimeError("could not reach main menu")


def parse_list(value, default):
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Smoke test Classik UI flows on an emulator.")
    parser.add_argument("--package", default="org.yegie.keenkenning.classik")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--difficulties", default=None)
    args = parser.parse_args()

    modes = parse_list(args.modes, DEFAULT_MODES)
    diffs = parse_list(args.difficulties, DEFAULT_DIFFS)

    adb_base = build_adb(args.serial)
    wait_for_boot(adb_base)
    adb_cmd(adb_base, ["logcat", "-c"])
    adb_cmd(
        adb_base,
        ["shell", "monkey", "-p", args.package, "-c", "android.intent.category.LAUNCHER", "1"],
    )
    time.sleep(2.0)
    ensure_menu(adb_base)

    for diff in diffs:
        ensure_menu(adb_base)
        root = dump_ui(adb_base)
        expand = find_expand_for_difficulty(root)
        if not expand:
            raise RuntimeError("difficulty expand not found")
        tap_node(adb_base, expand)
        time.sleep(0.8)

        root = dump_ui(adb_base)
        diff_node = find_by_text(root, diff)
        if not diff_node:
            print(f"Difficulty not found: {diff}")
            adb_cmd(adb_base, ["shell", "input", "keyevent", "4"])
            time.sleep(0.8)
            continue

        tap_node(adb_base, diff_node)
        time.sleep(0.8)

        for mode in modes:
            ensure_menu(adb_base)
            root = dump_ui(adb_base)
            mode_node = find_by_text(root, mode)
            if not mode_node:
                print(f"Mode not found: {mode} (diff={diff})")
                continue

            print(f"Starting game: diff={diff} mode={mode}")
            tap_node(adb_base, mode_node)
            time.sleep(0.5)

            root = dump_ui(adb_base)
            start_node = find_by_text(root, "START GAME")
            if not start_node:
                raise RuntimeError("START GAME button not found")

            adb_cmd(adb_base, ["logcat", "-c"])
            tap_node(adb_base, start_node)
            time.sleep(3.0)

            crash = check_crash(adb_base, args.package)
            if crash:
                print(f"CRASH after diff={diff} mode={mode}")
                print(crash)
                return 1

            adb_cmd(adb_base, ["shell", "input", "keyevent", "4"])
            time.sleep(1.0)
            root = dump_ui(adb_base)
            if not find_by_text(root, "START GAME"):
                adb_cmd(adb_base, ["shell", "input", "keyevent", "4"])
                time.sleep(1.0)

    print("UI smoke test complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
