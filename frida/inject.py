#!/usr/bin/env python3
"""Resident Frida injector for Temple Run (equivalent to: frida -U -p <pid> -l temple_run_hook.js).

Usage: python inject.py [script.js] [target]
  script.js : path to the Frida script (default: temple_run_hook.js next to this file)
  target    : process name or pid (default: com.imangi.templerun)
"""
import frida, sys, os, time

here = os.path.dirname(os.path.abspath(__file__))
script_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "temple_run_hook.js")
target = sys.argv[2] if len(sys.argv) > 2 else "com.imangi.templerun"

with open(script_path, "r", encoding="utf-8") as f:
    source = f.read()

def on_message(message, data):
    if message.get("type") == "log":
        print(message.get("payload", ""), flush=True)
    elif message.get("type") == "error":
        print("ERROR:", message.get("stack", message), flush=True)
    else:
        print("MSG:", message, flush=True)

dev = frida.get_usb_device(timeout=15)
try:
    pid = int(target)
    session = dev.attach(pid)
except ValueError:
    session = dev.attach(target)

script = session.create_script(source)
script.on("message", on_message)
script.load()
print("=== script loaded into %s ===" % target, flush=True)

# Stay resident so hooks/sockets keep running (self-terminate after 30 min).
deadline = time.time() + 1800
try:
    while time.time() < deadline:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    session.detach()
