#!/usr/bin/env python3
"""Inject a Frida script into a frida-GADGET (non-root) over a TCP endpoint.

Usage: python gadget_inject.py [script.js] [host:port]
  default script = temple_run_hook.js next to this file
  default host   = 127.0.0.1:27042  (adb forward tcp:27042 tcp:27042 first)
"""
import frida, sys, os, time

here = os.path.dirname(os.path.abspath(__file__))
script_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "temple_run_hook.js")
host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:27042"

with open(script_path, "r", encoding="utf-8") as f:
    source = f.read()

def on_message(message, data):
    if message.get("type") == "log":
        print(message.get("payload", ""), flush=True)
    elif message.get("type") == "send":
        print("DBG:", message.get("payload"), flush=True)
    elif message.get("type") == "error":
        print("ERROR:", message.get("stack", message), flush=True)
    else:
        print("MSG:", message, flush=True)

dm = frida.get_device_manager()
device = dm.add_remote_device(host)
session = device.attach("Gadget")           # the gadget exposes a single process named "Gadget"
script = session.create_script(source)
script.on("message", on_message)
script.load()
print("=== gadget script loaded via %s ===" % host, flush=True)

deadline = time.time() + 3600
try:
    while time.time() < deadline:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    session.detach()
