#!/usr/bin/env python3
"""Build a non-root Frida-gadget-injected Temple Run APK (unsigned).

Strategy (no smali editing): add a DT_NEEDED("libgadget.so") to libmain.so so the
Android linker auto-loads the gadget when Unity loads libmain.so. Add the gadget +
a config (listen 127.0.0.1:<port>, resume on load). Drop the old signature; the caller
then zipaligns and re-signs with apksigner (see docs/SETUP.md).

Inputs:
  --universal  a single universal APK (merge the split APKs first, e.g. with APKEditor:
               `java -jar APKEditor.jar m -i <dir-with-splits> -o universal.apk`)
  --gadget     the frida-gadget arm64 .so (frida-gadget-<ver>-android-arm64.so, decompressed)

Example:
  python tools/build_gadget_apk.py --universal universal.apk --gadget libgadget.so --out patched-unsigned.apk
  # then:
  zipalign -f 4 patched-unsigned.apk patched-aligned.apk
  apksigner sign --ks tr-debug.keystore --ks-pass pass:android --out TempleRun-gadget.apk patched-aligned.apk
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile

import lief


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universal", required=True, help="input universal (merged) APK")
    p.add_argument("--gadget", required=True, help="frida-gadget arm64 .so")
    p.add_argument("--out", default="patched-unsigned.apk", help="output unsigned APK")
    p.add_argument("--libpath", default="lib/arm64-v8a/libmain.so",
                   help="APK-internal path of the Unity lib to patch (default: libmain.so)")
    p.add_argument("--gadget-config-port", type=int, default=27042,
                   help="port the embedded gadget listens on (default: 27042)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    zin = zipfile.ZipFile(args.universal, "r")
    libmain = zin.read(args.libpath)
    print("libmain.so original size:", len(libmain))

    # Patch libmain.so: add DT_NEEDED libgadget.so
    tmpd = tempfile.mkdtemp()
    p_in, p_out = os.path.join(tmpd, "libmain.so"), os.path.join(tmpd, "libmain_patched.so")
    open(p_in, "wb").write(libmain)
    elf = lief.ELF.parse(p_in)
    before = list(elf.libraries)
    elf.add_library("libgadget.so")
    elf.write(p_out)
    elf2 = lief.ELF.parse(p_out)
    print("NEEDED before:", before)
    print("NEEDED after :", list(elf2.libraries))
    assert "libgadget.so" in elf2.libraries, "DT_NEEDED injection failed"
    libmain_patched = open(p_out, "rb").read()

    gadget = open(args.gadget, "rb").read()
    config = json.dumps({
        "interaction": {
            "type": "listen", "address": "127.0.0.1",
            "port": args.gadget_config_port, "on_load": "resume",
        }
    }).encode()

    lib_dir = os.path.dirname(args.libpath)  # e.g. lib/arm64-v8a
    SIG = (".RSA", ".DSA", ".EC", ".SF")
    zout = zipfile.ZipFile(args.out, "w")
    n_copied = 0
    for item in zin.infolist():
        name = item.filename
        if name.startswith("META-INF/") and (name.endswith(SIG) or name.endswith("MANIFEST.MF")):
            continue  # drop old signature; we re-sign
        data = libmain_patched if name == args.libpath else zin.read(name)
        zi = zipfile.ZipInfo(name, date_time=item.date_time)
        zi.compress_type = item.compress_type
        zi.external_attr = item.external_attr
        zout.writestr(zi, data)
        n_copied += 1
    zout.writestr(f"{lib_dir}/libgadget.so", gadget, zipfile.ZIP_DEFLATED)
    zout.writestr(f"{lib_dir}/libgadget.config.so", config, zipfile.ZIP_DEFLATED)
    zout.close()
    zin.close()
    print("entries copied:", n_copied, "+ gadget + config")
    print("written:", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
