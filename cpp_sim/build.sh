#!/bin/sh
# Build the C++ sim into a shared library next to this script.
#
#   sh cpp_sim/build.sh
#
# Toolchain on this box: C:/Users/bartb/gcc/bin/g++.exe -- GCC 4.6.3 (2011, MinGW).
# It needs -std=c++0x for <cstdint>; there is no C++11 stdlib, no std::thread and
# no sanitizers, so the source is deliberately plain C-with-structs.
#
# -O2 (not -O3): measured identical throughput here and compiles faster.
# -ffp-contract=off / -fno-fast-math: the port must reproduce the Python
# reference's double arithmetic step-exactly, so NO fused-multiply-add
# contraction and no reassociation are allowed.
set -e
CXX="${CXX:-C:/Users/bartb/gcc/bin/g++.exe}"
HERE="$(cd "$(dirname "$0")" && pwd)"

case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) OUT="$HERE/sim.dll" ;;
  *)                               OUT="$HERE/sim.so"  ;;
esac

"$CXX" -O2 -std=c++0x -fno-fast-math -ffp-contract=off -Wall \
       -shared -o "$OUT" "$HERE/sim.cpp"
echo "built $OUT"
