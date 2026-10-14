"""Measure device sustained-strafe-hold lateral dynamics: how does `lane` evolve when a strafe is
HELD on a straight, and at what |lane| does it Fall? Feeds the sim's over-strafe death model."""
import socket, json, time, sys

def sc():
    s = socket.create_connection(('127.0.0.1', 9999), timeout=8); s.settimeout(8)
    return s, s.makefile()

def ac():
    s = socket.create_connection(('127.0.0.1', 9998), timeout=8); return s

def send(a, byte):
    try: a.sendall(bytes([byte])); return True
    except Exception: return False

def read_state(f):
    line = f.readline()
    if not line: return None
    try: return json.loads(line)
    except Exception: return None

def wait_run(f, a, timeout=25):
    """Reset and wait until a run is progressing on a straight."""
    send(a, 8); time.sleep(0.4); send(a, 9); time.sleep(1.5)
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = read_state(f)
        if d and not d.get('is_dead') and (d.get('distance') or 0) > 40 and d.get('path_type') in (1, 2):
            return d
    return None

def hold_trial(hold_byte, label):
    s, f = sc(); a = ac()
    d = wait_run(f, a)
    if not d:
        print(f'  {label}: no run'); s.close(); a.close(); return None
    traj = []
    t0 = time.time()
    while time.time() - t0 < 8.0:
        send(a, hold_byte)
        d = read_state(f)
        if not d: continue
        traj.append((round(time.time() - t0, 2), round(d.get('lane', 0), 1), int(d.get('velocity', 0)),
                     d.get('path_type'), bool(d.get('is_dead')), d.get('death_type')))
        if d.get('is_dead'):
            break
        time.sleep(0.09)
    s.close(); a.close()
    return traj

for trial in range(4):
    byte = 1 if trial % 2 == 0 else 0     # alternate StrafeRight / StrafeLeft
    lab = f'hold-{"R" if byte==1 else "L"} #{trial}'
    tr = hold_trial(byte, lab)
    if not tr:
        continue
    lanes = [x[1] for x in tr]
    last = tr[-1]
    # print the lane trajectory (subsampled) + death
    print(f'{lab}: frames={len(tr)}  lane: {[x[1] for x in tr[::3]]}')
    print(f'   -> died={last[4]} at lane={last[1]} vel={last[2]} pt={last[3]} dtype={last[5]}  max|lane|={max(abs(x[1]) for x in tr):.1f}')
    time.sleep(1.0)
