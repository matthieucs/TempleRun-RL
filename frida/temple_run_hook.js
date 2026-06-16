/*
 * temple_run_hook.js  —  Frida instrumentation layer for an RL environment.
 *
 * Target: com.imangi.templerun 1.39.0 (Unity 2022.3.62f2, IL2CPP, arm64-v8a)
 * Offsets/RVAs are from Il2CppDumper dump.cs of THIS exact build. They are
 * valid only for this libil2cpp.so (verify if the APK is updated).
 *
 * What it does:
 *   - Resolves libil2cpp.so base at runtime.
 *   - Hooks GameController.Start()  -> forces IsTutorialMode=false, TutorialID=999, captures the GameController instance.
 *   - Hooks RecordManager.DeserializeRecordManager() -> on return, maxes every upgrade powerup level.
 *   - Hooks GameController.Update() -> drains the action queue ON THE GAME MAIN THREAD (thread-safe input).
 *   - Every 100 ms: reads game state (memory reads only) and broadcasts JSON over TCP 0.0.0.0:9999.
 *   - Listens on TCP 0.0.0.0:9998 for 1-byte action commands (0..7, raw byte or ASCII digit).
 *
 * Host wiring (ports live on the emulator, so forward them):
 *   adb forward tcp:9999 tcp:9999      # state stream  (host connects, reads JSON lines)
 *   adb forward tcp:9998 tcp:9998      # action sink   (host connects, writes 1 byte per action)
 *
 * NOTE: Frida's Interceptor/NativeFunction operate on NATIVE code. If the app runs
 * arm64 via ARM-to-x86 translation on an x86_64 emulator, the hooks below will not
 * fire and the NativeFunctions will not invoke correctly. Run this on a NATIVE arm64
 * target (arm64 system image or a real arm64 device) for it to function.
 */

'use strict';

// ---------------------------------------------------------------------------
// Config: offsets (instance field byte offsets) and RVAs (offset from .so base)
// ---------------------------------------------------------------------------
const OFF = {
  // CharacterPlayer
  IsDead: 0xD8, IsJumping: 0xF8, IsSliding: 0x100, IsOnGround: 0x130,
  IsStumbling: 0x10C, HasInvincibility: 0x154, HasAngelWings: 0x161,
  HasVacuum: 0x16C, HasBoost: 0x174, IsMegaBoost: 0x175, Score: 0x138, IsStrafing: 0x125,
  Velocity: 0x34, // Vector3 (from MovingObject base): x@0x34 y@0x38 z@0x3C
  Facing: 0x78,   // CommonPlayer.Facing (PathDirection enum) — player heading
  GameOverInterface: 0x1A0, // CharacterPlayer -> EndGameUI*
  // GameController
  TutorialID: 0x48, GamePlayer: 0x70, IsGameOver: 0xDE,
  IsTutorialMode: 0x104, NearestElement: 0x88, RecordManager: 0xB0,
  // PathElement
  PE_Type: 0x30, PE_Level: 0x34, PE_Obstacles: 0x38,
};

const RVA = {
  Start: 0x1269CE8,
  Update: 0x126C60C,
  DeserializeRecordManager: 0x128AF58,
  SetPlayerLevelForUpgradeType: 0x128D7BC,
  GetPlayerLevelForUpgradeType: 0x128D568,
  GetStoreItemMaxLevel: 0x128E5D8,
  SerializeRecordManager: 0x128BDC8,
  StrafeLeft: 0x1298874,
  StrafeRight: 0x1298908,
  Jump: 0x1297B80,   // bool Jump(float delay)
  Slide: 0x1298768,  // bool Slide()
  TurnLeft: 0x129899C,
  TurnRight: 0x1298A14,
  RestartGame: 0x126B5B8,
  RunAgainClicked: 0x12BCCF0,   // EndGameUI "Rejouer" button handler (full UI restart)
  GetRunVelocity: 0x12982D8,    // CharacterPlayer.GetRunVelocity() -> float (forward run speed)
  PlayerPrefs_GetInt: 0x24B37B8, // PlayerPrefs.GetInt(string key, int default) -> int
};

// StoreItemType values to max out: AngelWings,HeadStart,HeadStartMega,Vacuum,Invincibility,CoinBonus,CoinValue,Boost
const UPGRADE_TYPES = [0, 1, 2, 5, 6, 7, 8, 9];
const PLAYER_ID = 1; // confirmed: PlayerManager.GetActivePlayer() returns 1

const STATE_PORT = 9999;
const ACTION_PORT = 9998;
const POLL_MS = 100;

// ObstacleType -> required reaction
const MUST_JUMP_TYPES = [1, 4];   // kObstacleHole, kObstacleTreeJump
const MUST_SLIDE_TYPES = [5, 6];  // kObstacleTreeSlide, kObstacleFlameTower

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let base = null;
let gcInstance = null;   // GameController*
let rmInstance = null;   // RecordManager*
const fn = {};           // resolved NativeFunctions
const pendingActions = []; // action codes queued from the action socket
const stateClients = []; // SocketConnection output streams subscribed to state
const NULLPTR = ptr(0);
let updateTicks = 0;   // increments every GameController.Update — proves the main loop is ticking
let upgradesMaxed = false;
let cachedRunVel = 0;  // GetRunVelocity() sampled on the main thread (Update), read by the broadcast thread

function log(m) { console.log('[TR] ' + m); }
function dbg() { try { send({ tr: Array.prototype.slice.call(arguments) }); } catch (_) {} }

function strToBuf(s) {
  const a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i) & 0xff;
  return a.buffer;
}

function readBool(p, off) {
  try { return p.add(off).readU8() !== 0; } catch (_) { return false; }
}
function readI32(p, off) {
  try { return p.add(off).readS32(); } catch (_) { return 0; }
}
function readF32(p, off) {
  try { return p.add(off).readFloat(); } catch (_) { return 0.0; }
}
function readIl2CppString(p) { // Il2CppString: length @0x10, UTF-16 chars @0x14
  try {
    if (p.isNull()) return null;
    const len = p.add(0x10).readS32();
    if (len < 0 || len > 512) return null;
    return p.add(0x14).readUtf16String(len);
  } catch (_) { return null; }
}

// ---------------------------------------------------------------------------
// Resolve module base + NativeFunctions
// ---------------------------------------------------------------------------
function resolve() {
  const m = Process.findModuleByName('libil2cpp.so');
  if (m === null) return false;
  base = m.base;
  log('libil2cpp.so base = ' + base);

  // Instance methods carry a trailing IL2CPP MethodInfo* arg; we pass NULL.
  fn.SetPlayerLevel = new NativeFunction(base.add(RVA.SetPlayerLevelForUpgradeType),
    'void', ['pointer', 'int', 'int', 'int', 'pointer']);          // (RM*, playerId, itemType, level, mi)
  fn.GetMaxLevel = new NativeFunction(base.add(RVA.GetStoreItemMaxLevel),
    'int', ['pointer', 'int', 'pointer']);                          // (RM*, itemType, mi) -> int
  fn.GetPlayerLevel = new NativeFunction(base.add(RVA.GetPlayerLevelForUpgradeType),
    'int', ['pointer', 'int', 'int', 'pointer']);                   // (RM*, playerId, itemType, mi) -> int
  fn.SerializeRecord = new NativeFunction(base.add(RVA.SerializeRecordManager),
    'void', ['pointer', 'pointer']);                                // (RM*, mi) -> persist to disk
  fn.StrafeLeft = new NativeFunction(base.add(RVA.StrafeLeft), 'void', ['pointer', 'pointer']);
  fn.StrafeRight = new NativeFunction(base.add(RVA.StrafeRight), 'void', ['pointer', 'pointer']);
  fn.Jump = new NativeFunction(base.add(RVA.Jump), 'bool', ['pointer', 'float', 'pointer']);
  fn.Slide = new NativeFunction(base.add(RVA.Slide), 'bool', ['pointer', 'pointer']);
  fn.TurnLeft = new NativeFunction(base.add(RVA.TurnLeft), 'void', ['pointer', 'pointer']);
  fn.TurnRight = new NativeFunction(base.add(RVA.TurnRight), 'void', ['pointer', 'pointer']);
  fn.RestartGame = new NativeFunction(base.add(RVA.RestartGame), 'void', ['pointer', 'pointer']);
  fn.RunAgainClicked = new NativeFunction(base.add(RVA.RunAgainClicked), 'void', ['pointer', 'pointer']); // (EndGameUI*, mi)
  fn.GetRunVelocity = new NativeFunction(base.add(RVA.GetRunVelocity), 'float', ['pointer', 'pointer']);   // (CharacterPlayer*, mi) -> float
  return true;
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------
function installHooks() {
  // GameController.Start(): force-skip tutorial + capture the GameController instance.
  Interceptor.attach(base.add(RVA.Start), {
    onEnter(args) {
      gcInstance = args[0];
      try {
        gcInstance.add(OFF.IsTutorialMode).writeU8(0); // IsTutorialMode = false
        gcInstance.add(OFF.TutorialID).writeS32(999);  // TutorialID = 999
        log('GameController.Start hooked; gc=' + gcInstance + ' (tutorial skipped)');
      } catch (e) { log('Start patch error: ' + e); }
    }
  });

  // PlayerPrefs.GetInt("TR Tutorial") -> force 999 so the persisted tutorial state always reads complete.
  Interceptor.attach(base.add(RVA.PlayerPrefs_GetInt), {
    onEnter(args) { this.key = readIl2CppString(args[0]); },
    onLeave(retval) {
      if (this.key === 'TR Tutorial') { retval.replace(ptr(999)); dbg('prefs_override', 'TR Tutorial', 999); }
    }
  });

  // RecordManager.DeserializeRecordManager(): after load, max all upgrade levels.
  Interceptor.attach(base.add(RVA.DeserializeRecordManager), {
    onEnter(args) { rmInstance = args[0]; },
    onLeave() {
      if (rmInstance === null) return;
      try {
        UPGRADE_TYPES.forEach((t) => {
          const max = fn.GetMaxLevel(rmInstance, t, NULLPTR);
          fn.SetPlayerLevel(rmInstance, PLAYER_ID, t, max, NULLPTR);
        });
        log('Maxed upgrades ' + JSON.stringify(UPGRADE_TYPES) + ' on rm=' + rmInstance);
      } catch (e) { log('upgrade-max error: ' + e); }
    }
  });

  // GameController.Update(): runs every frame on the MAIN thread -> safe place to dispatch actions.
  Interceptor.attach(base.add(RVA.Update), {
    onEnter(args) {
      gcInstance = args[0];
      updateTicks++;
      if (!upgradesMaxed && maxAllUpgrades(gcInstance)) upgradesMaxed = true; // one-shot, on main thread
      const cp = gcInstance.add(OFF.GamePlayer).readPointer(); // CharacterPlayer*
      if (!cp.isNull()) { try { cachedRunVel = fn.GetRunVelocity(cp, NULLPTR); } catch (_) {} } // main-thread sample
      if (pendingActions.length === 0) return;
      while (pendingActions.length) {
        const a = pendingActions.shift();
        dbg('dispatch', a, 'ticks', updateTicks, 'gc', gcInstance.toString(), 'cp', cp.toString());
        try { dispatchAction(a, gcInstance, cp); dbg('dispatch_ok', a); }
        catch (e) { dbg('dispatch_err', a, String(e)); }
      }
    }
  });

  log('hooks installed');
}

// Max every powerup upgrade to its store max, read back to confirm, and persist to disk.
// Reached via GameController.RecordManager (@0xB0); returns false until RecordManager is set.
function maxAllUpgrades(gc) {
  const rm = gc.add(OFF.RecordManager).readPointer();
  if (rm.isNull()) return false;
  const result = {};
  UPGRADE_TYPES.forEach((t) => {
    const max = fn.GetMaxLevel(rm, t, NULLPTR);
    fn.SetPlayerLevel(rm, PLAYER_ID, t, max, NULLPTR);
    result['type' + t] = { max: max, readback: fn.GetPlayerLevel(rm, PLAYER_ID, t, NULLPTR) };
  });
  try { fn.SerializeRecord(rm, NULLPTR); } catch (e) { dbg('serialize_err', String(e)); }
  dbg('upgrades_maxed', result);
  return true;
}

function dispatchAction(a, gc, cp) {
  switch (a) {
    case 0: if (!cp.isNull()) fn.StrafeLeft(cp, NULLPTR); break;
    case 1: if (!cp.isNull()) fn.StrafeRight(cp, NULLPTR); break;
    case 2: if (!cp.isNull()) fn.Jump(cp, 0.0, NULLPTR); break;
    case 3: if (!cp.isNull()) fn.Slide(cp, NULLPTR); break;
    case 4: if (!cp.isNull()) fn.TurnLeft(cp, NULLPTR); break;
    case 5: if (!cp.isNull()) fn.TurnRight(cp, NULLPTR); break;
    case 6: /* NoOp */ break;
    case 7: // RestartGame() per spec (resets game logic; see RunAgainClicked for full UI "Rejouer" flow)
      if (!gc.isNull()) { fn.RestartGame(gc, NULLPTR); dbg('restartgame'); }
      break;
    default: log('unknown action ' + a);
  }
}

// ---------------------------------------------------------------------------
// State snapshot (pure memory reads -> safe from the poll thread)
// ---------------------------------------------------------------------------
function snapshot() {
  const s = {
    is_dead: false, is_jumping: false, is_sliding: false, is_on_ground: false,
    is_stumbling: false, has_invincibility: false, has_angel_wings: false,
    has_vacuum: false, has_boost: false, is_mega_boost: false, is_strafing: false,
    score: 0, velocity: 0.0, path_type: -1, path_level: -1, facing: -1,
    must_jump: false, must_slide: false, is_game_over: false,
  };
  if (gcInstance === null || gcInstance.isNull()) return s;

  s.is_game_over = readBool(gcInstance, OFF.IsGameOver);

  const cp = gcInstance.add(OFF.GamePlayer).readPointer();
  if (!cp.isNull()) {
    s.is_dead = readBool(cp, OFF.IsDead);
    s.is_jumping = readBool(cp, OFF.IsJumping);
    s.is_sliding = readBool(cp, OFF.IsSliding);
    s.is_on_ground = readBool(cp, OFF.IsOnGround);
    s.is_stumbling = readBool(cp, OFF.IsStumbling);
    s.has_invincibility = readBool(cp, OFF.HasInvincibility);
    s.has_angel_wings = readBool(cp, OFF.HasAngelWings);
    s.has_vacuum = readBool(cp, OFF.HasVacuum);
    s.has_boost = readBool(cp, OFF.HasBoost);
    s.is_mega_boost = readBool(cp, OFF.IsMegaBoost);
    s.score = readI32(cp, OFF.Score);
    s.is_strafing = readBool(cp, OFF.IsStrafing);
    s.facing = readI32(cp, OFF.Facing);
    s.velocity = cachedRunVel; // CharacterPlayer.GetRunVelocity() sampled on the main thread
  }

  const pe = gcInstance.add(OFF.NearestElement).readPointer();
  if (!pe.isNull()) {
    s.path_type = readI32(pe, OFF.PE_Type);
    s.path_level = readI32(pe, OFF.PE_Level);
    const obs = readI32(pe, OFF.PE_Obstacles);
    s.must_jump = MUST_JUMP_TYPES.indexOf(obs) !== -1;
    s.must_slide = MUST_SLIDE_TYPES.indexOf(obs) !== -1;
  }
  return s;
}

function broadcastState() {
  if (stateClients.length === 0) return;
  const line = JSON.stringify(snapshot()) + '\n';
  const buf = strToBuf(line);
  for (let i = stateClients.length - 1; i >= 0; i--) {
    const c = stateClients[i];
    c.output.write(buf).catch(() => { stateClients.splice(i, 1); }); // drop dead clients
  }
}

// ---------------------------------------------------------------------------
// Sockets (run inside the target process, i.e. on the emulator)
// ---------------------------------------------------------------------------
function startStateServer() {
  Socket.listen({ family: 'ipv4', host: '0.0.0.0', port: STATE_PORT }).then(async (listener) => {
    log('state server listening on ' + STATE_PORT);
    while (true) {
      let conn;
      try { conn = await listener.accept(); }
      catch (e) { dbg('state_accept_err', String(e)); break; }
      conn.setNoDelay(true);
      stateClients.push(conn);
      dbg('state_conn', stateClients.length);
    }
  }).catch((e) => log('state listen error: ' + e));
}

function startActionServer() {
  Socket.listen({ family: 'ipv4', host: '0.0.0.0', port: ACTION_PORT }).then(async (listener) => {
    log('action server listening on ' + ACTION_PORT);
    // robust accept loop: handle unlimited sequential connections; read multiple bytes per connection
    while (true) {
      let conn;
      try { conn = await listener.accept(); }
      catch (e) { dbg('accept_err', String(e)); break; }
      conn.setNoDelay(true);
      dbg('action_conn');
      (async function reader(c) {
        try {
          while (true) {
            const data = await c.input.read(64);
            if (data.byteLength === 0) break; // peer closed
            const bytes = new Uint8Array(data);
            dbg('recv', Array.prototype.slice.call(bytes), 'ticks', updateTicks);
            for (let i = 0; i < bytes.length; i++) {
              let b = bytes[i];
              if (b >= 0x30 && b <= 0x37) b -= 0x30; // ASCII '0'..'7'
              if (b >= 0 && b <= 7) pendingActions.push(b);
            }
          }
        } catch (e) { dbg('read_err', String(e)); }
      })(conn);
    }
  }).catch((e) => log('action listen error: ' + e));
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(function main() {
  let tries = 0;
  const t = setInterval(() => {
    if (resolve()) {
      clearInterval(t);
      installHooks();
      startStateServer();
      startActionServer();
      setInterval(broadcastState, POLL_MS);
      // diagnostic heartbeat: shows whether GameController.Update is still ticking + queue depth
      setInterval(function () {
        dbg('alive', 'ticks', updateTicks, 'gc', gcInstance ? gcInstance.toString() : 'null', 'pending', pendingActions.length);
      }, 2000);
      dbg('ready', 'base', base.toString());
      log('ready: state->' + STATE_PORT + ' action<-' + ACTION_PORT);
    } else if (++tries > 100) {
      clearInterval(t);
      log('ERROR: libil2cpp.so not found after 10s');
    }
  }, 100);
})();
