/*
 * trace_logger.js  —  Frida instrumentation that records REAL Temple Run phone play
 *                     for offline sim-fidelity testing.
 *
 * Based on temple_run_hook.js (same hooks + upgrade-maxing + the 9998 action server +
 * the 9999 state stream), but on EVERY tick it ALSO emits, via Frida send(), a rich
 * "trace frame" carrying the data needed to REPLAY a run deterministically through the
 * headless sim. See docs/SIM_CONTRACT.md §1 for the canonical JSONL schema; the host
 * recorder (scripts/device/record_traces.py) reads these send() frames and writes the JSONL.
 *
 * Each trace frame (message {tr_frame: {...}}) contains:
 *   - state : the 22 canonical raw fields (constants.STATE_FIELDS) — identical to snapshot()
 *   - hidden: extra real state for deterministic replay (DistanceTraveled, lane best-effort,
 *             ground_height, active slide/strafe/stumble timer remainders, death_type) — any
 *             value we cannot read is emitted as null (the harness tolerates nulls).
 *   - track : the next K=8 PathElements AHEAD of the player (follow the forward/facing neighbor),
 *             each with {type, level, obstacles, is_turn, can_turn_left, can_turn_right}.
 *   - action: the action byte the Update hook DISPATCHED on the tick that produced this frame
 *             (null when no gameplay action ran that tick — i.e. an idle/reset tick).
 *
 * The trace frame is emitted from the MAIN thread inside the GameController.Update hook (so
 * the native CanTurn / IsTurn calls and the track walk are thread-safe and consistent with the
 * frame the player is actually on). The 9999 broadcast still runs on the POLL_MS timer too, so
 * an env / smoke test can read state exactly as with temple_run_hook.js.
 *
 * Target: com.imangi.templerun 1.39.0 (Unity 2022.3.62f2, IL2CPP, arm64-v8a). Offsets/RVAs are
 * from Il2CppDumper dump.cs of THIS exact build; verify if the APK is updated. Run on NATIVE
 * arm64 (real device / arm64 image) — Frida cannot hook ARM-via-x86-translation.
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
  Velocity: 0x34,   // Vector3 (from MovingObject base): x@0x34 y@0x38 z@0x3C
  Location: 0x14,   // MovingObject.Location (Vector3): x@0x14 y@0x18 z@0x1C  (for best-effort lane)
  Facing: 0x78,     // CommonPlayer.Facing (PathDirection enum) — player heading
  GameOverInterface: 0x1A0, // CharacterPlayer -> EndGameUI*
  // CharacterPlayer extras for the `hidden` block:
  DeathType: 0xE0,            // CharacterPlayer.DeathTypes (enum int)
  TimeSinceSlideStart: 0x104, SlideDuration: 0x108,
  TimeSinceStumbleStart: 0x114, StumbleDuration: 0x118,
  TimeSinceStrafeStart: 0x128, StrafeDuration: 0x12C,
  GroundHeight: 0x134,        // float
  // GameController
  TutorialID: 0x48, GamePlayer: 0x70, IsGameOver: 0xDE,
  IsTutorialMode: 0x104, NearestElement: 0x88, RecordManager: 0xB0,
  DistanceTraveled: 0x124,    // GameController.DistanceTraveled (float)
  // PathElement
  PE_Type: 0x30, PE_Level: 0x34, PE_Obstacles: 0x38,
  // PathElement directional neighbors (PathElement*):
  PE_North: 0x50, PE_East: 0x58, PE_West: 0x60, PE_South: 0x68,
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
  RunAgainClicked: 0x12BCCF0,   // EndGameUI "Rejouer" button handler (full UI restart from game-over)
  MainMenuUpdate: 0x12C94C8,    // MainMenu.Update() — runs each frame at the menu; used to capture the instance
  OnPlayClicked: 0x12CA204,     // MainMenu.OnPlayClicked() — the "JOUER"/Play button handler (start a run from the menu)
  GetRunVelocity: 0x12982D8,    // CharacterPlayer.GetRunVelocity() -> float (forward run speed)
  PlayerPrefs_GetInt: 0x24B37B8, // PlayerPrefs.GetInt(string key, int default) -> int
  // PathElement turn queries (origin = player's PathDirection / Facing):
  CanTurnLeft: 0x12E2E88,        // bool PathElement.CanTurnLeft(PathDirection origin)
  CanTurnRight: 0x12E2F44,       // bool PathElement.CanTurnRight(PathDirection origin)
  IsTurn: 0x12E28B0,             // bool PathElement.IsTurn()
};

// StoreItemType values to max out: AngelWings,HeadStart,HeadStartMega,Vacuum,Invincibility,CoinBonus,CoinValue,Boost
const UPGRADE_TYPES = [0, 1, 2, 5, 6, 7, 8, 9];
const PLAYER_ID = 1; // confirmed: PlayerManager.GetActivePlayer() returns 1

const STATE_PORT = 9999;
const ACTION_PORT = 9998;
const POLL_MS = 100;
const TRACK_K = 8;   // how many PathElements ahead to walk for the `track` block

// ObstacleType -> required reaction
const MUST_JUMP_TYPES = [1, 4];   // kObstacleHole, kObstacleTreeJump
const MUST_SLIDE_TYPES = [5, 6];  // kObstacleTreeSlide, kObstacleFlameTower

// PathDirection enum: None0 North1 East2 South3 West4
const DIR_NORTH = 1, DIR_EAST = 2, DIR_SOUTH = 3, DIR_WEST = 4;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let base = null;
let gcInstance = null;   // GameController*
let rmInstance = null;   // RecordManager*
let mmInstance = null;   // MainMenu* (captured in MainMenu.Update; valid while at the menu)
let mmSeenAtTick = -1000; // updateTicks value when MainMenu.Update last ran (recency gate for byte 9)
const fn = {};           // resolved NativeFunctions
const pendingActions = []; // action codes queued from the action socket
const stateClients = []; // SocketConnection output streams subscribed to state
const NULLPTR = ptr(0);
let updateTicks = 0;   // increments every GameController.Update — proves the main loop is ticking
let upgradesMaxed = false;
let cachedRunVel = 0;  // GetRunVelocity() sampled on the main thread (Update), read by the broadcast thread
let cachedCanLeft = false, cachedCanRight = false, cachedIsTurn = false; // turn availability at the nearest element

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
  fn.OnPlayClicked = new NativeFunction(base.add(RVA.OnPlayClicked), 'void', ['pointer', 'pointer']);     // (MainMenu*, mi) — start a run from the menu
  fn.GetRunVelocity = new NativeFunction(base.add(RVA.GetRunVelocity), 'float', ['pointer', 'pointer']);   // (CharacterPlayer*, mi) -> float
  fn.CanTurnLeft = new NativeFunction(base.add(RVA.CanTurnLeft), 'bool', ['pointer', 'int', 'pointer']);   // (PathElement*, origin, mi) -> bool
  fn.CanTurnRight = new NativeFunction(base.add(RVA.CanTurnRight), 'bool', ['pointer', 'int', 'pointer']); // (PathElement*, origin, mi) -> bool
  fn.IsTurn = new NativeFunction(base.add(RVA.IsTurn), 'bool', ['pointer', 'pointer']);                    // (PathElement*, mi) -> bool
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

  // MainMenu.Update(): runs each frame while at the main menu -> capture the instance so byte 9
  // (PlayFromMenu) can press "JOUER". mmSeenAtTick gates the call to "currently at the menu".
  Interceptor.attach(base.add(RVA.MainMenuUpdate), {
    onEnter(args) { mmInstance = args[0]; mmSeenAtTick = updateTicks; }
  });

  // GameController.Update(): runs every frame on the MAIN thread -> safe place to dispatch actions,
  // sample velocity/turn queries, AND emit the per-tick trace frame.
  Interceptor.attach(base.add(RVA.Update), {
    onEnter(args) {
      gcInstance = args[0];
      updateTicks++;
      if (!upgradesMaxed && maxAllUpgrades(gcInstance)) upgradesMaxed = true; // one-shot, on main thread
      const cp = gcInstance.add(OFF.GamePlayer).readPointer(); // CharacterPlayer*
      if (!cp.isNull()) {
        try { cachedRunVel = fn.GetRunVelocity(cp, NULLPTR); } catch (_) {} // main-thread sample
        // turn availability at the nearest element, via the game's own queries (origin = player Facing)
        try {
          const pe = gcInstance.add(OFF.NearestElement).readPointer();
          if (!pe.isNull()) {
            const facing = cp.add(OFF.Facing).readS32();
            cachedCanLeft = !!fn.CanTurnLeft(pe, facing, NULLPTR);
            cachedCanRight = !!fn.CanTurnRight(pe, facing, NULLPTR);
            cachedIsTurn = !!fn.IsTurn(pe, NULLPTR);
          } else { cachedCanLeft = cachedCanRight = cachedIsTurn = false; }
        } catch (_) {}
      }

      // Drain the action queue (main thread). Record the action(s) dispatched THIS tick so the
      // trace frame can be tagged with the action that produced it. If several queued at once,
      // we tag the frame with the LAST dispatched gameplay byte (the env sends 1 byte/step, so in
      // practice there is at most one gameplay action per tick).
      let dispatchedThisTick = null;
      while (pendingActions.length) {
        const a = pendingActions.shift();
        dbg('dispatch', a, 'ticks', updateTicks, 'gc', gcInstance.toString(), 'cp', cp.toString());
        try { dispatchAction(a, gcInstance, cp); dbg('dispatch_ok', a); }
        catch (e) { dbg('dispatch_err', a, String(e)); }
        if (a >= 0 && a <= 6) dispatchedThisTick = a; // only gameplay/NoOp bytes tag a frame
      }

      // Emit the rich trace frame for the host recorder (main-thread, consistent snapshot).
      try { emitTraceFrame(dispatchedThisTick); }
      catch (e) { dbg('trace_err', String(e)); }
    }
  });

  log('hooks installed');
}

// Max every powerup upgrade to its store max, read back to confirm, and persist to disk.
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
    case 7: // RestartGame() (resets game LOGIC only; the run stays in game-over limbo)
      if (!gc.isNull()) { fn.RestartGame(gc, NULLPTR); dbg('restartgame'); }
      break;
    case 8: // RunAgainClicked() — start a fresh run from the GAME-OVER screen.
      if (!cp.isNull()) {
        const egui = cp.add(OFF.GameOverInterface).readPointer();
        if (!egui.isNull()) { fn.RunAgainClicked(egui, NULLPTR); dbg('runagain'); }
        else { dbg('runagain_no_egui'); } // not at game-over (e.g. at the main menu) -> use byte 9
      }
      break;
    case 9: // OnPlayClicked() — press "JOUER" to start a run from the MAIN MENU.
      if (mmInstance !== null && !mmInstance.isNull() && (updateTicks - mmSeenAtTick) <= 5) {
        fn.OnPlayClicked(mmInstance, NULLPTR); dbg('play_from_menu');
      } else { dbg('play_from_menu_skip', updateTicks - mmSeenAtTick); }
      break;
    default: log('unknown action ' + a);
  }
}

// ---------------------------------------------------------------------------
// State snapshot (the 22 canonical raw fields) — pure memory reads.
// Order/names mirror constants.STATE_FIELDS exactly.
// ---------------------------------------------------------------------------
function snapshot() {
  const s = {
    is_dead: false, is_jumping: false, is_sliding: false, is_on_ground: false,
    is_stumbling: false, has_invincibility: false, has_angel_wings: false,
    has_vacuum: false, has_boost: false, is_mega_boost: false, is_strafing: false,
    score: 0, velocity: 0.0, path_type: -1, path_level: -1, facing: -1,
    must_jump: false, must_slide: false, is_game_over: false,
    can_turn_left: false, can_turn_right: false, is_turn: false,
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
  // turn availability (computed on the main thread in the Update hook via the game's queries)
  s.can_turn_left = cachedCanLeft;
  s.can_turn_right = cachedCanRight;
  s.is_turn = cachedIsTurn;
  return s;
}

// ---------------------------------------------------------------------------
// Hidden block — extra real state for deterministic replay. Anything we cannot
// read confidently is emitted as null (the harness tolerates nulls).
// ---------------------------------------------------------------------------
function hiddenSnapshot() {
  const h = {
    distance: null, lane: null, ground_height: null,
    slide_t: null, strafe_t: null, stumble_t: null, death_type: null,
    location: null, // [x,y,z] best-effort, useful for the sim (extra; harness ignores unknown keys)
  };
  if (gcInstance === null || gcInstance.isNull()) return h;
  h.distance = readF32(gcInstance, OFF.DistanceTraveled);

  const cp = gcInstance.add(OFF.GamePlayer).readPointer();
  if (cp.isNull()) return h;

  h.ground_height = readF32(cp, OFF.GroundHeight);
  h.death_type = readI32(cp, OFF.DeathType);

  // Active-timer REMAINDERS, best-effort: duration - elapsed while the flag is set, else null.
  if (readBool(cp, OFF.IsSliding)) {
    const dur = readF32(cp, OFF.SlideDuration), el = readF32(cp, OFF.TimeSinceSlideStart);
    h.slide_t = Math.max(0, dur - el);
  }
  if (readBool(cp, OFF.IsStrafing)) {
    const dur = readF32(cp, OFF.StrafeDuration), el = readF32(cp, OFF.TimeSinceStrafeStart);
    h.strafe_t = Math.max(0, dur - el);
  }
  if (readBool(cp, OFF.IsStumbling)) {
    const dur = readF32(cp, OFF.StumbleDuration), el = readF32(cp, OFF.TimeSinceStumbleStart);
    h.stumble_t = Math.max(0, dur - el);
  }

  // Best-effort lane (L=-1 / C=0 / R=1): bucket the player's LATERAL coordinate relative to the
  // nearest element's lateral coordinate. Which world axis is "lateral" depends on facing:
  //   heading N/S -> lateral is X ; heading E/W -> lateral is Z.
  // Lanes are ~one third of a 60-unit tile apart; we threshold the signed offset. This is a
  // heuristic — if we can't read a coherent reference we leave lane = null.
  try {
    const facing = readI32(cp, OFF.Facing);
    const px = readF32(cp, OFF.Location);          // Location.x @0x14
    const pz = readF32(cp, OFF.Location + 0x8);    // Location.z @0x1C
    h.location = [px, readF32(cp, OFF.Location + 0x4), pz];
    const pe = gcInstance.add(OFF.NearestElement).readPointer();
    if (!pe.isNull() && facing >= 1 && facing <= 4) {
      // PathElement is a DynamicElement (MonoBehaviour) — its world transform isn't a plain
      // field we can trust by offset across builds, so derive the lane purely from the player's
      // lateral coordinate modulo the 60-unit grid (lane centers at ~ -20/0/+20 within a tile).
      const lateral = (facing === DIR_NORTH || facing === DIR_SOUTH) ? px : pz;
      const TILE = 60.0, LANE = TILE / 3.0; // ~20 units per lane
      // offset of the player from the tile center, wrapped into [-TILE/2, +TILE/2)
      let off = lateral % TILE;
      if (off >= TILE / 2) off -= TILE;
      if (off < -TILE / 2) off += TILE;
      if (off < -LANE / 2) h.lane = (facing === DIR_NORTH || facing === DIR_EAST) ? -1 : 1;
      else if (off > LANE / 2) h.lane = (facing === DIR_NORTH || facing === DIR_EAST) ? 1 : -1;
      else h.lane = 0;
    }
  } catch (_) { h.lane = null; }

  return h;
}

// ---------------------------------------------------------------------------
// Track block — walk the next K PathElements ahead of the player, in travel order.
// "Forward" = the neighbor matching the player's heading (Facing). At a turn the
// straight-ahead neighbor may be null; we follow forward-by-facing and stop on null.
// Each element: {type, level, obstacles, is_turn, can_turn_left, can_turn_right}.
// ---------------------------------------------------------------------------
function forwardNeighborOffset(facing) {
  switch (facing) {
    case DIR_NORTH: return OFF.PE_North;
    case DIR_EAST: return OFF.PE_East;
    case DIR_SOUTH: return OFF.PE_South;
    case DIR_WEST: return OFF.PE_West;
    default: return OFF.PE_North; // unknown heading -> default to North (best-effort)
  }
}

function readPathElement(pe, facing) {
  const obs = readI32(pe, OFF.PE_Obstacles);
  let isTurn = false, canL = false, canR = false;
  try { isTurn = !!fn.IsTurn(pe, NULLPTR); } catch (_) {}
  if (facing >= 1 && facing <= 4) {
    try { canL = !!fn.CanTurnLeft(pe, facing, NULLPTR); } catch (_) {}
    try { canR = !!fn.CanTurnRight(pe, facing, NULLPTR); } catch (_) {}
  }
  return {
    type: readI32(pe, OFF.PE_Type),
    level: readI32(pe, OFF.PE_Level),
    obstacles: obs,
    is_turn: isTurn,
    can_turn_left: canL,
    can_turn_right: canR,
  };
}

function trackSnapshot() {
  const out = [];
  if (gcInstance === null || gcInstance.isNull()) return out;
  const cp = gcInstance.add(OFF.GamePlayer).readPointer();
  let facing = -1;
  if (!cp.isNull()) facing = readI32(cp, OFF.Facing);

  let pe = gcInstance.add(OFF.NearestElement).readPointer();
  const seen = {}; // guard against cycles in the linked graph
  for (let i = 0; i < TRACK_K; i++) {
    if (pe.isNull()) break;
    const key = pe.toString();
    if (seen[key]) break;     // cycle / repeated element -> stop
    seen[key] = true;
    out.push(readPathElement(pe, facing));
    // Advance to the forward neighbor for the player's heading. (We follow forward-by-facing as
    // the documented walk model — facing may change at a future turn, but we cannot know that
    // here.) Stop on a null neighbor (end of the generated track ahead).
    try { pe = pe.add(forwardNeighborOffset(facing)).readPointer(); }
    catch (_) { break; }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Trace frame: emitted every Update tick from the main thread.
// ---------------------------------------------------------------------------
function emitTraceFrame(actionByte) {
  const frame = {
    seq: updateTicks,
    action: (actionByte === null || actionByte === undefined) ? null : actionByte,
    state: snapshot(),
    hidden: hiddenSnapshot(),
    track: trackSnapshot(),
  };
  try { send({ tr_frame: frame }); } catch (_) {}
}

// ---------------------------------------------------------------------------
// 9999 state broadcast (kept identical to temple_run_hook.js so the env/smoke test works).
// ---------------------------------------------------------------------------
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
// Sockets (run inside the target process, i.e. on the phone)
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
              if (b >= 0x30 && b <= 0x39) b -= 0x30; // ASCII '0'..'9'
              if (b >= 0 && b <= 9) pendingActions.push(b);
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
      setInterval(function () {
        dbg('alive', 'ticks', updateTicks, 'gc', gcInstance ? gcInstance.toString() : 'null', 'pending', pendingActions.length);
      }, 2000);
      dbg('ready', 'base', base.toString());
      log('ready: state->' + STATE_PORT + ' action<-' + ACTION_PORT + ' trace->send()');
    } else if (++tries > 100) {
      clearInterval(t);
      log('ERROR: libil2cpp.so not found after 10s');
    }
  }, 100);
})();
