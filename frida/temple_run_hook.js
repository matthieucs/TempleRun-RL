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
 *   - Listens on TCP 0.0.0.0:9998 for 1-byte action commands (0..9, raw byte or ASCII digit).
 *     0..6 gameplay/NoOp; 7 RestartGame; 8 RunAgainClicked (game-over restart); 9 OnPlayClicked (menu start).
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
  DistanceTraveled: 0x124, // GameController.DistanceTraveled (float) — clean distance, no score multiplier
  // PathElement
  PE_Type: 0x30, PE_Level: 0x34, PE_Obstacles: 0x38,
  // directional neighbors (PathElement*) — the "forward" cell for 1-cell lookahead
  PathNorth: 0x50, PathEast: 0x58, PathWest: 0x60, PathSouth: 0x68,
};

// PathDirection (N1 E2 S3 W4) -> neighbor offset in that heading.
// forward: N->North E->East S->South W->West.  left = 90deg CCW, right = 90deg CW.
const FWD_NEIGHBOR = { 1: 0x50, 2: 0x58, 3: 0x68, 4: 0x60 };
const LEFT_NEIGHBOR = { 1: 0x60, 2: 0x50, 3: 0x58, 4: 0x68 };  // N->W E->N S->E W->S
const RIGHT_NEIGHBOR = { 1: 0x58, 2: 0x68, 3: 0x60, 4: 0x50 }; // N->E E->S S->W W->N

// Turn availability from GEOMETRY (which exit cell exists), not the position-dependent
// CanTurnLeft/Right query (which flickers 0/1 at the junction and is useless for lookahead).
function neighborExists(cell, off) {
  if (!off) return false;
  try { const n = cell.add(off).readPointer(); return n && !n.isNull(); } catch (_) { return false; }
}

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
  // POSITION gate: the game's real "is the player physically over this element's turn area" test.
  // IsTurn()/CanTurn* are TOPOLOGICAL (the cell HAS a branch); this is the longitudinal window in
  // which a turn actually lands on the corner. point = player horizontal pos (x,z), radius = tolerance.
  GetPosition2D: 0x12E0E9C,       // Vector2 MovingObject.GetPosition2D()  -> (x, z)
  IsPointOverTurnArea2D: 0x12E818C, // bool PathElement.IsPointOverTurnArea(Vector2 point, float radius)
};

// Player lateral offset within the cell (strafe lane). CharacterPlayer.PlayerXOffset @ 0x190.
const OFF_PlayerXOffset = 0x190;
const OFF_PE_Rectangle = 0xA0; // UnityEngine.Rect {xmin@0xA0, ymin@0xA4, w@0xA8, h@0xAC}

// StoreItemType values to max out: AngelWings,HeadStart,HeadStartMega,Vacuum,Invincibility,CoinBonus,CoinValue,Boost
const UPGRADE_TYPES = [0, 1, 2, 5, 6, 7, 8, 9];
const PLAYER_ID = 1; // confirmed: PlayerManager.GetActivePlayer() returns 1

const STATE_PORT = 9999;
const ACTION_PORT = 9998;
const POLL_MS = 100;  // 10 Hz broadcast. (Tried 20 Hz to cut turn-lag — the 20 Hz-retrained model was
                      // WORSE on device, median 180 vs 607; reverted. The turn-barrier is not the rate.)

// ObstacleType -> required reaction
// TreeStumble (3) is lane-dependent on the phone (left lane = jump-or-die, middle = stumble/2nd-hit
// kills, right = free). Jumping clears it on EVERY lane, so we report it as must_jump -> the policy
// learns ONE robust response (jump) without needing lane awareness. Matches sim _MUST_JUMP.
// LevelChange (7) marks a ZONE/ALTITUDE TRANSITION (path_level changes, e.g. 2->3 mid->high) which is
// a CREVASSE that must be jumped — captured live: obstacle=7 + path_level 2->3 + no jump => fell+died.
const MUST_JUMP_TYPES = [1, 3, 4, 7]; // Hole, TreeStumble, TreeJump, LevelChange(zone-transition crevasse)
const MUST_SLIDE_TYPES = [5, 6];      // kObstacleTreeSlide, kObstacleFlameTower

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
// 1-cell lookahead: the same queries on the forward-neighbor cell (anticipation).
let cachedNextIsTurn = false, cachedNextCanLeft = false, cachedNextCanRight = false;
let cachedNextMustJump = false, cachedNextMustSlide = false;
const HOOK_VERSION = 'nn-only-obs-11a'; // bump on each hook edit to verify the right build is injected
const DIAG_TURN = true;             // emit the lane / cell-progress diagnostic in state (calibration)
let cachedDbg = null;               // raw junction structure for diagnosis
let cachedTurnDiag = null;          // lane offset + cell progress, for calibration
// TURN BUFFERING (the big on-device fix): a native TurnLeft/Right fired at the START of the junction
// cell (cell_progress ~0.1, where the policy naturally reacts to is_turn) swings the player ~12 units
// off-center -> it falls off the exit path (measured: 9/12 on-device deaths). Firing the SAME turn at
// the cell CENTER (progress ~0.45) lands it centered every time (postLat <=2). So we don't execute the
// turn byte immediately: we LATCH it and fire when the player reaches the cell center.
const TURN_BUFFER = false;      // NN decides turns (no hook buffering). lane/cell_progress obs stay on.
const TURN_FIRE_PROG = 0.40;        // fire a buffered turn once cell_progress crosses this (cell-center)
const TURN_PENDING_MAX_AGE = 20;    // drop a buffered turn if it can't fire within N Update frames
let pendingTurn = 0;                // 0 none, 4 TurnLeft, 5 TurnRight (buffered, fires at cell center)
let pendingTurnAge = 0;
let pendingSawTurnCell = false;     // became true once we were on the turn cell while a turn is pending
let cachedLane = 0.0;               // signed lateral offset from path center (+ = right of travel)
let cachedCellProg = 0.0;           // longitudinal position within the nearest cell along travel, 0..1
let cachedRawIsTurn = false;        // nearest cell is a turn piece (IsTurn()), pre-latch
// CREVASSE JUMP-ASSIST: the zone/altitude transition (ObstacleType.LevelChange=7) is a ~2-cell GAP.
// The policy reacts to the 1-cell lookahead and jumps too EARLY -> comes down IN the gap -> falls
// (or doesn't jump at all). The right response is deterministic: jump just as you reach the gap so the
// ~0.6 s airtime spans both cells. So near a crevasse we take over the jump: suppress the policy's jump
// and fire one well-timed native Jump. Calibrate JUMP_FIRE_PROG live like the turn fire-progress.
const OBST_LEVEL_CHANGE = 7;
// WIDE GAPS need the same late, well-timed single jump as the crevasse: a 2-cell hole
// (HoleLongA/B Vert 12/13, Horz 15/16) and the zone LevelChange caps (33-36). The policy jumps too
// early (reacts to the 1-cell lookahead) and falls into the far half. Single 1-cell holes (11/14) and
// trees are fine for the policy's early jump -> NOT assisted here.
const WIDE_GAP_TYPES = [12, 13, 15, 16, 33, 34, 35, 36];
const CREVASSE_ASSIST = false;  // NN decides jumps.
const JUMP_FIRE_PROG = 0.90;        // fire the assisted jump once the pre-gap cell crosses this progress
                                    // (near the gap entry -> max airtime over the gap; 60 Hz Update = fine)
let cachedIsLevelChange = false;    // nearest cell IS a LevelChange crevasse
let cachedNextIsLevelChange = false;// forward-neighbor cell is a LevelChange crevasse (gap ahead)
let crevasseJumped = false;         // one assisted jump per crevasse (reset when clear of LevelChange)
// NARROW STRAFE-ASSIST: a NarrowLong piece (path_type 17..30) has a one-lane strip offset to one side;
// the runner (which never strafes — sits at its home center lane) falls off the edge. The subtype
// encodes the safe side: NarrowLong{Left 19-21=W / Right 22-24=E} for N/S travel, {25-27=N / 28-30=S}
// for E/W travel (device-measured); NarrowSmall 17/18 is multi-lane safe (no strafe). Strafe ONCE
// toward the safe lane on approach.
const NARROW_STRAFE = false;    // NN decides strafes.
// A narrow's safe strip is on an ABSOLUTE compass side; the strafe to reach it depends on heading.
// DEVICE-MEASURED (human capture 2026-06-26, captures/human_calib_2026-06-26.jsonl): the Vert group
// 19-24 = W/W/W/E/E/E is CORRECT (confirmed 20N->L, 21S->R, 23N->R). The Horz group 25-30 was N<->S
// SWAPPED in the old guess: it is really 25-27=N / 28-30=S (confirmed 26E->L=>N, 27W->R=>N, 29E->R=>S,
// 30W->L=>S — all 4 contradicted the old map). Small 17/18 = multi-lane safe -> no side (center ok).
const NARROW_SIDE = { 19: 'W', 20: 'W', 21: 'W', 22: 'E', 23: 'E', 24: 'E',
                      25: 'N', 26: 'N', 27: 'N', 28: 'S', 29: 'S', 30: 'S' };  // [device-measured] 25-27=N, 28-30=S
// compass side a StrafeLeft / StrafeRight moves the runner to, per facing (N1 E2 S3 W4).
const STRAFE_L_SIDE = { 1: 'W', 2: 'N', 3: 'E', 4: 'S' };
const STRAFE_R_SIDE = { 1: 'E', 2: 'S', 3: 'W', 4: 'N' };
function narrowStrafeDirFor(subtype, facing) {   // -1 strafe left, +1 strafe right, 0 none/center
  const side = NARROW_SIDE[subtype];
  if (!side) return 0;
  if (STRAFE_L_SIDE[facing] === side) return -1;
  if (STRAFE_R_SIDE[facing] === side) return 1;
  return 0;
}
// Measured geometry: home/center lane ~ -1.3, one StrafeLeft -> ~ -5, StrafeRight -> ~ +2.4 (spacing
// ~3.7). Open-loop out-and-back, so no bucket thresholds needed — just one strafe each way.
const STRAFE_COOLDOWN_TICKS = 14;   // ~0.25 s strafe settle @ 60 Hz Update — don't restrafe mid-strafe
let cachedPathType = -1;            // nearest cell PE_Type
let cachedNextPathType = -1;        // forward-neighbor cell PE_Type
let lastStrafeTick = -1000;         // updateTicks of the last assisted strafe (cooldown)
let narrowStrafeDir = 0;            // -1 strafed left / +1 right for the current narrow (0 = none/centered)
let cachedFacing = 0;              // player PathDirection (N1 E2 S3 W4), for facing-aware narrow strafe
// OBSTACLE REACTOR (safety net): the policy jumps/slides plain obstacles ~95% but misses some — esp. a
// hole right AFTER a turn (busy turning -> reacts a frame too late). If the policy hasn't acted by the
// last moment before a must_jump/must_slide cell, fire the reaction ourselves. Only fires when the
// policy DIDN'T (we check IsJumping/IsSliding first), so no double-action. Excludes wide gaps (handled).
const OBSTACLE_REACTOR = false; // no safety net — the NN must react.
const OBST_REACT_PROG = 0.72;       // fire when the pre-obstacle cell crosses this (about to enter)
let cachedObst = 0;                 // nearest cell ObstacleType enum
let cachedMustJump = false, cachedMustSlide = false;  // nearest single (non-wide-gap) must_jump/slide
let obstacleReacted = false;        // one safety reaction per obstacle (reset when none imminent)
// One-turn-per-junction latch: the real game keeps is_turn TRUE for many frames AFTER a turn
// (the sim consumes it), so a sim-trained policy over-turns into the approach cell and dies.
let prevNearest = ptr(0), turnedHere = false, prevFacing = 0;

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
  // Struct-by-value (arm64 HFA): Vector2 = {float,float} returned/passed in s0,s1 (libffi handles it).
  fn.GetPosition2D = new NativeFunction(base.add(RVA.GetPosition2D), ['float', 'float'], ['pointer', 'pointer']); // (MovingObject*, mi) -> Vector2
  fn.IsPointOverTurnArea2D = new NativeFunction(base.add(RVA.IsPointOverTurnArea2D),
    'bool', ['pointer', ['float', 'float'], 'float', 'pointer']);  // (PathElement*, Vector2 point, float radius, mi) -> bool
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
  // (PlayFromMenu) can press "JOUER". mmSeenAtTick gates the call to "currently at the menu" so we
  // never invoke OnPlayClicked on a stale pointer once a run has started.
  Interceptor.attach(base.add(RVA.MainMenuUpdate), {
    onEnter(args) { mmInstance = args[0]; mmSeenAtTick = updateTicks; }
  });

  // GameController.Update(): runs every frame on the MAIN thread -> safe place to dispatch actions.
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
            cachedFacing = facing;
            const rawIsTurn = !!fn.IsTurn(pe, NULLPTR);
            cachedRawIsTurn = rawIsTurn;
            // Player horizontal position -> signed lane (lateral offset from path center, + = right of
            // travel) + cell_progress (longitudinal position within the nearest cell, 0..1). Both drive
            // turn-buffering (fire at center) and are exposed in the obs for the next retrain.
            try {
              const pos = fn.GetPosition2D(cp, NULLPTR); // [x, z]
              const rx = readF32(pe, OFF_PE_Rectangle), rz = readF32(pe, OFF_PE_Rectangle + 4);
              const rw = readF32(pe, OFF_PE_Rectangle + 8), rh = readF32(pe, OFF_PE_Rectangle + 12);
              const cx = rx + rw / 2, cz = rz + rh / 2;
              const px = pos[0], pz = pos[1];
              // right-of-travel lateral + forward progress, per heading (N1 E2 S3 W4)
              if (facing === 1) { cachedLane = px - cx; cachedCellProg = (pz - rz) / rh; }
              else if (facing === 2) { cachedLane = -(pz - cz); cachedCellProg = (px - rx) / rw; }
              else if (facing === 3) { cachedLane = -(px - cx); cachedCellProg = 1 - (pz - rz) / rh; }
              else if (facing === 4) { cachedLane = pz - cz; cachedCellProg = 1 - (px - rx) / rw; }
              if (DIAG_TURN) cachedTurnDiag = {
                lane: Math.round(cachedLane * 10) / 10, prog: Math.round(cachedCellProg * 100) / 100,
                raw: rawIsTurn, pend: pendingTurn, fac: facing,
                lc: cachedIsLevelChange, nlc: cachedNextIsLevelChange, cj: crevasseJumped,
                pt: cachedPathType, npt: cachedNextPathType,
              };
            } catch (e) { if (DIAG_TURN) cachedTurnDiag = { err: String(e) }; }
            // Fire a buffered turn once the player reaches the cell center (lands centered, not off-edge).
            maybeFireTurn(cp);
            // New junction cell -> reset the "already turned here" latch.
            if (!pe.equals(prevNearest)) { turnedHere = false; prevNearest = pe; }
            // Turned on this junction (heading changed while ON a turn cell)? Latch it so we stop
            // exposing the turn -> the policy turns exactly ONCE (matching the sim, which consumes
            // is_turn on the turn). Without this it over-turns into the approach cell and dies.
            if (rawIsTurn && prevFacing > 0 && facing !== prevFacing) turnedHere = true;
            prevFacing = facing;
            // is_turn / turn DIRECTION (geometry: which exit cell exists, 90deg off heading) — exposed
            // only until the one turn is taken.
            cachedIsTurn = rawIsTurn && !turnedHere;
            cachedCanLeft = cachedIsTurn ? neighborExists(pe, LEFT_NEIGHBOR[facing]) : false;
            cachedCanRight = cachedIsTurn ? neighborExists(pe, RIGHT_NEIGHBOR[facing]) : false;
            cachedPathType = pe.add(OFF.PE_Type).readS32();
            cachedObst = pe.add(OFF.PE_Obstacles).readS32();
            cachedIsLevelChange = (cachedObst === OBST_LEVEL_CHANGE)
              || WIDE_GAP_TYPES.indexOf(cachedPathType) !== -1;  // wide gap (long hole / level change)
            cachedMustJump = !cachedIsLevelChange && MUST_JUMP_TYPES.indexOf(cachedObst) !== -1;
            cachedMustSlide = MUST_SLIDE_TYPES.indexOf(cachedObst) !== -1;
            // DEBUG: absolute neighbor existence + the query, to verify direction logic at junctions.
            cachedDbg = {
              fac: facing,
              N: neighborExists(pe, 0x50), E: neighborExists(pe, 0x58),
              W: neighborExists(pe, 0x60), S: neighborExists(pe, 0x68),
              qL: !!fn.CanTurnLeft(pe, facing, NULLPTR), qR: !!fn.CanTurnRight(pe, facing, NULLPTR),
            };
            // 1-cell lookahead: the forward neighbor in the player's heading.
            const noff = FWD_NEIGHBOR[facing];
            const nxt = noff ? pe.add(noff).readPointer() : NULL;
            if (nxt && !nxt.isNull()) {
              cachedNextIsTurn = !!fn.IsTurn(nxt, NULLPTR);
              cachedNextCanLeft = neighborExists(nxt, LEFT_NEIGHBOR[facing]);
              cachedNextCanRight = neighborExists(nxt, RIGHT_NEIGHBOR[facing]);
              const nobs = nxt.add(OFF.PE_Obstacles).readS32();
              cachedNextMustJump = MUST_JUMP_TYPES.indexOf(nobs) !== -1;
              cachedNextMustSlide = MUST_SLIDE_TYPES.indexOf(nobs) !== -1;
              cachedNextPathType = nxt.add(OFF.PE_Type).readS32();
              cachedNextIsLevelChange = (nobs === OBST_LEVEL_CHANGE)
                || WIDE_GAP_TYPES.indexOf(cachedNextPathType) !== -1;
            } else {
              cachedNextIsTurn = cachedNextCanLeft = cachedNextCanRight = false;
              cachedNextMustJump = cachedNextMustSlide = false;
              cachedNextIsLevelChange = false;
              cachedNextPathType = -1;
            }
            // Fire one well-timed jump for an upcoming crevasse (after lane/prog are known this frame).
            maybeJumpCrevasse(cp);
            // Strafe once toward a narrow piece's safe lane on approach.
            maybeStrafeNarrow(cp);
            // Safety net: fire jump/slide for a plain obstacle the policy hasn't reacted to in time.
            maybeReactObstacle(cp);
          } else {
            cachedCanLeft = cachedCanRight = cachedIsTurn = false;
            cachedNextIsTurn = cachedNextCanLeft = cachedNextCanRight = false;
            cachedNextMustJump = cachedNextMustSlide = false;
            cachedIsLevelChange = cachedNextIsLevelChange = false;
            cachedPathType = cachedNextPathType = -1;
            cachedMustJump = cachedMustSlide = false; cachedObst = 0;
          }
        } catch (_) {}
      }
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

// Fire a buffered turn (pendingTurn) once the player reaches the junction-cell center, so the turn
// arc lands it ON the exit path instead of ~12 units off-edge (the dominant on-device death). Called
// every Update frame (after lane/prog are refreshed) and again right after a turn byte is latched.
function maybeFireTurn(cp) {
  if (!pendingTurn || cp.isNull()) return;
  pendingTurnAge++;
  if (cachedRawIsTurn) pendingSawTurnCell = true;
  if (cachedRawIsTurn && cachedCellProg >= TURN_FIRE_PROG) {
    try { if (pendingTurn === 4) fn.TurnLeft(cp, NULLPTR); else fn.TurnRight(cp, NULLPTR); } catch (_) {}
    dbg('turn_fire', pendingTurn, 'prog', Math.round(cachedCellProg * 100) / 100);
    pendingTurn = 0; pendingTurnAge = 0; pendingSawTurnCell = false;
    return;
  }
  // Give up: we were ON the turn cell and have now left it (missed the window), or it aged out.
  if ((pendingSawTurnCell && !cachedRawIsTurn) || pendingTurnAge > TURN_PENDING_MAX_AGE) {
    dbg('turn_drop', pendingTurn, 'prog', Math.round(cachedCellProg * 100) / 100, 'age', pendingTurnAge);
    pendingTurn = 0; pendingTurnAge = 0; pendingSawTurnCell = false;
  }
}

// Jump a zone crevasse (LevelChange) at the right moment: just before the gap so the airtime spans
// both gap cells. One jump per crevasse; a safety late-jump if we somehow reach the gap still grounded.
function maybeJumpCrevasse(cp) {
  if (!CREVASSE_ASSIST || cp.isNull()) return;
  if (!cachedIsLevelChange && !cachedNextIsLevelChange) { crevasseJumped = false; return; }
  if (crevasseJumped) return;
  const airborne = readBool(cp, OFF.IsJumping);
  if (cachedNextIsLevelChange && cachedCellProg >= JUMP_FIRE_PROG && !airborne) {
    try { fn.Jump(cp, 0.0, NULLPTR); } catch (_) {}
    crevasseJumped = true;
    dbg('crevasse_jump', 'prog', Math.round(cachedCellProg * 100) / 100);
  } else if (cachedIsLevelChange && !airborne) {
    try { fn.Jump(cp, 0.0, NULLPTR); } catch (_) {}    // safety: on the gap and grounded -> jump now
    crevasseJumped = true;
    dbg('crevasse_jump_late', 'prog', Math.round(cachedCellProg * 100) / 100);
  }
}

// OPEN-LOOP narrow handling (no lane feedback -> no oscillation): on approach to a Left/Right narrow,
// strafe ONCE toward its safe side; after clearing the narrow, strafe ONCE back to re-center. The
// runner's home lane is center, so out-then-back keeps each narrow's single strafe starting from
// center. Small narrows (no target) are left centered (their safe lane is still unverified).
function maybeStrafeNarrow(cp) {
  if (!NARROW_STRAFE || cp.isNull()) return;
  if (cachedIsTurn || cachedNextIsTurn || cachedIsLevelChange || cachedNextIsLevelChange) return;
  if (updateTicks - lastStrafeTick < STRAFE_COOLDOWN_TICKS) return;
  if (readBool(cp, OFF.IsStrafing)) return;
  const nextNarrow = (cachedNextPathType >= 17 && cachedNextPathType <= 30);
  const nearNarrow = (cachedPathType >= 17 && cachedPathType <= 30);
  if (nextNarrow && narrowStrafeDir === 0) {
    const d = narrowStrafeDirFor(cachedNextPathType, cachedFacing);
    if (d === -1) { try { fn.StrafeLeft(cp, NULLPTR); } catch (_) {} narrowStrafeDir = -1; lastStrafeTick = updateTicks; dbg('narrow_strafe', 'L', cachedNextPathType, 'fac', cachedFacing); }
    else if (d === 1) { try { fn.StrafeRight(cp, NULLPTR); } catch (_) {} narrowStrafeDir = 1; lastStrafeTick = updateTicks; dbg('narrow_strafe', 'R', cachedNextPathType, 'fac', cachedFacing); }
  } else if (!nextNarrow && !nearNarrow && narrowStrafeDir !== 0) {
    if (narrowStrafeDir === -1) { try { fn.StrafeRight(cp, NULLPTR); } catch (_) {} }
    else { try { fn.StrafeLeft(cp, NULLPTR); } catch (_) {} }
    dbg('narrow_recenter', -narrowStrafeDir);
    narrowStrafeDir = 0; lastStrafeTick = updateTicks;
  }
}

// Safety-net reaction for a plain (non-wide-gap) obstacle: if the policy hasn't jumped/slid by the
// last moment before the obstacle cell, do it. Fires only when the runner isn't already jumping/sliding
// (so it never doubles a good policy play). One reaction per obstacle.
function maybeReactObstacle(cp) {
  if (!OBSTACLE_REACTOR || cp.isNull()) return;
  if (cachedIsLevelChange || cachedNextIsLevelChange) return;  // wide gaps owned by the jump-assist
  const wantJump = cachedMustJump || (cachedNextMustJump && cachedCellProg >= OBST_REACT_PROG);
  const wantSlide = cachedMustSlide || (cachedNextMustSlide && cachedCellProg >= OBST_REACT_PROG);
  if (wantJump && !obstacleReacted && !readBool(cp, OFF.IsJumping)) {
    try { fn.Jump(cp, 0.0, NULLPTR); } catch (_) {}
    obstacleReacted = true; dbg('react_jump', cachedObst, 'prog', Math.round(cachedCellProg * 100) / 100);
  } else if (wantSlide && !obstacleReacted && !readBool(cp, OFF.IsSliding)) {
    try { fn.Slide(cp, NULLPTR); } catch (_) {}
    obstacleReacted = true; dbg('react_slide', cachedObst);
  } else if (!wantJump && !wantSlide) {
    obstacleReacted = false;
  }
}

function dispatchAction(a, gc, cp) {
  switch (a) {
    case 0: if (!cp.isNull()) fn.StrafeLeft(cp, NULLPTR); break;
    case 1: if (!cp.isNull()) fn.StrafeRight(cp, NULLPTR); break;
    case 2: // Jump — suppressed near a crevasse (the assist owns that jump's timing), else normal.
      if (!cp.isNull()) {
        if (CREVASSE_ASSIST && (cachedNextIsLevelChange || cachedIsLevelChange)) { dbg('jump_suppress_crevasse'); }
        else fn.Jump(cp, 0.0, NULLPTR);
      }
      break;
    case 3: if (!cp.isNull()) fn.Slide(cp, NULLPTR); break;
    case 4: // TurnLeft — BUFFERED: latch it, fire at the cell center (maybeFireTurn) so it lands on-path.
    case 5: // TurnRight
      if (!cp.isNull()) {
        if (TURN_BUFFER) {
          // Only latch a turn the policy issues AT a junction (on it, or the next cell is one). A turn
          // on a pure straight is a policy error that would kill -> drop it (don't fire later, off-edge).
          if (cachedRawIsTurn || cachedNextIsTurn) {
            pendingTurn = a; pendingTurnAge = 0; pendingSawTurnCell = cachedRawIsTurn;
            maybeFireTurn(cp); // a turn issued LATE (already past cell center) fires immediately
          } else { dbg('turn_ignore_straight', a); }
        } else {
          if (a === 4) fn.TurnLeft(cp, NULLPTR); else fn.TurnRight(cp, NULLPTR);
        }
      }
      break;
    case 6: /* NoOp */ break;
    case 7: // RestartGame() per spec (resets game LOGIC only; the run stays in game-over limbo)
      if (!gc.isNull()) { fn.RestartGame(gc, NULLPTR); dbg('restartgame'); }
      break;
    case 8: // RunAgainClicked() — the "Rejouer" flow that STARTS a fresh run from the GAME-OVER screen.
      if (!cp.isNull()) {
        const egui = cp.add(OFF.GameOverInterface).readPointer();
        if (!egui.isNull()) { fn.RunAgainClicked(egui, NULLPTR); dbg('runagain'); }
        else { dbg('runagain_no_egui'); } // not at game-over (e.g. at the main menu) -> use byte 9
      }
      break;
    case 9: // OnPlayClicked() — press "JOUER" to start a run from the MAIN MENU.
      // Gated to "currently at the menu": MainMenu.Update must have run within the last few ticks,
      // else mmInstance is stale (a run is in progress) and calling it could crash.
      if (mmInstance !== null && !mmInstance.isNull() && (updateTicks - mmSeenAtTick) <= 5) {
        fn.OnPlayClicked(mmInstance, NULLPTR); dbg('play_from_menu');
      } else { dbg('play_from_menu_skip', updateTicks - mmSeenAtTick); }
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
    can_turn_left: false, can_turn_right: false, is_turn: false,
    distance: 0.0, obstacle: -1,
    next_is_turn: false, next_can_turn_left: false, next_can_turn_right: false,
    next_must_jump: false, next_must_slide: false, next_wide_gap: false, next_path_type: -1,
    lane: 0.0, cell_progress: 0.0, narrow_target_lane: null,
  };
  if (gcInstance === null || gcInstance.isNull()) return s;

  s.is_game_over = readBool(gcInstance, OFF.IsGameOver);
  // DistanceTraveled is the real run distance (no challenge multiplier) -> the reward signal.
  s.distance = readF32(gcInstance, OFF.DistanceTraveled);

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
    s.obstacle = obs;   // raw ObstacleType enum (diagnose crevasse / tree-root)
  }
  // turn availability (computed on the main thread in the Update hook via the game's queries)
  s.can_turn_left = cachedCanLeft;
  s.can_turn_right = cachedCanRight;
  s.is_turn = cachedIsTurn;
  s.next_is_turn = cachedNextIsTurn;
  s.next_can_turn_left = cachedNextCanLeft;
  s.next_can_turn_right = cachedNextCanRight;
  s.next_must_jump = cachedNextMustJump;
  s.next_must_slide = cachedNextMustSlide;
  s.next_wide_gap = cachedNextIsLevelChange;  // forward cell is a 2-cell gap (long hole / level change)
  s.next_path_type = cachedNextPathType;      // forward cell PE_Type (for narrow detection / mapping)
  // narrow_target_lane: PERCEPTION for the NN — which lane (-1 L/0 C/+1 R, runner-relative) the upcoming
  // narrow's strip is in (best-guess map via narrowStrafeDirFor; verify/refine from on-device deaths),
  // null if no narrow ahead. The NN reads it and decides the strafe (no hook action).
  s.narrow_target_lane = (cachedNextPathType >= 17 && cachedNextPathType <= 30)
    ? narrowStrafeDirFor(cachedNextPathType, cachedFacing) : null;
  s.lane = cachedLane;               // signed lateral offset from path center (+ = right of travel)
  s.cell_progress = cachedCellProg;  // longitudinal position within the nearest cell, 0..1
  s._hv = HOOK_VERSION;
  s._dbg = cachedDbg;
  if (DIAG_TURN) s._turn = cachedTurnDiag;
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
