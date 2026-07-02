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
  GameInput: 0x78,  // GameController.GameInput (the live TouchInput) — the DELAYED-TURN buffer state
  // GameInput fields (the HandleDelayedTurning buffer — swipe is queued, executes at the corner,
  // TimeSinceShouldTurn counts up until it EXPIRES): the real turn-execution mechanic.
  GI_JustTurned: 0x20, GI_ShouldTurnLeft: 0x21, GI_ShouldTurnRight: 0x22,
  GI_TimeSinceShouldTurn: 0x24, GI_TimeSinceLastTurn: 0x28,
  GI_ShouldJump: 0x2C, GI_TimeSinceShouldJump: 0x30,
  // PathElement
  PE_Type: 0x30, PE_Level: 0x34, PE_Obstacles: 0x38,
  // directional neighbors (PathElement*) — the "forward" cell for 1-cell lookahead
  PathNorth: 0x50, PathEast: 0x58, PathWest: 0x60, PathSouth: 0x68,
  // --- Phase-0 PROBE fields (read-only; confirm the game's own metric signals before the obs redesign) ---
  // CharacterPlayer (cp):
  PlayerXOffset: 0x190,   // float — game's OWN lateral offset from path center (the real lane; replaces px-cx)
  IsOverGround: 0x131,    // bool  — solid ground under feet vs hole/gap (distinct from IsOnGround@0x130)
  GroundHeight: 0x134,    // float — current ground world-Y (altitude)
  IsGroundHeightChangeHigher: 0x132, // bool — next ground step is UP (the mid->high crevasse)
  StumbleKillTimer: 0x11C, // float — countdown to death after a stumble ("2nd stumble within window kills")
  DidStumble: 0x124,      // bool
  DeathType: 0xE0,        // int   — CharacterPlayer.DeathTypes (1 Fall,2 Tree,3 Slide,4 Ledge,5 Burnt,8/9 Scenery)
  TimeSinceStrafeStart: 0x128, // float — strafe phase (with StrafeDuration -> fraction through a lane change)
  StrafeDuration: 0x12C,  // float
  MaxRunVelocity: 0xEC,   // float — current speed cap (normalize velocity)
  // GameController (gc): metric distance signals the policy never got
  DistanceToTurnSection: 0x134,      // float — metres to the next turn stretch (turn-timing countdown)
  DistanceToTurnSectionEnd: 0x138,   // float
  DistanceTraveledSinceLastTurnSection: 0x13C, // float
  DistanceToChangeAtTemple: 0x128,   // float — metres to next altitude change (temple)
  DistanceToChangeAtOther: 0x12C,    // float — metres to next altitude change (other)
  DistanceTraveledSinceLastLevelChange: 0x130, // float — metres since last altitude change
  DistanceRemainder: 0x120,          // float — sub-tile fractional distance into the current tile
  StumbleCount: 0x168,               // int   — stumbles this run
  // --- COINS (PathElement): coins are collected by TILTING to their lane -> a tilt means COINS or a
  // narrow, so we need coin placement to disambiguate + to model coin-collecting in the sim. ---
  PE_HasCoins: 0x3D,                 // bool  — this cell has a coin run
  PE_CoinLocStart: 0x44,             // int   — CoinRunLocation at cell entry (0 None,1 Center,2 Left,3 Right)
  PE_CoinLocEnd: 0x48,               // int   — CoinRunLocation at cell exit (start!=end => diagonal run)
  PE_CoinInAir: 0x4C,                // bool  — the coin run is in the air (needs a jump)
  PE_CoinRunCount: 0x88,             // int   — coins in this cell's run
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
  // LATERAL-ACCEL: the game's own analog tilt read. TouchInput (the live GameInput on device)
  // overrides float GetAccelerometerForceX(); HandleAccelerometerForce() consumes it every frame and
  // drives PlayerXOffset with the game's OWN glide/recenter dynamics. Replacing this = commanding the
  // tilt the way a human hand does (px writes lost the write-war with the char-update; see 15a note).
  GetAccelerometerForceX_Touch: 0x12AD500, // float TouchInput.GetAccelerometerForceX() override
  // NARROW GROUND TRUTH (2026-07-02, user prompt "isn't there a variable?"): each PathElement's
  // walkable surface is a set of Region2D POLYGONS (List<Vector2> Points) — the narrow strip's side
  // is GEOMETRY, not an enum (why all name-based side maps kept failing). This is the game's own
  // "is this point on ground" test (drives the narrow fall, gated by AllowFallOnNarrows/"FON").
  IsPointInsideAnyRegion2D: 0x12E8A90, // bool PathElement.IsPointInsideAnyRegion2D(Vector3 point)
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
const HOOK_VERSION = 'lateral-accel-16e'; // 16e = 16d + track forcing (PathElement statics via IL2CPP API)
const DIAG_TURN = true;             // emit the lane / cell-progress diagnostic in state (calibration)
let cachedDbg = null;               // raw junction structure for diagnosis
let cachedTurnDiag = null;          // lane offset + cell progress, for calibration
// TURN BUFFERING (the big on-device fix): a native TurnLeft/Right fired at the START of the junction
// cell (cell_progress ~0.1, where the policy naturally reacts to is_turn) swings the player ~12 units
// off-center -> it falls off the exit path (measured: 9/12 on-device deaths). Firing the SAME turn at
// the cell CENTER (progress ~0.45) lands it centered every time (postLat <=2). So we don't execute the
// turn byte immediately: we LATCH it and fire when the player reaches the cell center.
const TURN_BUFFER = false;      // NN decides turns (no hook buffering). lane/cell_progress obs stay on.
// FINER LATERAL CONTROL (2026-07-01, user-approved 3-level tilt): the native StrafeLeft/Right SET a
// persistent PlayerXOffset (±7) that the policy can NEVER un-set to 0 -> stuck off-center -> fatal turns.
// Instead the hook OWNS the lateral: the policy commands a TARGET (byte0 -> -reach hold, byte1 -> +reach
// hold, byte6/NoOp -> 0 = LEVEL/recenter) and we GLIDE PlayerXOffset toward it each Update frame. This is
// a CONTROL mapping (policy decides L/C/R), not a hook rule. Verified: writing PlayerXOffset moves the
// runner + NoOp glides it back to center. See [[lateral-control-recenter-mechanic]].
// 15a: LATERAL_TILT (direct px writes) FAILED the device probe (strafe_probe_tilt2, 2026-07-02): the
// char-update owns PlayerXOffset OUTSIDE GameController.Update, so our onLeave writes never land (px
// pinned ~0 during a 10-frame hold) and the two writers fight (px flicker -4.0<->-4.6). Dead flag.
const LATERAL_TILT = false;     // [DEAD 15a] px write-war lost — use LATERAL_ACCEL instead
const LAT_REACH = 7.0;          // |target| one held tilt reaches (clamped to the ±7 edge)
const LAT_GLIDE = 0.30;         // PlayerXOffset units moved toward the target per Update frame (~60 Hz)
let latTarget = 0.0;            // commanded lateral target (set by action bytes 0/1/6)
// LATERAL-ACCEL (15a): replace TouchInput.GetAccelerometerForceX() to return a COMMANDED tilt force.
// The game's HandleAccelerometerForce then applies its own sensitivity/glide/recenter — identical
// dynamics to a human tilting the phone. byte0 = hold LEFT, byte1 = hold RIGHT, byte6/NoOp = LEVEL
// (game recenters px itself). Still a pure control mapping: the policy decides L/R/level every step.
const LATERAL_ACCEL = true;     // the deploy mode (flip false only for legacy-policy capture sessions)
const TILT_FORCE = 0.60;        // |accel force| a held tilt commands (probe: tune so reach ~ lane ±7)
const TILT_LEFT_SIGN = -1.0;    // sign of force X that moves the runner LEFT (verify via probe; flip if wrong)
let tiltCmd = 0.0;              // commanded accelerometer force X (0 = level phone)
const TURN_FIRE_PROG = 0.40;        // fire a buffered turn once cell_progress crosses this (cell-center)
const TURN_PENDING_MAX_AGE = 20;    // drop a buffered turn if it can't fire within N Update frames
let pendingTurn = 0;                // 0 none, 4 TurnLeft, 5 TurnRight (buffered, fires at cell center)
let pendingTurnAge = 0;
let pendingSawTurnCell = false;     // became true once we were on the turn cell while a turn is pending
let cachedLane = 0.0;               // signed lateral offset from path center (+ = right of travel)
let cachedCellProg = 0.0;           // longitudinal position within the nearest cell along travel, 0..1
let cachedRawIsTurn = false;        // nearest cell is a turn piece (IsTurn()), pre-latch
// PROBE: the game's REAL turn-window test (IsPointOverTurnArea2D). It's a NATIVE call so it must run on
// the main thread (Update) and be cached for the broadcast thread. radius = player tolerance; tune from
// the probe (does over_turn_area align with the turnable window / cell_progress?).
let cachedOverTurnArea = false;
const TURN_AREA_RADIUS = 2.0;
// next-cell coin lookahead (the forward neighbor's coin run), cached in the Update hook.
let cachedNextHasCoins = false, cachedNextCoinStart = 0, cachedNextCoinEnd = 0;
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
// REGION-SAMPLED narrow band (perception, replaces the guessed NARROW_SIDE map): when the forward
// cell is a narrow (17-30), test lane offsets -7..+7 against ITS Region2D polygons via the game's
// IsPointInsideAnyRegion2D -> the true safe lateral band [lo, hi] in lane units (+ = right of travel).
let cachedNarrowBandLo = null, cachedNarrowBandHi = null;
let cachedOnRegion = true;          // player's own point vs the nearest element's regions (probe)
// TURN-BUFFER PROBE (GameController.GameInput @0x78 = the HandleDelayedTurning state): a swipe sets
// ShouldTurnLeft/Right + TimeSinceShouldTurn=0; the turn executes when the runner reaches the corner,
// but the buffer EXPIRES if TimeSinceShouldTurn grows too large -> run past -> water. Streamed so the
// turn-execution probe can measure the real window/timeout (not model it blind).
let cachedShouldTurnL = false, cachedShouldTurnR = false, cachedJustTurned = false;
let cachedTimeSinceShouldTurn = 0.0, cachedTimeSinceLastTurn = 0.0;
// IS_TURN LATCH (15d, perception): the streamed is_turn used to be a ~1-frame flash at 10 Hz (the
// nearest-element handoff hides the rest of the window) while the SIM presents is_turn across its
// whole 0.45 window — a sim-to-real obs mismatch. Latch the flash: keep is_turn (+ its can_turn
// sides) true from the first flash until progress 0.45 along the FLASHED cell's rect (the measured
// executable window: probe_turn_window 2026-07-02 — turns execute 0.14-0.56, reliable core <=0.45),
// or until the turn is taken / facing changes. Device obs now matches sim semantics; no obs resize.
// RE-ENABLED 2026-07-02 (extended to 0.60): probe_turn_exec MEASURED the turn executes at the swipe
// cprog and lands at offset ~(cprog-0.44)*45 -> the policy must swipe at cell CENTER (~0.44) to land
// centered. Raw is_turn is a 1-frame flash at ~0.23 -> the policy can only swipe there -> lands ~8 off
// -> water. The latch holds is_turn True across the executable window [flash, 0.60] so the policy
// (via cell_progress) can AIM the center. Double-turn is prevented by turn_linger-3 TRAINING, not by
// hiding the signal. The sim presents is_turn identically ([turn_exec_early, turn_exec_late]).
const TURN_LATCH = true;
const TURN_LATCH_FRAC = 0.60;
let latchRect = null;               // Rect of the flashed junction cell {rx, rz, rw, rh}
let latchFacing = 0;                // heading when the flash was latched (facing change = turned)
let latchCanL = false, latchCanR = false;
let cachedLatchOpen = false;        // latched window currently open (computed in the Update hook)
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
// ---------------------------------------------------------------------------
// TRACK FORCING (2026-07-02): control the DEVICE track generation so we can test one mechanic in
// isolation (many turns/run) AND do a cheap DIRECT sim-vs-real comparison (same track config + policy
// on both, diff outcomes) — the eval-loop inefficiency the user flagged. The game reads
// PathElement.s* STATIC flags (sAllowTurns/Holes/Narrows/... @0x7.., sMinSpaceBetweenTurns @0x2C);
// we set them via the IL2CPP static-field API (il2cpp_field_static_set_value). Re-applied each Update
// (the game may reset them at level start). NN still decides actions; this only shapes the terrain.
const TRACK_MODE = 'narrows'; // 'off' | 'turns' | 'clean' | 'narrows' (narrow-strip-dense, isolate the fall)
// NOTE 2026-07-02: 'turns' works (statics set) but generates CURVE pieces (pt3/4, path bends) that
// is_turn doesn't flag like T-junctions -> an is_turn-based policy runs off them. Refine to force
// T-junctions (pt7-10) specifically before using 'turns' as a clean turn-reliability test.
// per-mode PathElement static overrides. bools -> u8, ints -> i32. Not listed = left as the game set it.
const TRACK_CONFIGS = {
  turns: { sAllowTurns: 1, sAllowPathHoles: 0, sAllowPathNarrows: 0, sAllowFlameTowers: 0,
           sAllowTreeStumble: 0, sAllowTreeJump: 0, sAllowTreeSlide: 0, sAllowSidePaths: 1,
           sMinSpaceBetweenTurns: 10, sMinSpaceBetweenObstacles: 999 },
  // NOTE: sAllowSidePaths MUST be 1 — turns EXIT onto side paths; with it 0 every turn is a
  // dead-end and even a perfect center-swipe dies (verified 2026-07-02).
  clean: { sAllowTurns: 0, sAllowPathHoles: 0, sAllowPathNarrows: 0, sAllowFlameTowers: 0,
           sAllowTreeStumble: 0, sAllowTreeJump: 0, sAllowTreeSlide: 0, sAllowSidePaths: 0 },
  // narrows-only: force NarrowLong strips, no turns/holes/trees -> isolate the narrow-fall mechanic
  // for the replay_diff loop. sMinSpaceBetweenObstacles low so narrows come often.
  narrows: { sAllowTurns: 0, sAllowPathHoles: 0, sAllowPathNarrows: 1, sAllowFlameTowers: 0,
             sAllowTreeStumble: 0, sAllowTreeJump: 0, sAllowTreeSlide: 0, sAllowSidePaths: 0,
             sMinSpaceBetweenObstacles: 4 },
};
const TRACK_INT_FIELDS = { sMinSpaceBetweenTurns: 1, sMinSpaceBetweenObstacles: 1, sMaxBackToBackObstacles: 1 };
let il2 = null;             // resolved IL2CPP API (lazy)
let peClass = null;         // PathElement Il2CppClass*
let peFields = {};          // name -> FieldInfo*
let trackScratch = null;    // scratch buffer for set_value
let trackApplied = false;   // logged once
let cachedTrackStatics = null;   // current static values (streamed for verification)

function resolveIl2Api() {
  if (il2) return il2;
  try {
    const lib = Process.findModuleByName('libil2cpp.so');
    if (!lib) { dbg('il2_no_lib'); return null; }
    const E = (n) => {
      const p = lib.findExportByName ? lib.findExportByName(n) : lib.getExportByName(n);
      if (!p || p.isNull()) throw new Error('missing export ' + n);
      return p;
    };
    il2 = {
      domain_get: new NativeFunction(E('il2cpp_domain_get'), 'pointer', []),
      assembly_open: new NativeFunction(E('il2cpp_domain_assembly_open'), 'pointer', ['pointer', 'pointer']),
      asm_image: new NativeFunction(E('il2cpp_assembly_get_image'), 'pointer', ['pointer']),
      class_from_name: new NativeFunction(E('il2cpp_class_from_name'), 'pointer', ['pointer', 'pointer', 'pointer']),
      field_from_name: new NativeFunction(E('il2cpp_class_get_field_from_name'), 'pointer', ['pointer', 'pointer']),
      field_set: new NativeFunction(E('il2cpp_field_static_set_value'), 'void', ['pointer', 'pointer']),
      field_get: new NativeFunction(E('il2cpp_field_static_get_value'), 'void', ['pointer', 'pointer']),
    };
  } catch (e) { dbg('il2_resolve_err', String(e)); il2 = null; }
  return il2;
}

function resolvePathElementStatics() {
  if (peClass) return true;
  const a = resolveIl2Api();
  if (!a) return false;
  try {
    const dom = a.domain_get();
    const asm = a.assembly_open(dom, Memory.allocUtf8String('Assembly-CSharp'));
    if (asm.isNull()) { dbg('track_no_asm'); return false; }
    const img = a.asm_image(asm);
    peClass = a.class_from_name(img, Memory.allocUtf8String(''), Memory.allocUtf8String('PathElement'));
    if (peClass.isNull()) { dbg('track_no_class'); peClass = null; return false; }
    const names = ['sAllowTurns', 'sAllowPathHoles', 'sAllowPathNarrows', 'sAllowFlameTowers',
                   'sAllowTreeStumble', 'sAllowTreeJump', 'sAllowTreeSlide', 'sAllowSidePaths',
                   'sMinSpaceBetweenTurns', 'sMinSpaceBetweenObstacles', 'sMaxBackToBackObstacles'];
    names.forEach((n) => { peFields[n] = a.field_from_name(peClass, Memory.allocUtf8String(n)); });
    trackScratch = Memory.alloc(8);
    dbg('track_resolved', Object.keys(peFields).filter((k) => !peFields[k].isNull()).length);
    return true;
  } catch (e) { dbg('track_resolve_err', String(e)); peClass = null; return false; }
}

function forceTrack() {
  if (TRACK_MODE === 'off') return;
  if (!resolvePathElementStatics()) return;
  const cfg = TRACK_CONFIGS[TRACK_MODE];
  if (!cfg) return;
  const a = il2;
  try {
    for (const name in cfg) {
      const f = peFields[name];
      if (!f || f.isNull()) continue;
      if (TRACK_INT_FIELDS[name]) trackScratch.writeS32(cfg[name] | 0);
      else trackScratch.writeU8(cfg[name] ? 1 : 0);
      a.field_set(f, trackScratch);
    }
    if (!trackApplied) { trackApplied = true; log('track forced: ' + TRACK_MODE); }
  } catch (e) { dbg('track_force_err', String(e)); }
}

function readTrackStatics() {   // for the state stream — verify the forced values landed
  if (!peClass || !il2) return null;
  const out = {};
  try {
    for (const name in peFields) {
      const f = peFields[name];
      if (!f || f.isNull()) continue;
      if (TRACK_INT_FIELDS[name]) { il2.field_get(f, trackScratch); out[name] = trackScratch.readS32(); }
      else { il2.field_get(f, trackScratch); out[name] = trackScratch.readU8(); }
    }
  } catch (_) {}
  return out;
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
  fn.IsPointInsideAnyRegion2D = new NativeFunction(base.add(RVA.IsPointInsideAnyRegion2D),
    'bool', ['pointer', ['float', 'float', 'float'], 'pointer']);  // (PathElement*, Vector3 point, mi) -> bool ("2D" = projects x,z)
  if (LATERAL_ACCEL) {
    // Command the game's own tilt input: TouchInput.GetAccelerometerForceX() -> tiltCmd. Full replace
    // (no original call): a flat phone reads ~0 anyway, and tiltCmd IS the commanded phone angle.
    // Float return rides s0, so attach+retval.replace can't do this — NativeCallback replacement can.
    Interceptor.replace(base.add(RVA.GetAccelerometerForceX_Touch),
      new NativeCallback(function () { return tiltCmd; }, 'float', ['pointer', 'pointer']));
    log('lateral-accel: TouchInput.GetAccelerometerForceX replaced (tiltCmd-driven)');
  }
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
      resolvePathElementStatics();   // resolve the PathElement class/fields once (for read + force)
      forceTrack();   // shape the device track (turns-only / clean / off) via PathElement statics
      if ((updateTicks % 30) === 0) cachedTrackStatics = readTrackStatics();  // verify, ~2 Hz
      // turn-buffer state (GameInput @0x78) — the delayed-turn queue the probe measures
      try {
        const gi = gcInstance.add(OFF.GameInput).readPointer();
        if (!gi.isNull()) {
          cachedShouldTurnL = readBool(gi, OFF.GI_ShouldTurnLeft);
          cachedShouldTurnR = readBool(gi, OFF.GI_ShouldTurnRight);
          cachedJustTurned = readBool(gi, OFF.GI_JustTurned);
          cachedTimeSinceShouldTurn = readF32(gi, OFF.GI_TimeSinceShouldTurn);
          cachedTimeSinceLastTurn = readF32(gi, OFF.GI_TimeSinceLastTurn);
        }
      } catch (_) {}
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
              // PROBE: the game's REAL "is the player physically over this cell's turn area" test
              // (native call -> main thread only). Compare against is_turn/cell_progress in the capture.
              try { cachedOverTurnArea = !!fn.IsPointOverTurnArea2D(pe, [px, pz], TURN_AREA_RADIUS, NULLPTR); }
              catch (_) { cachedOverTurnArea = false; }
              // probe: the game's own on-ground test for the player's current point (drives narrow falls)
              try { cachedOnRegion = !!fn.IsPointInsideAnyRegion2D(pe, [px, 0.0, pz], NULLPTR); }
              catch (_) { cachedOnRegion = true; }
              // IS_TURN LATCH (15d): capture the flash + hold the window open along the FLASHED
              // cell's rect until TURN_LATCH_FRAC (works across the nearest-element handoff).
              if (TURN_LATCH && rawIsTurn && !turnedHere) {
                latchRect = { rx: rx, rz: rz, rw: rw, rh: rh };
                latchFacing = facing;
                latchCanL = neighborExists(pe, LEFT_NEIGHBOR[facing]);
                latchCanR = neighborExists(pe, RIGHT_NEIGHBOR[facing]);
              }
              if (latchRect !== null && (!TURN_LATCH || facing !== latchFacing || turnedHere)) latchRect = null;
              cachedLatchOpen = false;
              if (TURN_LATCH && latchRect !== null) {
                let lprog;
                if (facing === 1) lprog = (pz - latchRect.rz) / latchRect.rh;
                else if (facing === 2) lprog = (px - latchRect.rx) / latchRect.rw;
                else if (facing === 3) lprog = 1 - (pz - latchRect.rz) / latchRect.rh;
                else lprog = 1 - (px - latchRect.rx) / latchRect.rw;
                if (lprog <= TURN_LATCH_FRAC) cachedLatchOpen = true;
                else latchRect = null;      // window closed (past the executable core)
              }
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
            // only until the one turn is taken. 15d: OR the latched window, so is_turn presents the
            // full executable window (matching the sim) instead of the 1-frame handoff flash.
            const rawOpen = rawIsTurn && !turnedHere;
            cachedIsTurn = rawOpen || cachedLatchOpen;
            cachedCanLeft = rawOpen ? neighborExists(pe, LEFT_NEIGHBOR[facing])
                                    : (cachedLatchOpen ? latchCanL : false);
            cachedCanRight = rawOpen ? neighborExists(pe, RIGHT_NEIGHBOR[facing])
                                     : (cachedLatchOpen ? latchCanR : false);
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
              cachedNextHasCoins = readBool(nxt, OFF.PE_HasCoins);
              cachedNextCoinStart = readI32(nxt, OFF.PE_CoinLocStart);
              cachedNextCoinEnd = readI32(nxt, OFF.PE_CoinLocEnd);
              // Region-sample the upcoming narrow's safe band from the game's own polygons.
              cachedNarrowBandLo = cachedNarrowBandHi = null;
              if (cachedNextPathType >= 17 && cachedNextPathType <= 30) {
                try {
                  const nrx = readF32(nxt, OFF_PE_Rectangle), nrz = readF32(nxt, OFF_PE_Rectangle + 4);
                  const nrw = readF32(nxt, OFF_PE_Rectangle + 8), nrh = readF32(nxt, OFF_PE_Rectangle + 12);
                  const ncx = nrx + nrw / 2, ncz = nrz + nrh / 2;
                  let lo = null, hi = null;
                  for (let k = -7; k <= 7; k++) {
                    let sx = ncx, sz = ncz;              // lane +k = right of travel (mirror of cachedLane)
                    if (facing === 1) sx = ncx + k;
                    else if (facing === 2) sz = ncz - k;
                    else if (facing === 3) sx = ncx - k;
                    else if (facing === 4) sz = ncz + k;
                    if (fn.IsPointInsideAnyRegion2D(nxt, [sx, 0.0, sz], NULLPTR)) {
                      if (lo === null) lo = k;
                      hi = k;
                    }
                  }
                  cachedNarrowBandLo = lo; cachedNarrowBandHi = hi;
                } catch (_) { cachedNarrowBandLo = cachedNarrowBandHi = null; }
              }
            } else {
              cachedNextIsTurn = cachedNextCanLeft = cachedNextCanRight = false;
              cachedNextMustJump = cachedNextMustSlide = false;
              cachedNextIsLevelChange = false;
              cachedNextHasCoins = false; cachedNextCoinStart = 0; cachedNextCoinEnd = 0;
              cachedNextPathType = -1;
              cachedNarrowBandLo = cachedNarrowBandHi = null;
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
    },
    onLeave(retval) {
      // FINER LATERAL: apply the tilt glide AFTER the game's own character update (which runs during
      // Update and would overwrite an onEnter write, so an onEnter write did nothing). Glide PlayerXOffset
      // toward latTarget; latTarget=0 (NoOp) glides back to px 0 = lane 0 = CENTER.
      if (!LATERAL_TILT || gcInstance === null || gcInstance.isNull()) return;
      try {
        const cp = gcInstance.add(OFF.GamePlayer).readPointer();
        if (cp.isNull()) return;
        const pxo = cp.add(OFF_PlayerXOffset);
        let cur = pxo.readFloat();
        const d = latTarget - cur;
        if (Math.abs(d) <= LAT_GLIDE) cur = latTarget;
        else cur += (d > 0 ? LAT_GLIDE : -LAT_GLIDE);
        if (cur > LAT_REACH) cur = LAT_REACH; else if (cur < -LAT_REACH) cur = -LAT_REACH;
        pxo.writeFloat(cur);
      } catch (_) {}
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
    // Lateral: 3-level tilt (LATERAL_TILT) — set the target PlayerXOffset (px); the Update onLeave glides
    // px there. Measured: lane = -px, so LEFT (lane<0) needs px>0. byte0=LEFT (px +reach), byte1=RIGHT
    // (px -reach). NoOp (byte6) sets px target 0 = LEVEL = recenter to lane 0.
    case 0:
      if (LATERAL_ACCEL) tiltCmd = TILT_LEFT_SIGN * TILT_FORCE;
      else if (LATERAL_TILT) latTarget = +LAT_REACH;
      else if (!cp.isNull()) fn.StrafeLeft(cp, NULLPTR);
      break;
    case 1:
      if (LATERAL_ACCEL) tiltCmd = -TILT_LEFT_SIGN * TILT_FORCE;
      else if (LATERAL_TILT) latTarget = -LAT_REACH;
      else if (!cp.isNull()) fn.StrafeRight(cp, NULLPTR);
      break;
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
    case 6:
      if (LATERAL_ACCEL) tiltCmd = 0.0;         /* NoOp -> LEVEL the phone = game recenters px */
      else if (LATERAL_TILT) latTarget = 0.0;
      break;
    case 7: // RestartGame() per spec (resets game LOGIC only; the run stays in game-over limbo)
      if (!gc.isNull()) { fn.RestartGame(gc, NULLPTR); dbg('restartgame'); }
      break;
    case 8: // RunAgainClicked() — the "Rejouer" flow that STARTS a fresh run from the GAME-OVER screen.
      tiltCmd = 0.0; latTarget = 0.0;   // never carry a held tilt into a new run
      if (!cp.isNull()) {
        const egui = cp.add(OFF.GameOverInterface).readPointer();
        if (!egui.isNull()) { fn.RunAgainClicked(egui, NULLPTR); dbg('runagain'); }
        else { dbg('runagain_no_egui'); } // not at game-over (e.g. at the main menu) -> use byte 9
      }
      break;
    case 9: // OnPlayClicked() — press "JOUER" to start a run from the MAIN MENU.
      tiltCmd = 0.0; latTarget = 0.0;   // never carry a held tilt into a new run
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
    // --- Phase-0 PROBE fields (read-only; confirm semantics vs the failure modes) ---
    player_x_offset: 0.0, is_over_ground: false, ground_height: 0.0, ground_change_higher: false,
    stumble_kill_timer: 0.0, did_stumble: false, death_type: 0,
    time_since_strafe: 0.0, strafe_duration: 0.0, max_velocity: 0.0, vel_x: 0.0, vel_z: 0.0,
    dist_to_turn: 0.0, dist_to_turn_end: 0.0, dist_since_turn: 0.0,
    dist_to_change_temple: 0.0, dist_to_change_other: 0.0, dist_since_level_change: 0.0,
    dist_remainder: 0.0, stumble_count: 0, over_turn_area: false,
    // coins (disambiguate a tilt = coins vs narrow; model coin placement in the sim)
    has_coins: false, coin_loc_start: 0, coin_loc_end: 0, coin_in_air: false, coin_count: 0,
    next_has_coins: false, next_coin_loc_start: 0, next_coin_loc_end: 0,
  };
  if (gcInstance === null || gcInstance.isNull()) return s;

  s.is_game_over = readBool(gcInstance, OFF.IsGameOver);
  // DistanceTraveled is the real run distance (no challenge multiplier) -> the reward signal.
  s.distance = readF32(gcInstance, OFF.DistanceTraveled);
  // --- PROBE: GameController metric distance signals (pure reads) ---
  s.dist_to_turn = readF32(gcInstance, OFF.DistanceToTurnSection);          // metres to next turn stretch
  s.dist_to_turn_end = readF32(gcInstance, OFF.DistanceToTurnSectionEnd);
  s.dist_since_turn = readF32(gcInstance, OFF.DistanceTraveledSinceLastTurnSection);
  s.dist_to_change_temple = readF32(gcInstance, OFF.DistanceToChangeAtTemple); // metres to next altitude change
  s.dist_to_change_other = readF32(gcInstance, OFF.DistanceToChangeAtOther);
  s.dist_since_level_change = readF32(gcInstance, OFF.DistanceTraveledSinceLastLevelChange);
  s.dist_remainder = readF32(gcInstance, OFF.DistanceRemainder);            // sub-tile fractional distance
  s.stumble_count = readI32(gcInstance, OFF.StumbleCount);

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
    // --- PROBE: CharacterPlayer fields (pure reads) ---
    s.player_x_offset = readF32(cp, OFF.PlayerXOffset);   // the game's own lateral offset (candidate real lane)
    s.is_over_ground = readBool(cp, OFF.IsOverGround);
    s.ground_height = readF32(cp, OFF.GroundHeight);
    s.ground_change_higher = readBool(cp, OFF.IsGroundHeightChangeHigher);
    s.stumble_kill_timer = readF32(cp, OFF.StumbleKillTimer);
    s.did_stumble = readBool(cp, OFF.DidStumble);
    s.death_type = readI32(cp, OFF.DeathType);            // CharacterPlayer.DeathTypes (set on death)
    s.time_since_strafe = readF32(cp, OFF.TimeSinceStrafeStart);
    s.strafe_duration = readF32(cp, OFF.StrafeDuration);
    s.max_velocity = readF32(cp, OFF.MaxRunVelocity);
    s.vel_x = readF32(cp, OFF.Velocity);                 // MovingObject.Velocity.x @0x34
    s.vel_z = readF32(cp, OFF.Velocity + 8);             // MovingObject.Velocity.z @0x3C (lateral drift)
  }

  const pe = gcInstance.add(OFF.NearestElement).readPointer();
  if (!pe.isNull()) {
    s.path_type = readI32(pe, OFF.PE_Type);
    s.path_level = readI32(pe, OFF.PE_Level);
    const obs = readI32(pe, OFF.PE_Obstacles);
    s.must_jump = MUST_JUMP_TYPES.indexOf(obs) !== -1;
    s.must_slide = MUST_SLIDE_TYPES.indexOf(obs) !== -1;
    s.obstacle = obs;   // raw ObstacleType enum (diagnose crevasse / tree-root)
    // COINS on the nearest cell (CoinRunLocation: 0 None, 1 Center, 2 Left, 3 Right)
    s.has_coins = readBool(pe, OFF.PE_HasCoins);
    s.coin_loc_start = readI32(pe, OFF.PE_CoinLocStart);
    s.coin_loc_end = readI32(pe, OFF.PE_CoinLocEnd);
    s.coin_in_air = readBool(pe, OFF.PE_CoinInAir);
    s.coin_count = readI32(pe, OFF.PE_CoinRunCount);
  }
  s.next_has_coins = cachedNextHasCoins;
  s.next_coin_loc_start = cachedNextCoinStart;
  s.next_coin_loc_end = cachedNextCoinEnd;
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
  // REGION-SAMPLED band (game's own Region2D polygons, lane units + = right of travel): the true
  // safe lateral extent of the upcoming narrow. Candidate replacement for narrow_target_lane's
  // guessed map — validate against sweep outcomes, then feed the obs.
  s.narrow_band_lo = cachedNarrowBandLo;
  s.narrow_band_hi = cachedNarrowBandHi;
  s.on_region = cachedOnRegion;      // probe: player point vs nearest element's regions
  // turn-buffer (HandleDelayedTurning) probe fields
  s.should_turn_left = cachedShouldTurnL;
  s.should_turn_right = cachedShouldTurnR;
  s.just_turned = cachedJustTurned;
  s.time_since_should_turn = cachedTimeSinceShouldTurn;
  s.time_since_last_turn = cachedTimeSinceLastTurn;
  s.track_mode = TRACK_MODE;
  s.track_statics = cachedTrackStatics;   // current PathElement.s* (verify a forced track landed)
  s.lane = cachedLane;               // signed lateral offset from path center (+ = right of travel)
  s.cell_progress = cachedCellProg;  // longitudinal position within the nearest cell, 0..1
  s.over_turn_area = cachedOverTurnArea; // PROBE: game's real turn-window test (IsPointOverTurnArea2D)
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
