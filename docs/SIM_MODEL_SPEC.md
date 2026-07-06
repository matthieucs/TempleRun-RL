# Temple Run — Headless Sim Model Spec (v1.39.0, IL2CPP)

Dump-verified model for building `TempleRunSim-v0`. Tags: **[KNOWN]** = verbatim from
`il2cpp_work/output/dump.cs` (class/field/offset/enum/RVA). **[INFERRED]** = deduced from
names/structure/standard endless-runner design; needs short automated calibration on the phone.
All offsets/RVAs are for this exact `libil2cpp.so`. The sim must emit the **same 22-field raw-state
dict** the Frida hook (`frida/temple_run_hook.js`) produces, so `obs.py`/`constants.py` are reused verbatim.

## 0. State contract (22 fields) — sources [KNOWN]
Three live objects: **CharacterPlayer** (`GameController.GamePlayer`@0x70), **GameController**,
nearest **PathElement** (`GameController.NearestElement`@0x88).

| field | source |
|---|---|
| is_dead/jumping/sliding/on_ground/stumbling/strafing | CharacterPlayer bools @0xD8/0xF8/0x100/0x130/0x10C/0x125 |
| has_invincibility/angel_wings/vacuum/boost, is_mega_boost | @0x154/0x161/0x16C/0x174/0x175 |
| score | CharacterPlayer.Score @0x138 (int) |
| velocity | `CharacterPlayer.GetRunVelocity()` RVA 0x12982D8 |
| facing | CommonPlayer.Facing (PathDirection) @0x78 |
| is_game_over | GameController.IsGameOver @0xDE |
| path_type / path_level | NearestElement.Type@0x30 / .Level@0x34 |
| must_jump | NearestElement.Obstacles@0x38 ∈ {1 Hole, 4 TreeJump} |
| must_slide | NearestElement.Obstacles@0x38 ∈ {5 TreeSlide, 6 FlameTower} |
| can_turn_left / can_turn_right | `PathElement.CanTurnLeft/Right(facing)` RVA 0x12E2E88/0x12E2F44 |
| is_turn | `PathElement.IsTurn()` RVA 0x12E28B0 |

## 1. Track = doubly-linked graph of PathElement [KNOWN]
Not a tilemap. `PathElement` (dump.cs:283287): grid 60×60. Neighbors `PathNorth`@0x50 `PathEast`@0x58
`PathWest`@0x60 `PathSouth`@0x68 (PathElement*). Counters: `TileIndex`@0x70, `SpacesSinceLastBonusItem`@0x74,
`SpacesSinceLastTurn`@0x78, `SpacesSinceLastObstacle`@0x7C, `BackToBackObstacleCount`@0x80,
`SpacesSinceLastCoinRun`@0x84, `CoinRunCount`@0x88.
Generator (recursive, extends ahead): `AddRandomPath(depth,origin)` RVA 0x12E7584;
`GetRandomPathElementForSide` 0x12E3E48; `UpdatePathDescription` 0x12E664C; `Prune` 0x12E7994.
Nav: `GetPathForward/Left/Right(origin)` 0x12E34C4/3564/3660; `GetOpposite` 0x12E375C.

**Enums [KNOWN]:**
- `PathType` 0..37 (None0, StraightVert1, StraightHorz2, Turn A/B/C/D 3-6, T-junctions 7-10,
  hole-geometry 11-16, narrow variants 17-30, FlameTowerHigh 31-32, StepDown 33-36, Temple 37). → obs path_type /37.
- `PathLevel` None0/Plank1/Temple2/Cliff3/Max4 (default piece level=2). → obs path_level /4.
- `PathDirection` None0/North1/East2/South3/West4. → obs facing.

**Turn = piece whose PathType is a turn/T-type.** `CanTurnLeft/Right(origin)` check the left/right
neighbor relative to heading exists+traversable. Sim implements from neighbor graph + facing.

**Generation knobs = PathElement statics [KNOWN names, INFERRED values]** (dump.cs:283320-283360,
restorable block `SavedPathElementStatics`): `sAllow*` bools; `sMinSpacesBetweenBonusItems`@0x18,
`sMinSpaceBetweenTurns`@0x2C, `sMaxBackToBackObstacles`@0x30, `sMinSpaceBetweenObstacles`@0x34,
`sDoubleObstaclePercent`@0x38, `sMinSpacesBetweenCoinRuns`@0x3C, `sMaxCoinsPerRun`@0x40,
`sProbability{CoinBonus@0x1C,Invincibility@0x20,Vacuum@0x24,Boost@0x28}`, `sPathLevel`@0x44, `sIsFastTurnSection`@0x48.
Distance-gated (GameController): `ChangeLevel` 0x1272C80 when `DistanceTraveledSinceLastLevelChange`@0x130 >
`DistanceToChangeAtTemple`@0x128/`AtOther`@0x12C; turn sections via `DistanceToTurnSection(/End)`@0x134/0x138.

## 2. Obstacles — `PathElement.ObstacleType` enum [KNOWN]
0 None · 1 Hole→**jump** · 2 Narrow→strafe to center · 3 TreeStumble→stumble(slow, not death) ·
4 TreeJump→**jump** · 5 TreeSlide→**slide** · 6 FlameTower→**slide** · 7 LevelChange(marker).
One per element: `Obstacles`@0x38 (+`HasObstacles`@0x3C). `AddObstacle` RVA 0x12E2934. Authoritative
must_jump/must_slide signal = the `Obstacles` enum (that's what the hook reads).

## 3. Player physics — CharacterPlayer
**Velocity [KNOWN]:** `GetRunVelocity()` 0x12982D8. Per-instance `StartRunVelocity`@0xE4,`MaxRunVelocity`@0xEC,
`DeathRunVelocity`@0xE8. Piecewise accel curve by distance: `GetAccelerationCurveThreshold(i)` 0x1272E3C /
`GetAccelerationCurveForce(i)` 0x1272EE4 (PlayerConfigData AccCurveThresholds/Forces). `HandlePacing` 0x126E768.
Config defaults [KNOWN]: SRV=100, MRV=210, StumbleSlowDown=0.85, ResurrectSlowDown=0.85, ResurrectBoostDist=150
(matches obs.py VEL_MIN 100/VEL_MAX 220).
**Actions = direct method calls [KNOWN]:** Jump(delay) 0x1297B80 (→IsJumping@0xF8); Slide() 0x1298768
(→IsSliding@0x100, TimeSinceSlideStart@0x104/SlideDuration@0x108); StrafeLeft/Right 0x1298874/0x1298908
(→IsStrafing@0x125, dur@0x12C; 3 lanes L/C/R, shift one); TurnLeft/Right 0x129899C/0x1298A14 (→Facing@0x78,
**valid only at junction — forced turn mid-path kills**); Stumble(delay) 0x12979D8 (→IsStumbling@0x10C,
StumbleKillTimer@0x11C); Kill(DeathType) 0x1297EB0 (→IsDead@0xD8).
**Ground:** IsOnGround@0x130, IsFalling@0xF0, GroundHeight@0x134.
**[INFERRED-calibrate]:** SlideDuration, StrafeDuration, StumbleDuration, jump airtime, action-accept windows.

## 4. Death — `DeathTypes` enum [KNOWN], triggers [INFERRED]
0 Unknown,1 Fall,2 Tree,3 Slide,4 Ledge,5 Burnt,6 Eaten,7 Tangle,8 SceneryTree,9 SceneryRock.
Kill paths in GameController.Update: `HandleCollisionWithObstacles` 0x12713E4 (obstacle hit un-actioned);
`CheckHittingScenery` 0x1270C3C; `MakeSureDidntRunPastTurn(old,new)` 0x127089C (**missed turn → Fall**);
unrecovered Stumble. Sim model: at player's cell, if Obstacles needs jump/slide and the matching flag isn't
active when passed → Kill; at a junction, if no valid turn issued before passing → Kill(Fall).

## 5. Scoring [KNOWN fields, INFERRED constants]
Score@0x138 via `AddScore` 0x1298A8C; ScoreMultiplier@0x13C; `AddCoins` 0x1298AA8. Distance via
`ComputeDistanceTraveled` 0x126E280 → DistanceTraveled@0x124. CLAUDE.md §9 empirics: ~+27 score/100ms-step
at run start (grows with velocity), coin pickups give larger jumps, terminal step ≈ −1 reward (env reward =
raw score delta). Sim need only reproduce the per-step score-increment trend; calibrate distance→score scale.

## 6. Powerups (secondary) — assume ALL MAXED (hook maxes 0,1,2,5,6,7,8,9 at startup)
CharacterPlayer flags/timers @0x154-0x178. Activators 0x1298C24.. . Spawn odds = sProbability* statics.

## 7. Data-driven config
**No level data on disk** — only dump.cs/stringliteral.json/script.json (metadata, no value initializers) +
the unpacked-not APKs. Only external config = **PromoFile** (Runner SRV/MRV/ACCPTS/ACCFCS, Controls FOC/FON/SSD).
Code defaults [KNOWN]: SRV=100,MRV=210,SSDP=0.85,RSDP=0.85,RBD=150. **All generation statics + DistanceToChange*
are CODE-ONLY → primary calibration targets.**

## KNOWN vs INFERRED
**KNOWN (no calibration):** all enums; field offsets/method RVAs; obstacle→action map; linked-graph track +
neighbor offsets; generation-param NAMES; velocity defaults; "actions=direct calls"; "upgrades maxed".
**Needs short automated calibration (Frida-read statics + a few auto-runs):**
1. Accel curve ACCPTS/ACCFCS. 2. Gen knobs (sMinSpace*, sMaxBackToBack*, sDoubleObstaclePercent,
sMinSpacesBetweenCoinRuns, sMaxCoinsPerRun, sProbability*). 3. Distance thresholds (DistanceToChange*).
4. Action timings (Slide/Strafe/Stumble durations, accept windows). 5. Score constants. 6. The PRNG source.

**Single golden read-point:** `SavedPathElementStatics.Apply` RVA **0x12E9868** reads the WHOLE gen-param
block at once → one Frida read dumps every generation static. Also `PromoFile.ProcessPromo` 0x12A3CE0 for velocity config.

## Worth Ghidra-decompiling for exact algorithms (by RVA, if needed)
Generation: AddRandomPath 0x12E7584, GetRandomPathElementForSide 0x12E3E48, UpdatePathDescription 0x12E664C,
AddObstacle 0x12E2934, AddCoins 0x12E4BF4, HalfTheTime 0x12E2910.
Turn: CanTurnLeft 0x12E2E88, CanTurnRight 0x12E2F44, IsTurn 0x12E28B0, GetNearestPathElement 0x12E7B34.
Player: CharacterPlayer.Update 0x12971F0, GetRunVelocity 0x12982D8, Jump 0x1297B80, Kill 0x1297EB0.
Game loop: GameController.Update 0x126C60C, HandlePacing 0x126E768, ChangeLevel 0x1272C80,
HandleCollisionWithObstacles 0x12713E4, MakeSureDidntRunPastTurn 0x127089C.
