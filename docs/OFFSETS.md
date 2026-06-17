# IL2CPP offsets & RVAs reference

> **These are specific to Temple Run v1.39.0 (`com.imangi.templerun`, Unity 2022.3.62f2, IL2CPP).**
> They are baked into `frida/temple_run_hook.js`. A different APK version **will** have different
> values — re-dump with Il2CppDumper (see [SETUP.md](SETUP.md) step 2) and update the `OFF` / `RVA`
> blocks in the agent. The host Python layer never uses these directly; it keys only on the JSON
> field names, so as long as the 19-field state contract holds, no Python change is needed.

All values come from `il2cpp_work/output/dump.cs`. **Field** = instance byte offset; **RVA** = offset
added to the runtime base of `libil2cpp.so`.

## CharacterPlayer (instance fields)
| Field | Offset | | Field | Offset |
|---|---|---|---|---|
| IsDead | `0xD8` | | Score | `0x138` |
| IsJumping | `0xF8` | | Velocity (Vec3) | `0x34` |
| IsSliding | `0x100` | | Facing (PathDirection, from CommonPlayer) | `0x78` |
| IsOnGround | `0x130` | | IsStrafing | `0x125` |
| IsStumbling | `0x10C` | | HasInvincibility | `0x154` |
| HasAngelWings | `0x161` | | HasVacuum | `0x16C` |
| HasBoost | `0x174` | | IsMegaBoost | `0x175` |
| GameOverInterface (EndGameUI*) | `0x1A0` | | | |

## GameController (instance fields)
| Field | Offset |
|---|---|
| TutorialID | `0x48` |
| GamePlayer (CharacterPlayer*) | `0x70` |
| IsGameOver | `0xDE` |
| IsTutorialMode | `0x104` |
| NearestElement (PathElement*) | `0x88` |
| RecordManager | `0xB0` |

## PathElement (instance fields)
| Field | Offset | | Field | Offset |
|---|---|---|---|---|
| Type | `0x30` | | PathNorth | `0x50` |
| Level | `0x34` | | PathEast | `0x58` |
| Obstacles (enum) | `0x38` | | PathWest | `0x60` |
| | | | PathSouth | `0x68` |

`must_jump` = `Obstacles ∈ {1 hole, 4 treeJump}` · `must_slide` = `Obstacles ∈ {5 treeSlide, 6 flameTower}`.
`PathDirection` enum (= `CharacterPlayer.Facing`): None 0, North 1, East 2, South 3, West 4.
Turn availability at the nearest element is read via the game's own methods (below), with `origin` = the
player's `Facing`.

## Method RVAs
| Method | RVA | | Method | RVA |
|---|---|---|---|---|
| GameController.Start | `0x1269CE8` | | Jump | `0x1297B80` |
| GameController.Update | `0x126C60C` | | Slide | `0x1298768` |
| RestartGame | `0x126B5B8` | | StrafeLeft | `0x1298874` |
| EndGameUI.RunAgainClicked | `0x12BCCF0` | | StrafeRight | `0x1298908` |
| GetRunVelocity | `0x12982D8` | | TurnLeft | `0x129899C` |
| DeserializeRecordManager | `0x128AF58` | | TurnRight | `0x1298A14` |
| SerializeRecordManager | `0x128BDC8` | | SetPlayerLevelForUpgradeType | `0x128D7BC` |
| GetPlayerLevelForUpgradeType | `0x128D568` | | GetStoreItemMaxLevel | `0x128E5D8` |
| PlayerPrefs.GetInt(string,int) | `0x24B37B8` | | PathElement.IsTurn() | `0x12E28B0` |
| PathElement.CanTurnLeft(dir) | `0x12E2E88` | | PathElement.CanTurnRight(dir) | `0x12E2F44` |
| MainMenu.Update() | `0x12C94C8` | | MainMenu.OnPlayClicked() | `0x12CA204` |

`MainMenu` is a singleton (static `Instance` @ field `0x0`); the agent captures the live instance in
`MainMenu.Update` and calls `OnPlayClicked` (the "JOUER" button) for action byte 9 — starting a run from
the main menu, the counterpart to `RunAgainClicked` (byte 8) which starts from the game-over screen.

`Il2CppString` layout (used by the PlayerPrefs hook): length @ `0x10`, UTF-16 chars @ `0x14`.
Instance-method native ABI: first arg = `this`, trailing arg = `MethodInfo*` (pass NULL).

## StoreItemType (upgrade indices)
AngelWings 0 · HeadStart 1 · HeadStartMega 2 · Vacuum 5 · Invincibility 6 · CoinBonus 7 · CoinValue 8 · Boost 9.
Leveled (max > 0): 5, 6, 7, 8, 9 → maxed to Vacuum/Invincibility/CoinBonus/Boost = 8, CoinValue = 5
(0/1/2 are consumable counts, max 0).
