/* One-shot reader for the track-generation statics (PathElement static fields).
 *
 * The generation params are static fields of PathElement (dump.cs:283320-283351): obstacle/turn
 * spacing, double-obstacle %, coin runs, powerup spawn probabilities, allow-flags, level.
 * SavedPathElementStatics.Apply() only restores them at section transitions (which an un-steered
 * run never reaches), so instead we read the live statics directly via the exported il2cpp_* API:
 * resolve the PathElement class, iterate its fields, read every STATIC one by value.
 *
 * Emits the whole block over send({statics:{...}}). No run needed.
 */
'use strict';

const MOD = Process.findModuleByName('libil2cpp.so');
function ex(name, ret, args) {
  const p = MOD.findExportByName(name);
  if (!p) throw new Error('missing export ' + name);
  return new NativeFunction(p, ret, args);
}

function main() {
  const il2cpp_domain_get = ex('il2cpp_domain_get', 'pointer', []);
  const il2cpp_domain_get_assemblies = ex('il2cpp_domain_get_assemblies', 'pointer', ['pointer', 'pointer']);
  const il2cpp_assembly_get_image = ex('il2cpp_assembly_get_image', 'pointer', ['pointer']);
  const il2cpp_image_get_name = ex('il2cpp_image_get_name', 'pointer', ['pointer']);
  const il2cpp_class_from_name = ex('il2cpp_class_from_name', 'pointer', ['pointer', 'pointer', 'pointer']);
  const il2cpp_class_get_fields = ex('il2cpp_class_get_fields', 'pointer', ['pointer', 'pointer']);
  const il2cpp_field_get_name = ex('il2cpp_field_get_name', 'pointer', ['pointer']);
  const il2cpp_field_get_flags = ex('il2cpp_field_get_flags', 'int', ['pointer']);
  const il2cpp_field_static_get_value = ex('il2cpp_field_static_get_value', 'void', ['pointer', 'pointer']);

  const FIELD_ATTRIBUTE_STATIC = 0x10;

  // Find the Assembly-CSharp image.
  const domain = il2cpp_domain_get();
  const countPtr = Memory.alloc(Process.pointerSize);
  const assemblies = il2cpp_domain_get_assemblies(domain, countPtr);
  const count = countPtr.readU32();
  let image = null;
  for (let i = 0; i < count; i++) {
    const asm = assemblies.add(i * Process.pointerSize).readPointer();
    if (asm.isNull()) continue;
    const img = il2cpp_assembly_get_image(asm);
    if (img.isNull()) continue;
    const nm = il2cpp_image_get_name(img).readCString();
    if (nm === 'Assembly-CSharp.dll') { image = img; break; }
  }
  if (image === null) { send({ statics_log: ['no Assembly-CSharp image', count] }); return; }

  const klass = il2cpp_class_from_name(image, Memory.allocUtf8String(''), Memory.allocUtf8String('PathElement'));
  if (klass.isNull()) { send({ statics_log: ['PathElement class not found'] }); return; }

  // The fields we care about (name -> "int"|"float"|"bool"). Names matched against the live fields.
  const WANT = {
    sCount: 'int', sSafetyNetEnabled: 'bool', sAllowBonusItems: 'bool', sAllowVariations: 'bool',
    sAllowTurns: 'bool', sAllowPathHoles: 'bool', sAllowPathNarrows: 'bool', sAllowFlameTowers: 'bool',
    sAllowTreeStumble: 'bool', sAllowTreeJump: 'bool', sAllowTreeSlide: 'bool',
    sAllowTurnAfterObstacles: 'bool', sAllowSidePaths: 'bool', sAllowCoins: 'bool',
    sMinSpacesBetweenBonusItems: 'int', sProbabilityCoinBonus: 'float',
    sProbabilityInvincibility: 'float', sProbabilityVacuum: 'float', sProbabilityBoost: 'float',
    sMinSpaceBetweenTurns: 'int', sMaxBackToBackObstacles: 'int', sMinSpaceBetweenObstacles: 'int',
    sDoubleObstaclePercent: 'float', sMinSpacesBetweenCoinRuns: 'int', sMaxCoinsPerRun: 'int',
    sPathLevel: 'int',
  };

  const out = {};
  const iter = Memory.alloc(Process.pointerSize);
  iter.writePointer(ptr(0));
  let f;
  const valbuf = Memory.alloc(16);
  while (!(f = il2cpp_class_get_fields(klass, iter)).isNull()) {
    const flags = il2cpp_field_get_flags(f);
    if (!(flags & FIELD_ATTRIBUTE_STATIC)) continue;
    const name = il2cpp_field_get_name(f).readCString();
    if (!(name in WANT)) continue;
    valbuf.writeU64(0);
    il2cpp_field_static_get_value(f, valbuf);
    const t = WANT[name];
    if (t === 'float') out[name] = valbuf.readFloat();
    else if (t === 'bool') out[name] = valbuf.readU8() !== 0;
    else out[name] = valbuf.readS32();
  }
  send({ statics: out });
}

try { main(); } catch (e) { send({ statics_log: ['error', String(e)] }); }
