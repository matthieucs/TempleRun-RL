// Diagnostic: what does Frida see inside this (possibly translated) process?
console.log('Process.arch    = ' + Process.arch);
console.log('Process.pageSize= ' + Process.pageSize);
console.log('pointerSize     = ' + Process.pointerSize);
const mods = Process.enumerateModules();
console.log('total modules   = ' + mods.length);
console.log('findModuleByName(libil2cpp.so) = ' + Process.findModuleByName('libil2cpp.so'));
console.log('findModuleByName(libunity.so)  = ' + Process.findModuleByName('libunity.so'));
console.log('--- modules of interest ---');
mods.forEach(function (m) {
  var n = (m.name + ' ' + (m.path || '')).toLowerCase();
  if (n.indexOf('il2cpp') >= 0 || n.indexOf('unity') >= 0 || n.indexOf('imangi') >= 0 ||
      n.indexOf('templerun') >= 0 || n.indexOf('translat') >= 0 || n.indexOf('houdini') >= 0 ||
      n.indexOf('ndk') >= 0 || n.indexOf('arm') >= 0 || n.indexOf('libc.so') >= 0) {
    console.log('  ' + m.name + '  base=' + m.base + '  size=' + m.size + '  ' + (m.path || ''));
  }
});
console.log('--- first 12 module names ---');
mods.slice(0, 12).forEach(function (m) { console.log('  ' + m.name + '  ' + (m.path || '')); });
console.log('=== DIAG DONE ===');
