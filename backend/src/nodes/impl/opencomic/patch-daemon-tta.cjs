// opencomic-ai-bin 1.6.1 configures the native network before the first image.
// Put TTA in initArgs, which are also the key for reusing a daemon.
const fs = require('node:fs');
const path = require('node:path');
const {randomUUID} = require('node:crypto');
const {createRequire} = require('node:module');

const original = "const initFlags = ['-m', '-n', '-g', '-z', '-t', '-y', '-u', '-k', '-p', '-A', '--tile-profile'];";
const corrected = original.replace("'-p'", "'-x', '-p'");

function checkInstalled(entry) {
    if (!fs.readFileSync(entry, 'utf8').includes(corrected)) {
        throw new Error('This OpenComic runtime needs the daemon TTA fix. Run install-windows.ps1 for this runtime; see OPENCOMIC.md.');
    }
}

function patchRuntime(directory) {
    const deps = createRequire(path.resolve(directory, 'package.json'));
    const dist = path.dirname(deps.resolve('opencomic-ai-bin'));
    const pkg = JSON.parse(fs.readFileSync(path.join(dist, '..', 'package.json'), 'utf8'));
    if (pkg.version !== '1.6.1') throw new Error('Review daemon TTA support before updating opencomic-ai-bin ' + pkg.version);
    for (const name of ['index.cjs', 'index.mjs']) {
        const entry = path.join(dist, name);
        const source = fs.readFileSync(entry, 'utf8');
        if (source.includes(corrected)) continue;
        if (source.split(original).length !== 2) throw new Error('Unexpected daemon flag layout in ' + entry);
        const tmp = entry + '.' + randomUUID() + '.tmp';
        try {
            fs.writeFileSync(tmp, source.replace(original, corrected), {flag: 'wx'});
            fs.renameSync(tmp, entry);
        } finally {
            fs.rmSync(tmp, {force: true});
        }
    }
}

module.exports = {checkInstalled, patchRuntime};
if (require.main === module) patchRuntime(process.argv[2]);
