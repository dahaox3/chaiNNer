const assert = require('node:assert/strict');
const {createRequire} = require('node:module');
const path = require('node:path');
const test = require('node:test');
const {checkInstalled} = require('../src/nodes/impl/opencomic/patch-daemon-tta.cjs');

const runtime = process.env.CHAINNER_OPENCOMIC_RUNTIME || path.join(__dirname, '../src/nodes/impl/opencomic');
const deps = createRequire(path.resolve(runtime, 'package.json'));

test('TTA starts a distinct network and each mode reuses its loaded daemon', async t => {
    checkInstalled(deps.resolve('opencomic-ai-bin'));
    const AI = deps('opencomic-ai-bin');
    const started = [];
    const submitted = [];
    // Exercise the package's real argument partitioning and key lookup. Only
    // replace the native process so this regression can run without Vulkan.
    t.mock.method(AI, 'daemon', (key, binary, args, model, ready) => {
        started.push(args.slice());
        const daemon = {
            push: page => submitted.push({key, page}),
            close() {},
        };
        AI.daemons.set(key, daemon);
        ready();
        return daemon;
    });
    t.after(() => AI.closeAllDaemons());
    for (const tta of [false, true, true, false]) {
        const args = ['-i', 'page.png', '-o', 'out.png', '-m', 'models', ...(tta ? ['-x'] : []), '-p', '-n', 'model'];
        await AI.spawnDaemon('upscayl', args, {}, {});
    }
    assert.deepEqual(started, [
        ['-m', 'models', '-p', '-n', 'model'],
        ['-m', 'models', '-x', '-p', '-n', 'model'],
    ]);
    assert.equal(submitted[0].key, submitted[3].key);
    assert.equal(submitted[1].key, submitted[2].key);
    assert.notEqual(submitted[0].key, submitted[1].key);
    for (const {page} of submitted) assert.deepEqual(page, ['-i', 'page.png', '-o', 'out.png']);
});
