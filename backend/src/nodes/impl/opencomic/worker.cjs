// A private stdio worker per chaiNNer execution. The official library owns the GPU daemons.
const {createRequire} = require('node:module');
const path = require('node:path');
const readline = require('node:readline');
const {checkInstalled} = require('./patch-daemon-tta.cjs');
const deps = createRequire(path.join(process.argv[2], 'package.json'));
const AI = deps('opencomic-ai-bin');
AI.setSharp(deps('sharp'));
AI.setModelsPath(process.argv[3]);
AI.keepIccProfile('rgb16');

const lines = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
const send = data => process.stdout.write(JSON.stringify(data) + '\n');

function registerCustom(step) {
    if (!step.custom) return;
    const {param, scale, stage} = step.custom;
    const name = path.basename(param, path.extname(param));
    AI.setCustomModel({
        key: step.model, type: stage, name, upscaler: 'upscayl', latency: 0,
        scales: [scale], scaleFiles: {[scale]: name}, modelsPath: path.dirname(param),
        folder: '.', files: [name + '.param', name + '.bin'],
        tileAlignment: 4, tileSizeFromMem: AI.tileMemoryProfiles.ESRGAN_Upscale,
    });
    delete step.custom;
}

async function main() {
    let ttaChecked = false;
    for await (const line of lines) {
        try {
            const {source, dest, steps} = JSON.parse(line);
            if (!ttaChecked && steps.some(step => step.tta)) {
                checkInstalled(deps.resolve('opencomic-ai-bin'));
                ttaChecked = true;
            }
            steps.forEach(registerCustom);
            await AI.pipeline(source, dest, steps, progress => send({progress}));
            send({done: true});
        } catch (error) {
            send({error: String(error.message || error)});
        }
    }
}
main().finally(() => AI.closeAllDaemons()).catch(error => {
    console.error(error);
    process.exitCode = 1;
});
