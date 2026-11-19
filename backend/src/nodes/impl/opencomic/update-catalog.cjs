// Run after installing the pinned npm dependencies. Commit the generated catalog.
const fs = require('node:fs/promises');
const path = require('node:path');
const AI = require('opencomic-ai-bin');
const revision = 'ca1c17faabcb029a14b8bcabfc864227d6b9e2b2';

async function main() {
    const response = await fetch('https://api.github.com/repos/ollm/opencomic-ai-models/git/trees/' + revision + '?recursive=1');
    if (!response.ok) throw new Error('GitHub returned ' + response.status);
    const {tree, truncated} = await response.json();
    if (truncated) throw new Error('Incomplete model listing');
    const blobs = new Map(tree.map(entry => [entry.path, entry]));
    const models = {}, files = {};
    for (const type of ['artifact-removal', 'descreen', 'descreen-mask', 'upscale']) {
        models[type] = {};
        for (const key of AI.modelsTypeList[type]) {
            const model = AI.model(key);
            const names = model.files.map(file => path.posix.join(type, model.folder, file));
            models[type][key] = {name: model.name, scales: model.scales, noise: model.noise, files: names};
            for (const name of names) {
                const entry = blobs.get('models/' + name);
                if (!entry || entry.type !== 'blob') throw new Error('Missing ' + name);
                files[name] = [entry.size, entry.sha];
            }
        }
    }
    await fs.writeFile(path.join(__dirname, 'catalog.json'), JSON.stringify({revision, models, files}, null, 2) + '\n');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
