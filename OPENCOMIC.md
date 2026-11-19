# OpenComic 漫画修复

在 chaiNNer 的 Image 分类搜索 **OpenComic AI**。节点依次执行去压缩伪影、去网点和超分，每个模型菜单都能选 **Off / 关闭**。这三个菜单来自 OpenComic 官方运行库 1.6.1，分别有 12、12、31 个内置选项。模型名称中有版本和质量档位。

## 阅读时使用

- 打开随本机交付的 `ComicRead-OpenComic-修复与超分-2x.chn`，点击 **Run**。它用现有 ComicRead 连接，默认启用 Balanced V3 去伪影、Fast V4 Preview 去网点、保留大网点和 Fast V3 固定 2×。示例的输入面积上限为 0，表示所有尺寸都做超分。已经运行其他阅读工作流时，先保存并停止，再打开这个示例。
- 仓库和便携压缩包附通用模板 `ComicRead-OpenComic-2x.chn`，首次打开时在 Receive Image 填自己的令牌和成品缓存目录。本机交付的中文文件已填入现有连接设置。
- 在现有黑白／彩色工作流中，若想让自己的 PyTorch 模型负责超分，把 OpenComic AI 接在图片通往 Upscale Image 的连线上，OpenComic 节点的 Upscaling 选 Off。原图仍可直接供 Is Grayscale 判断，Request ID 仍直接连到 Return Image。
- **保留大网点**按页面高度和尺寸阈值检测应保留的网点，默认阈值 3%。阈值越大，越偏向保留较粗的网点。开启后，官方管线会生成掩膜并合成图像。若已全图去伪影，直接使用清理过的图像。只开去网点时，官方会单独清理保留区域。V4 Preview 模型自身会抑制摩尔纹。
- **超分**可选手动或自动倍率。手动倍率取该模型支持的最近值。自动倍率根据输入面积和 Max Input Megapixels 计算 2× 到 4× 的目标，再取模型支持的最近值。超过上限时，去伪影和去网点照常执行，只跳过超分。上限为 0 时解除限制，自动倍率仍用 2 MP 作为参考。Waifu2x 支持多次 2× 推理。Real-CUGAN 和 Waifu2x 会显示降噪级别。
- 三步都支持自定义兼容 Upscayl 的 NCNN 权重。将同名 `.param` 和 `.bin` 放在一起，选择对应步骤的 Custom NCNN，然后选择 `.param`。自定义超分模型另填原生倍率。高级选项有分块大小、TTA 和强制 FP32，应用于已启用的步骤。
- 用在普通图片时，也可以连接 Load Image 和 Save Image。节点按 8 位 RGB 将像素传给官方后端，灰度图返回灰度。透明度单独保留，放大时线性缩放。修改图参数或模型后，停止并重新运行阅读工作流。

## 运行库与文件

便携版的 `resources/opencomic-runtime` 包含 Node、Sharp 和官方原生 Vulkan 程序。每次工作流执行共享一个通过私有管道连接的进程，官方库复用已加载的 GPU 模型，按自己的空闲策略释放。停止工作流后释放进程。该节点与现有 ComicRead 服务共用相同的接收和返回图片节点。

权重保存在便携根目录的 `models/opencomic`，其中包含各类型的子目录。启动处理前核对所选模型及辅助模型的文件，首次缺失时从固定版本的官方模型仓库下载，校验长度和 Git 对象摘要后再放入模型目录。下载中断可重试。临时 PNG 在 `runtime-temp/opencomic`，任务结束后清理。ComicRead 的成品缓存继续使用 Receive Image 节点里配置的目录。

## 维护与重建

适配代码位于 `backend/src/nodes/impl/opencomic`。在 Windows x64 开发机上安装 Node 和 npm 后，将脚本的 Destination 指向便携版的 `resources/opencomic-runtime`，或源码检出的 `backend/opencomic-runtime`。安装脚本在此专用目录执行 npm ci，版本以随附的 lockfile 为准。

```powershell
.\backend\src\nodes\impl\opencomic\install-windows.ps1 -Destination 'F:\chaiNNer-GFISRv2\resources\opencomic-runtime'
```

可用 `CHAINNER_OPENCOMIC_RUNTIME` 和 `CHAINNER_OPENCOMIC_MODELS` 指定其他目录。升级时核对 OpenComic 阅读器的步骤顺序、掩膜与自动倍率逻辑，同步 package.json、lockfile 和 update-catalog.cjs 的模型仓库版本。将 NODE_PATH 指向运行库的 node_modules 后运行 `node update-catalog.cjs` 重新生成目录，更新 pipeline.py 的缓存版本并运行 `pytest backend/tests/test_opencomic.py`。GPU 回归覆盖固定／自动倍率、保留大网点的官方样图，以及服务停止后的原生进程释放。

OpenComic AI 模型作者为 Oleguer Llopart Mora，官方权重采用 CC BY 4.0。模型文件按上游原样使用。第三方模型分别适用其作者的条款。官方运行库附带其代码和模型许可，原生 Upscayl 后端的许可和源码来自作者维护的仓库，Node 与 npm 依赖的许可随运行库保留。[原生后端源码与构建说明](https://github.com/ollm/upscayl-ncnn) · [官方 AI 指南](https://www.opencomic.ollm.dev/docs/guides/ai/) · [官方运行库](https://github.com/ollm/opencomic-ai-bin) · [固定版本的权重及来源](https://github.com/ollm/opencomic-ai-models/tree/ca1c17faabcb029a14b8bcabfc864227d6b9e2b2) · [阅读器的处理顺序](https://github.com/ollm/OpenComic/blob/34d5e3112a53cd10107cf924525ba20c2376c0a4/scripts/reading/render/ai.js#L97-L166)
