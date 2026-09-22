# GitHub 私有仓库与自动同步

## 2026-09-22 · 上传与同步范围确定
- 用户关键原话：“把这个项目直接传到github，后续更新其他功能的时候，在github上自动更新文件和代码”；此前要求“保留完整仓库私有”。
- 关键产出：采用私有仓库 `ny-glitch/ecommerce-support-agent`；保留本地提交历史与当前分支，上传代码、文档、测试及展示材料，排除.env、缓存、虚拟环境、原始录制帧及数据库备份。同步以提交/合并为触发点，不监听文件保存。
- 拒绝或纠偏：不是公开展示仓库，不开放后端或GitHub Pages；不因上传而改变隐私要求。未提交的临时修改不会自动上传，推送不使用force。
- 翻车与返工：本机无GitHub CLI，已从官方发行源下载2.101.0并校验checksum，仅放在缓存；插件账号已连接，但CLI尚无推送认证，已发起官方设备授权等待用户完成。历史609个blob的已知配置密钥及常见token/私钥模式扫描无匹配，未输出密钥值。

## 2026-09-22 · 自动同步实现与隔离验证
- 用户关键原话：“后续更新其他功能的时候，在github上自动更新文件和代码”。
- 关键产出：新增 `.githooks/post-commit`、`post-merge` 与 `scripts/github_auto_push.sh`，只在本仓库显式启用后向确认过的origin同名分支推送。新增 `AGENTS.md` 持续约定及 `docs/github-sync.md` 使用说明。
- 拒绝或纠偏：不自动暂存、不强推；未启用/单次跳过/地址改变/多推送地址/变基过程中不推送；失败保留本地提交并提示重试。测试使用真实临时Git仓库和本地bare远端，不替换Git操作为假函数。
- 翻车与返工：先见3项自动推送能力测试失败，再实现得到8 passed（4.89s）；当前全量required suite进行中。用户已完成第一次网页登录，但CLI领取凭据时unexpected EOF，auth status确认未保存；无效占位参数连通性检查恢复正常后重启设备授权，未打印真实token或关闭TLS验证。

## 2026-09-22 · 认证、仓库创建与完整回归
- 用户关键原话：“已完成授权”。
- 关键产出：GitHub CLI 确认登录 `ny-glitch`；创建私有仓库 `https://github.com/ny-glitch/ecommerce-support-agent` 并核实 `isPrivate: true`。官方已校验的 CLI 安装到 `~/.local/bin/gh`，Git 认证助手改用稳定路径，避免依赖可清理缓存。全量隔离验证 `pytest --require-mysql --require-postgres --require-milvus --require-local-models -q --tb=line` 得到 **815 passed, 1 warning（126.82s）**。
- 拒绝或纠偏：保留已有分支和提交历史，不额外创建或切换 `main`；私有仓库创建成功不等于代码已上传，后续仍需提交、推送及远端 SHA 核验。没有发起真实模型 API 调用。
- 翻车与返工：首次 OAuth token 交换 EOF 在重新走官方授权后恢复；既有 Starlette/AnyIO 弃用提示与本次同步功能无关。当前同步实现交由独立 reviewer 检查，通过后启用。

## 2026-09-22 · Code review 结论与返工
- 用户关键原话：“后续更新其他功能的时候，在github上自动更新文件和代码”。
- 关键产出：独立 reviewer 确认核心范围符合要求，提出两项 P2：首次推送失败时没有 upstream，原 `git push` 重试提示无效；分支与 tag 同名时 `symbolic-ref --short` 可产生错误的目标分支。两项均用真实临时仓库复现失败后修复：改读完整 `refs/heads/…`，重试复用带远程校验的同步脚本。
- 拒绝或纠偏：没有给钩子增加全量测试、周期扫描或额外后台任务；钩子继续只负责推送已提交内容，验证与私有权限核验由开发/发布流程负责。
- 翻车与返工：恢复提示回归先见 2 failed / 7 passed；同名 tag 回归先见 1 failed。修复后同步专项 **10 passed（5.24s）**，shell 语法检查通过。reviewer 二次只读复核确认两项均闭合，无剩余 Critical / Important 问题。暂存检查提示 SRT/VTT 最后字幕块后的格式空行，保留合法字幕分隔，仅对这两个文件忽略 blank-at-eof 再检查，其余文件维持完整 whitespace 检查。
