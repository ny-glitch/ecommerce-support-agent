# GitHub 私有仓库与自动同步

完整项目上传到私有仓库。源码、文档、测试和展示材料随 Git 提交更新；模型缓存、运行密钥、数据库数据和本地备份不进入 Git。

仓库地址：[ny-glitch/ecommerce-support-agent](https://github.com/ny-glitch/ecommerce-support-agent)。默认分支沿用现有的 `codex/ch04-hybrid-rag`，其中已合入后续章节功能；分支名称不代表功能只到第 4 章。

## 何时同步

本机配置完成后：

1. 完成功能修改并运行适用验证。
2. 提交本次应交付的文件。
3. `post-commit` 自动将当前分支推送到 `origin` 的同名分支；成功合并后的 `post-merge` 也执行同步。

仅保存文件不会上传，未暂存和未提交的改动不会自动打包。功能分支的推送只更新对应分支，不会自动合并到默认分支。以后让 Codex 在本项目中开发时，根目录 `AGENTS.md` 要求完成验证后提交并核对同步结果。

推送失败不会丢失本地提交。修复网络或登录后，在自动同步仍启用的情况下，于仓库根目录执行：

```bash
sh scripts/github_auto_push.sh
```

这个命令会重新核对远程地址，并为首次成功推送的分支设置 upstream。成功时应看到 Git 的推送结果；失败或被跳过时脚本仍返回 0，以免影响本地提交，因此不能只凭退出码判断同步成功。

远程历史有分歧时，先检查并正常合并或变基；不要使用强推覆盖他人的提交。变基期间自动推送暂时跳过，结束后人工核对再推送。

## 本机网络配置

本机 macOS 已启用 `127.0.0.1:7890` 代理，Git 直连 GitHub 曾出现空响应和超时。因此当前仓库的本地 Git 设置沿用该代理，并设置传输低速超时。该设置不进入仓库，也不会影响其他项目。

若以后更换代理端口，更新当前仓库的 `http.https://github.com.proxy`；若系统已不使用代理且能直连 GitHub，可移除这项设置：

```bash
git config --local --unset http.https://github.com.proxy
```

推送失败时，本地提交仍然保留；恢复网络后按上面的命令重试。不要关闭 TLS 校验。

## 新电脑或重新克隆后启用

先配置自己的 GitHub Git 认证，确认 `origin` 指向本项目私有仓库。再在仓库根目录执行：

```bash
git remote -v
git config --local core.hooksPath .githooks
git config --local support.expectedRemote "$(git remote get-url --push origin)"
git config --local support.autoPush true
```

这些是本地仓库设置，Git 不会将它们自动带到新克隆中。若此前已有自己的 hooksPath，应先整合现有钩子，不要直接覆盖。本项目钩子只向已确认的单一 `origin` 推送地址发送内容；地址改变或多推送地址时会停止自动同步。

## 暂停

```bash
# 暂停后续自动推送；本地提交功能不受影响
git config --local support.autoPush false

# 恢复
git config --local support.autoPush true

# 仅本次提交不推送
SUPPORT_SKIP_AUTO_PUSH=1 git commit -m "本地暂存进度"
```

## 展示材料

完整私有仓库与对外展示包是两回事。需要对外展示时，只发送 `showcase/` 或单独发布其静态页面，不要将完整仓库改成公开。本配置不会发布 GitHub Pages，也不会部署或重启客服后端。
