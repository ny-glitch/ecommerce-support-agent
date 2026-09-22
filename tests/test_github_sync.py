"""Exercise publishing hooks against real, isolated Git repositories."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, check=True, extra_env=None):
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    # Tests never inherit a hook's repository routing when called from Git.
    for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR'):
        env.pop(key, None)
    env.update(extra_env or {})
    return subprocess.run(['git', '-C', str(path), *args], text=True,
                          capture_output=True, check=check, env=env)


@pytest.fixture
def repositories(tmp_path):
    local, remote = tmp_path / 'working', tmp_path / 'remote.git'
    local.mkdir()
    git(local, 'init', '-b', 'codex/sync-test')
    git(local, 'config', 'user.name', 'Sync Test')
    git(local, 'config', 'user.email', 'sync@example.invalid')
    git(local, 'init', '--bare', str(remote))
    git(local, 'remote', 'add', 'origin', str(remote))
    # Absent hooks are a no-op in the initial red run.
    if (ROOT / '.githooks').exists():
        shutil.copytree(ROOT / '.githooks', local / '.githooks')
    (local / 'scripts').mkdir()
    source = ROOT / 'scripts/github_auto_push.sh'
    if source.exists():
        shutil.copy2(source, local / 'scripts/github_auto_push.sh')
    git(local, 'config', 'core.hooksPath', '.githooks')
    git(local, 'config', 'support.autoPush', 'true')
    git(local, 'config', 'support.expectedRemote', str(remote))
    return local, remote


def commit(local, value, extra_env=None):
    (local / 'change.txt').write_text(value)
    git(local, 'add', 'change.txt')
    return git(local, 'commit', '-m', value, extra_env=extra_env)


def remote_head(remote):
    result = git(remote, 'rev-parse', '--verify', 'refs/heads/codex/sync-test', check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def test_commit_publishes_to_same_branch_without_staging_uncommitted_files(repositories):
    local, remote = repositories
    (local / 'private-uncommitted.txt').write_text('local only')
    commit(local, 'first')
    assert remote_head(remote) == git(local, 'rev-parse', 'HEAD').stdout.strip()
    assert git(local, 'rev-parse', '--abbrev-ref', '@{upstream}').stdout.strip() == 'origin/codex/sync-test'
    assert git(remote, 'ls-tree', '--name-only', 'refs/heads/codex/sync-test').stdout.splitlines() == ['change.txt']


@pytest.mark.parametrize('mode', ['disabled', 'explicit-skip', 'wrong-remote', 'extra-push-remote', 'rebase'])
def test_does_not_publish_without_authorized_context(repositories, tmp_path, mode):
    local, remote = repositories
    extra_env = None
    if mode == 'disabled':
        git(local, 'config', 'support.autoPush', 'false')
    elif mode == 'explicit-skip':
        extra_env = {'SUPPORT_SKIP_AUTO_PUSH': '1'}
    elif mode == 'wrong-remote':
        git(local, 'config', 'support.expectedRemote', str(tmp_path / 'another.git'))
    elif mode == 'extra-push-remote':
        git(local, 'remote', 'set-url', '--add', '--push', 'origin', str(remote))
        git(local, 'remote', 'set-url', '--add', '--push', 'origin', str(tmp_path / 'another.git'))
    elif mode == 'rebase':
        (local / '.git/rebase-merge').mkdir()
    result = commit(local, 'local-change', extra_env)
    assert result.returncode == 0
    assert remote_head(remote) is None
    assert git(local, 'show', 'HEAD:change.txt').stdout == 'local-change'


def test_rejected_push_keeps_local_commit_and_remote_history(repositories):
    local, remote = repositories
    commit(local, 'first')
    first = remote_head(remote)
    assert first is not None
    reject = remote / 'hooks/pre-receive'
    reject.write_text('#!/bin/sh\nexit 1\n')
    reject.chmod(0o755)
    result = commit(local, 'second')
    assert remote_head(remote) == first
    assert git(local, 'show', 'HEAD:change.txt').stdout == 'second'
    assert 'run: sh scripts/github_auto_push.sh' in result.stderr


def test_first_push_failure_can_recover_without_an_upstream(repositories):
    local, remote = repositories
    git(local, 'config', 'push.default', 'simple')
    git(local, 'config', 'push.autoSetupRemote', 'false')
    reject = remote / 'hooks/pre-receive'
    reject.write_text('#!/bin/sh\nexit 1\n')
    reject.chmod(0o755)
    result = commit(local, 'first')
    assert remote_head(remote) is None
    assert git(local, 'rev-parse', '@{upstream}', check=False).returncode != 0
    assert git(local, 'push', check=False).returncode != 0
    assert 'run: sh scripts/github_auto_push.sh' in result.stderr

    reject.unlink()
    git(local, '-c', 'alias.retry-sync=!sh scripts/github_auto_push.sh', 'retry-sync')
    assert remote_head(remote) == git(local, 'rev-parse', 'HEAD').stdout.strip()
    assert git(local, 'rev-parse', '--abbrev-ref', '@{upstream}').stdout.strip() == 'origin/codex/sync-test'


def test_fast_forward_merge_publishes_the_new_head(repositories):
    local, remote = repositories
    commit(local, 'first')
    git(local, 'switch', '-c', 'codex/feature')
    commit(local, 'feature', {'SUPPORT_SKIP_AUTO_PUSH': '1'})
    feature_head = git(local, 'rev-parse', 'HEAD').stdout.strip()
    git(local, 'switch', 'codex/sync-test')
    git(local, 'merge', '--ff-only', 'codex/feature')
    assert remote_head(remote) == feature_head


def test_same_named_tag_does_not_change_destination_branch(repositories):
    local, remote = repositories
    commit(local, 'first')
    git(local, 'tag', 'codex/sync-test')
    commit(local, 'second')
    assert remote_head(remote) == git(local, 'rev-parse', 'HEAD').stdout.strip()
    assert git(remote, 'for-each-ref', '--format=%(refname)', 'refs/heads').stdout.splitlines() == ['refs/heads/codex/sync-test']
