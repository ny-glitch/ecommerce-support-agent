#!/bin/sh
# Explicit repository opt-in; never stage files or rewrite remote history.
set -u

[ "${SUPPORT_SKIP_AUTO_PUSH:-0}" = "1" ] && exit 0
[ "$(git config --bool --get support.autoPush 2>/dev/null || :)" = "true" ] || exit 0

branch_ref=$(git symbolic-ref --quiet HEAD) || exit 0
case "$branch_ref" in
    refs/heads/*) branch=${branch_ref#refs/heads/} ;;
    *) exit 0 ;;
esac
for operation in rebase-merge rebase-apply; do
    [ ! -d "$(git rev-parse --git-path "$operation")" ] || exit 0
done

expected=$(git config --get support.expectedRemote 2>/dev/null || :)
actual=$(git remote get-url --push --all origin 2>/dev/null || :)
if [ -z "$expected" ] || [ "$actual" != "$expected" ]; then
    printf '%s\n' '[GitHub sync] Remote differs from the configured destination; local commit kept, auto-push skipped.' >&2
    exit 0
fi

printf '[GitHub sync] Publishing committed branch %s...\n' "$branch" >&2
if ! GIT_TERMINAL_PROMPT=0 GH_PROMPT_DISABLED=1 git push --set-upstream origin "HEAD:$branch_ref"; then
    printf '%s\n' '[GitHub sync] Push failed. Your local commit is safe. Resolve authentication/network/history issues, then run: sh scripts/github_auto_push.sh' >&2
fi
exit 0
