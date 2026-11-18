#!/usr/bin/env bash
# Delete only the VM named by ARBUTUS_VM_NAME. Floating IPs are deliberately
# retained by default because releasing them can make a reserved address unavailable.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: destroy_arbutus_vm.sh --dry-run|--apply [--release-floating-ip]

The --apply form deletes the configured server only after its name is resolved
through the authenticated OpenStack CLI. --release-floating-ip additionally
releases any floating IP attached to that server; use it only when the address
is no longer needed.
EOF
}

mode=""
release_ip=0
for argument in "$@"; do
    case "$argument" in
        --dry-run|--apply) mode="$argument" ;;
        --release-floating-ip) release_ip=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [[ -z "$mode" || -z "${ARBUTUS_VM_NAME:-}" ]]; then
    usage >&2
    printf '\nARBUTUS_VM_NAME must be set.\n' >&2
    exit 2
fi

if [[ "$mode" == "--dry-run" ]]; then
    printf 'Would delete VM: %s\n' "$ARBUTUS_VM_NAME"
    [[ "$release_ip" == "1" ]] && printf 'Would also release its attached floating IP.\n'
    exit 0
fi

if ! command -v openstack >/dev/null 2>&1 || [[ -z "${OS_AUTH_URL:-}" ]]; then
    printf 'Source the private OpenRC file and ensure the OpenStack CLI is installed.\n' >&2
    exit 1
fi

if ! openstack server show "$ARBUTUS_VM_NAME" >/dev/null 2>&1; then
    printf 'VM not found: %s\n' "$ARBUTUS_VM_NAME" >&2
    exit 1
fi

floating_ip="$(openstack floating ip list --server "$ARBUTUS_VM_NAME" -f value -c "Floating IP Address" \
    | head -n 1 || true)"
openstack server delete --wait "$ARBUTUS_VM_NAME"

if [[ "$release_ip" == "1" && -n "$floating_ip" ]]; then
    openstack floating ip delete "$floating_ip"
    printf 'Released floating IP: %s\n' "$floating_ip"
elif [[ -n "$floating_ip" ]]; then
    printf 'Retained floating IP: %s\n' "$floating_ip"
fi
