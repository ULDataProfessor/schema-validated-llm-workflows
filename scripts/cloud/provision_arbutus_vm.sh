#!/usr/bin/env bash
# Provision one reproducible Arbutus OpenStack VM for the live API benchmark.
#
# Usage:
#   source ~/Downloads/<project>-openrc.sh
#   source .arbutus.env
#   scripts/cloud/provision_arbutus_vm.sh --dry-run
#   scripts/cloud/provision_arbutus_vm.sh --apply
#
# The script deliberately never reads, writes, or transmits provider API keys.
# It only uses the authenticated OpenStack CLI session supplied by your OpenRC.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: provision_arbutus_vm.sh --dry-run|--apply

Creates, if absent, the configured security group and VM. It optionally allocates
and attaches one floating IP. Configuration comes from environment variables; see
scripts/cloud/arbutus.env.example and docs/arbutus_cloud.md.

--dry-run  Print the exact resources that would be created; make no API calls.
--apply    Perform the OpenStack API calls.
EOF
}

mode=""
case "${1:-}" in
    --dry-run|--apply) mode="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

required=(
    ARBUTUS_VM_NAME
    ARBUTUS_IMAGE
    ARBUTUS_FLAVOR
    ARBUTUS_NETWORK
    ARBUTUS_KEY_NAME
    ARBUTUS_SSH_CIDR
)
for variable in "${required[@]}"; do
    if [[ -z "${!variable:-}" ]]; then
        printf 'Missing required environment variable: %s\n' "$variable" >&2
        exit 2
    fi
done

security_group="${ARBUTUS_SECURITY_GROUP:-benchmark-runner-ssh}"
assign_floating_ip="${ARBUTUS_ASSIGN_FLOATING_IP:-1}"
user_data="${ARBUTUS_USER_DATA:-}"

if [[ "$assign_floating_ip" != "0" && "$assign_floating_ip" != "1" ]]; then
    printf 'ARBUTUS_ASSIGN_FLOATING_IP must be 0 or 1\n' >&2
    exit 2
fi
if [[ "$assign_floating_ip" == "1" && -z "${ARBUTUS_FLOATING_NETWORK:-}" ]]; then
    printf 'ARBUTUS_FLOATING_NETWORK is required when ARBUTUS_ASSIGN_FLOATING_IP=1\n' >&2
    exit 2
fi
if [[ -n "$user_data" && ! -f "$user_data" ]]; then
    printf 'ARBUTUS_USER_DATA does not exist: %s\n' "$user_data" >&2
    exit 2
fi

print_plan() {
    cat <<EOF
Arbutus provisioning plan
  VM:              $ARBUTUS_VM_NAME
  Image:           $ARBUTUS_IMAGE
  Flavor:          $ARBUTUS_FLAVOR
  Private network: $ARBUTUS_NETWORK
  SSH key pair:    $ARBUTUS_KEY_NAME
  Security group:  $security_group (TCP/22 from $ARBUTUS_SSH_CIDR)
  Floating IP:     $assign_floating_ip${ARBUTUS_FLOATING_NETWORK:+ on $ARBUTUS_FLOATING_NETWORK}
  Cloud-init:      ${user_data:-none}
EOF
}

if [[ "$mode" == "--dry-run" ]]; then
    print_plan
    exit 0
fi

if ! command -v openstack >/dev/null 2>&1; then
    printf 'The OpenStack CLI is required. See docs/arbutus_cloud.md.\n' >&2
    exit 1
fi
if [[ -z "${OS_AUTH_URL:-}" ]]; then
    printf 'OpenStack credentials are absent. Source the private OpenRC file first.\n' >&2
    exit 1
fi

ensure_security_group() {
    if ! openstack security group show "$security_group" >/dev/null 2>&1; then
        openstack security group create --description \
            "SSH access for the Pydantic benchmark runner" "$security_group"
    fi

    # A duplicate rule is harmless; OpenStack rejects it and leaves the existing
    # rule intact. Any other error is surfaced to the caller.
    if ! openstack security group rule create --ingress --protocol tcp \
        --dst-port 22:22 --remote-ip "$ARBUTUS_SSH_CIDR" "$security_group" 2>/dev/null; then
        if ! openstack security group rule list "$security_group" -f value -c "Remote IP Prefix" \
            | grep -Fxq "$ARBUTUS_SSH_CIDR"; then
            printf 'Could not create or verify the SSH security-group rule.\n' >&2
            exit 1
        fi
    fi
}

ensure_server() {
    if openstack server show "$ARBUTUS_VM_NAME" >/dev/null 2>&1; then
        printf 'VM already exists: %s\n' "$ARBUTUS_VM_NAME"
        return
    fi

    create=(
        openstack server create
        --image "$ARBUTUS_IMAGE"
        --flavor "$ARBUTUS_FLAVOR"
        --network "$ARBUTUS_NETWORK"
        --key-name "$ARBUTUS_KEY_NAME"
        --security-group "$security_group"
        --wait
    )
    if [[ -n "$user_data" ]]; then
        create+=(--user-data "$user_data")
    fi
    create+=("$ARBUTUS_VM_NAME")
    "${create[@]}"
}

ensure_floating_ip() {
    [[ "$assign_floating_ip" == "1" ]] || return

    local existing_ip
    existing_ip="$(openstack floating ip list --server "$ARBUTUS_VM_NAME" -f value -c "Floating IP Address" \
        | head -n 1 || true)"
    if [[ -n "$existing_ip" ]]; then
        printf 'Floating IP already attached: %s\n' "$existing_ip"
        return
    fi

    local address
    address="$(openstack floating ip create "$ARBUTUS_FLOATING_NETWORK" -f value -c floating_ip_address)"
    openstack server add floating ip "$ARBUTUS_VM_NAME" "$address"
    printf 'Attached floating IP: %s\n' "$address"
}

print_plan
ensure_security_group
ensure_server
ensure_floating_ip

printf '\nProvisioning complete. Inspect with:\n  openstack server show %q\n' "$ARBUTUS_VM_NAME"
