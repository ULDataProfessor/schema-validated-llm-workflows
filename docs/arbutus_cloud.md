# Arbutus Cloud runner

Use one small Arbutus OpenStack VM for live provider calls. The benchmark is
network- and provider-quota-bound, so do not request GPUs. Use Narval for
CPU-only generation, rescoring, and release checks; archive completed bundles
to HPSS.

## One-time setup

1. Log in to the [Arbutus dashboard](https://arbutus.alliancecan.ca) and confirm
   that you are in the intended cloud project.
2. Download that project's OpenRC file. Keep it outside this repository and do
   not share it; it contains authentication settings.
3. Install the OpenStack client on the machine where you will provision the VM:

   ```bash
   python3 -m pip install --user python-openstackclient
   source ~/Downloads/<project>-openrc.sh
   openstack image list
   openstack flavor list
   openstack network list
   ```

4. Create or import an SSH public key in the dashboard, then verify it with
   `openstack keypair list`. Select a current Ubuntu or Rocky Linux image and a
   modest CPU flavor (4--8 vCPUs and 16--32 GB RAM is sufficient).

## Provision the runner

Create a private local configuration file, never a committed one:

```bash
cp scripts/cloud/arbutus.env.example .arbutus.env
# Edit .arbutus.env using the exact names from the three `openstack ... list` commands.
source .arbutus.env
scripts/cloud/provision_arbutus_vm.sh --dry-run
scripts/cloud/provision_arbutus_vm.sh --apply
```

For a supported Ubuntu image, set
`ARBUTUS_USER_DATA=scripts/cloud/cloud-init-ubuntu-benchmark.yaml` to install
Git, Python virtual-environment support, rsync, and tmux during first boot. The
profile does not clone a repository or contain credentials; do not use it with
a non-Ubuntu image.

Set `ARBUTUS_SSH_CIDR` to a single trusted address such as `203.0.113.7/32`; do
not open SSH to `0.0.0.0/0`. The script creates only the configured VM and a
dedicated SSH security group. It is idempotent: re-running it reuses an existing
server and floating IP instead of creating another VM.

The script uses the OpenStack CLI, which is a supported client for the Arbutus
OpenStack API. It makes no API calls in `--dry-run` mode and never handles model
provider keys.

## Configure and run the benchmark

After `openstack server show <vm-name>` reports an attached floating IP, SSH to
the image's documented default user (commonly `ubuntu` for Ubuntu images):

```bash
ssh ubuntu@<floating-ip>
git clone <your-repository-url> pydantic-benchmark
cd pydantic-benchmark/benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
chmod 600 .env
# Add OPENAI_API_KEY only to this VM's .env file.
python scripts/generate_synthetic_data.py
python scripts/check_readiness.py --pilot
python scripts/run_benchmark.py --all --pilot --run-id arbutus-pilot-001
```

Use the same `--run-id` with `--resume` after an interruption. Start with the
pilot, inspect the result card and artifact audit, then launch a hardened full
matrix only after its protocol gate has passed. Cap concurrency to the provider
limits; more VM cores do not bypass API rate limits.

## Stop or remove the VM

Stopping a VM may continue to consume storage quota. Delete it after copying
the final result bundle to HPSS or another archival location:

```bash
source ~/Downloads/<project>-openrc.sh
source .arbutus.env
scripts/cloud/destroy_arbutus_vm.sh --dry-run
scripts/cloud/destroy_arbutus_vm.sh --apply
```

The floating IP is retained by default. Add `--release-floating-ip` only when
you no longer need that address.
