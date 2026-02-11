# Soccer-AI Server Setup (Ubuntu 22.04 Bare Metal)

These steps set up a **clean, bare-metal** Ubuntu 22.04 server with **Tesla P40 + Docker GPU**, plus the `/scratch` + `/tank` storage layout for the 360 soccer processing pipeline.

> **Important:** Be *absolutely sure* which disks are which before running any `mkfs` commands. Formatting the wrong disk is unrecoverable.

---

## 🧱 PHASE 1 — Install base OS (bare metal)

Use **Ubuntu 22.04 LTS Server**  
Do **NOT** use 24.04 yet (NVIDIA stack tends to be smoother on 22.04).

### Install options
- Minimal install
- OpenSSH server ✔
- No snaps (optional, but can make Docker life cleaner)

### After install
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget htop nvme-cli
```

### Set hostname
```bash
sudo hostnamectl set-hostname soccer-ai
```

### Reboot
```bash
sudo reboot
```

---

## 💾 PHASE 2 — Mount your drives properly

Goal:
- **512GB NVMe → `/scratch`**
- **4TB SSD → `/tank`**

### Check disks
```bash
lsblk
```

Assume:
- NVMe: `nvme0n1`
- SSD:  `sda`

> If your device names differ, **stop** and adjust commands accordingly.

### Format NVMe (scratch)
```bash
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir -p /scratch
sudo mount /dev/nvme0n1 /scratch
```

### Format 4TB SSD
```bash
sudo mkfs.ext4 /dev/sda
sudo mkdir -p /tank
sudo mount /dev/sda /tank
```

### Make mounts permanent (fstab)
Get UUIDs:
```bash
sudo blkid
```

Edit fstab:
```bash
sudo nano /etc/fstab
```

Add lines (replace `UUID=xxxx` with real UUIDs from `blkid`):
```text
UUID=xxxx  /scratch  ext4  defaults,noatime  0  2
UUID=xxxx  /tank     ext4  defaults,noatime  0  2
```

Mount everything:
```bash
sudo mount -a
```

Verify:
```bash
df -h | egrep '/scratch|/tank'
```

---

## 🧠 PHASE 3 — NVIDIA + CUDA (Tesla P40)

This is the part people usually mess up. Do it cleanly.

### Install drivers (datacenter)
```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

### Reboot
```bash
sudo reboot
```

### Check GPU
```bash
nvidia-smi
```

You should see: **Tesla P40 24GB**  
If yes → good.

---

## 🐳 PHASE 4 — Docker + GPU support

### Install Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Log out and log back in (or reboot) so group membership applies.

### Install NVIDIA container toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure
sudo systemctl restart docker
```

### Test GPU inside Docker
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

If you see the P40 → perfect.

---

## 📂 PHASE 5 — Create folder structure

```bash
sudo mkdir -p /tank/{ingest,processed,highlights,archive_raw,models,labeling}
sudo mkdir -p /scratch/work
sudo chown -R $USER:$USER /tank
sudo chown -R $USER:$USER /scratch
```

---

## 🔁 PHASE 6 — Auto move + cleanup logic (simple)

Pipeline will handle most of this later, but set up scratch cleanup now.

### Create daily cleanup script
```bash
sudo nano /etc/cron.daily/scratch-clean
```

Paste:
```bash
#!/bin/bash
find /scratch -type f -mtime +2 -delete
```

Make executable:
```bash
sudo chmod +x /etc/cron.daily/scratch-clean
```

(Optional) Run once to confirm it works (should do nothing unless files exist):
```bash
sudo /etc/cron.daily/scratch-clean
```

---

## 🧪 PHASE 7 — Test raw performance

Copy a large video into `/scratch` and run:

```bash
ffmpeg -i test.mp4 -f null -
```

Watch CPU usage:
```bash
htop
```

You should see many cores active. That means decoding will fly.

> If `ffmpeg` isn’t installed yet:
```bash
sudo apt install -y ffmpeg
```

---

## 🚀 PHASE 8 — Ready for AI pipeline repo

At this point your server is:
- Bare metal optimized
- GPU working
- Docker GPU enabled
- Storage structured
- Cleanup automated
- Ready for repo deployment

Next: hand the “big prompt” to your coding agent to build:
- Repo
- `docker-compose` stack
- Watcher service
- Pipeline worker
- Training workflow

---

## ⚙️ Optional but smart (5 min)

Install `nvtop`:
```bash
sudo apt install -y nvtop
```

Run:
```bash
nvtop
```

Live GPU usage viewer. You’ll use this constantly.

---

## 🧠 What happens next

Once your AI agent builds the repo, the loop is:

1. Record match  
2. Drop file into:
```bash
/tank/ingest
```
3. Go to bed  
4. Wake up → processed match + highlights ready

No editing.
