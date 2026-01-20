# Docker Setup

## Quick Start

### 1. Setup secrets

```bash
cp secrets.env.example secrets.env
# Edit secrets.env and add your Hugging Face token
```

The `secrets.env` file is git-ignored and will not be tracked.

### 2. Build the image

```bash
docker build -t primat:latest .
```

### 3. Run the container

```bash
./run_docker.sh
```

The script automatically loads your HF token from `secrets.env`.

## What's Included

The Docker image includes:
- **neovim** - Text editor
- **ranger** - File manager
- **zip** - Archive utility
- AMD ROCm optimizations pre-configured
- Profiling tools enabled

## Usage Examples

```bash
# Interactive shell
./run_docker.sh

# Run training
./run_docker.sh ./run_primus_all.sh

# Override HF token
HF_TOKEN=another_token ./run_docker.sh
```
