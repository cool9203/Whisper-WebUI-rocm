# Build with rocm

<div align="center">

[Upstream README](README.md) | **ROCm Install Guide**
</div>

## Quick install with gpu-gfx906 and rocm-6.3

If not is gfx906 and rocm-6.3, you need build your image

```bash
docker pull yogawulala/whisper-webui-rocm:rocm6.3-gfx906
```

## Change to your device setting

https://github.com/cool9203/Whisper-WebUI-rocm/blob/44a878fe597eaa2d1bafc61d4eb58fe8d37a9acb/Dockerfile-rocm#L14

Change to your rocm version.

https://github.com/cool9203/Whisper-WebUI-rocm/blob/44a878fe597eaa2d1bafc61d4eb58fe8d37a9acb/Dockerfile-rocm#L24

Change to your amdgpu.
You can use `rocminfo | grep gfx` to get.

## Build

```bash
docker build . -f Dockerfile-rocm -t <IMAGE_NAME>
```

## Run

First change `docker-compose.yaml` image.

```bash
# When change image done.
docker compose up -d
```

## Troubleshooting

### CTranslate2 make error

reference: https://github.com/arlo-phoenix/CTranslate2-rocm/issues/10#issuecomment-2751068648

### torch.load error

set env `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`  
reference: https://github.com/ultralytics/yolov5/issues/13513#issuecomment-3104647609

### miopen error

set env `ROCM_PATH=/opt/rocm`  
reference: https://github.com/pytorch/pytorch/issues/160141#issuecomment-3294278253

## References

[CTranslate2-rocm](https://github.com/arlo-phoenix/CTranslate2-rocm)  
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)

