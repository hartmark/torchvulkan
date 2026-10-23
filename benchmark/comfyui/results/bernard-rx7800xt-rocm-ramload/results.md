# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | rocm |
|---|---|
| status | ok |
| images OK | 10 |
| device | AMD Radeon Graphics (gfx1101) |
| first image (cold), s | 35.24 |
| warm image median, s | 20.22 |
| warm image min / max, s | 20.07 / 21.04 |
| sampling, s/it (median) | 2.25 |
| all images, s | 218.7 |
| GPU busy avg, % | 88.6 |
| VRAM start / peak, MB | 2128.0 / 14738.0 |
| GTT peak, MB | 201.0 |
| process RSS peak, MB | 14170.0 |
| host MemAvailable min, MB | 26009.0 |
| ComfyUI CPU avg / max, % of one core | 201.7 / 558.8 |
| host CPU avg, % | 14.2 |
| GPU edge temp start / avg / max, °C | 50.0 / 60.8 / 64.0 |
| GPU junction temp avg / max, °C | 81.9 / 90.0 |
| GPU mem temp avg / max, °C | 76.3 / 80.0 |
| GPU power avg / max, W | 196.7 / 279.0 |
| GPU sclk avg, MHz | 2108.2 |
| CPU temp max, °C | – |


rocm versions: backend=rocm, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+rocm7.1, torchvision 0.28.0+rocm7.1, triton-rocm 3.7.1  
