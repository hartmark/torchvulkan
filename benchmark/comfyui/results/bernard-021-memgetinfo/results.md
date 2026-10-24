# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | rocm | vulkan |
|---|---|---|
| status | ok | ok |
| images OK | 10 | 10 |
| device | AMD Radeon Graphics (gfx1101) | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 30.7 | 83.4 |
| warm image median, s | 19.65 | 80.67 |
| warm image min / max, s | 19.56 / 20.54 | 48.09 / 81.67 |
| sampling, s/it (median) | 2.25 | 9.25 |
| all images, s | 209.9 | 708.1 |
| GPU busy avg, % | 91.6 | 87.3 |
| VRAM start / peak, MB | 2478.0 / 15093.0 | 2478.0 / 15731.0 |
| GTT peak, MB | 270.0 | 4720.0 |
| process RSS peak, MB | 15435.0 | 12873.0 |
| host MemAvailable min, MB | 36013.0 | 34322.0 |
| ComfyUI CPU avg / max, % of one core | 203.6 / 531.5 | 13.6 / 322.7 |
| host CPU avg, % | 14.0 | 6.0 |
| GPU edge temp start / avg / max, °C | 42.0 / 58.7 / 63.0 | 54.0 / 59.4 / 64.0 |
| GPU junction temp avg / max, °C | 80.5 / 89.0 | 76.7 / 93.0 |
| GPU mem temp avg / max, °C | 74.0 / 78.0 | 76.5 / 82.0 |
| GPU power avg / max, W | 206.1 / 257.0 | 161.8 / 308.0 |
| GPU sclk avg, MHz | 2168.3 | 2124.3 |
| GPU mclk / fclk avg, MHz | 1218.0 / 2301.0 | 1218.0 / 2301.0 |
| CPU temp max, °C | 78.0 | 79.0 |

**vulkan CPU fallbacks (every occurrence logged, 10 images):** none

rocm versions: backend=rocm, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+rocm7.1, torchvision 0.28.0+rocm7.1, triton-rocm 3.7.1  
vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.1  
