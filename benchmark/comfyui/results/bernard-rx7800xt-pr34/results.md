# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | vulkan |
|---|---|
| status | ok |
| images OK | 10 |
| device | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 92.96 |
| warm image median, s | 83.19 |
| warm image min / max, s | 56.62 / 103.75 |
| sampling, s/it (median) | 9.5 |
| all images, s | 819.5 |
| GPU busy avg, % | 87.6 |
| VRAM start / peak, MB | 2250.0 / 15704.0 |
| GTT peak, MB | 5104.0 |
| process RSS peak, MB | 13689.0 |
| host MemAvailable min, MB | 31438.0 |
| ComfyUI CPU avg / max, % of one core | 13.9 / 448.6 |
| host CPU avg, % | 10.9 |
| GPU edge temp start / avg / max, °C | 46.0 / 58.5 / 65.0 |
| GPU junction temp avg / max, °C | 75.1 / 94.0 |
| GPU mem temp avg / max, °C | 74.7 / 82.0 |
| GPU power avg / max, W | 157.8 / 321.0 |
| GPU sclk avg, MHz | 2174.4 |
| CPU temp max, °C | 79.0 |

**vulkan CPU fallbacks (every occurrence logged, 10 images):** none

vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0+pr34  
