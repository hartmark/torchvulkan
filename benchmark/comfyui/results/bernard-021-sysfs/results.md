# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | vulkan |
|---|---|
| status | ok |
| images OK | 10 |
| device | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 78.89 |
| warm image median, s | 77.16 |
| warm image min / max, s | 64.13 / 82.67 |
| sampling, s/it (median) | 8.88 |
| all images, s | 754.7 |
| GPU busy avg, % | 89.4 |
| VRAM start / peak, MB | 2481.0 / 15731.0 |
| GTT peak, MB | 4976.0 |
| process RSS peak, MB | 13725.0 |
| host MemAvailable min, MB | 30157.0 |
| ComfyUI CPU avg / max, % of one core | 12.1 / 333.0 |
| host CPU avg, % | 5.8 |
| GPU edge temp start / avg / max, °C | 56.0 / 60.1 / 65.0 |
| GPU junction temp avg / max, °C | 76.9 / 94.0 |
| GPU mem temp avg / max, °C | 77.0 / 80.0 |
| GPU power avg / max, W | 159.7 / 312.0 |
| GPU sclk avg, MHz | 2195.2 |
| GPU mclk / fclk avg, MHz | 1218.0 / 2301.0 |
| CPU temp max, °C | 78.0 |

**vulkan CPU fallbacks (every occurrence logged, 10 images):** none

vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.1  
