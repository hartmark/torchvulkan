# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 5 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | vulkan |
|---|---|
| status | ok |
| images OK | 5 |
| device | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 110.94 |
| warm image median, s | 96.61 |
| warm image min / max, s | 96.21 / 98.72 |
| sampling, s/it (median) | 9.5 |
| all images, s | 499.1 |
| GPU busy avg, % | 76.9 |
| VRAM start / peak, MB | 3268.0 / 15556.0 |
| GTT peak, MB | 5046.0 |
| process RSS peak, MB | 15055.0 |
| host MemAvailable min, MB | 28773.0 |
| ComfyUI CPU avg / max, % of one core | 33.1 / 356.8 |
| host CPU avg, % | 9.3 |
| GPU edge temp start / avg / max, °C | 56.0 / 56.9 / 61.0 |
| GPU junction temp avg / max, °C | 72.0 / 85.0 |
| GPU mem temp avg / max, °C | 73.7 / 76.0 |
| GPU power avg / max, W | 140.1 / 299.0 |
| GPU sclk avg, MHz | 1936.6 |
| CPU temp max, °C | 77.0 |

**vulkan CPU fallbacks (every occurrence logged, 5 images):** `aten::silu.out` × 140; `aten::add.out` × 36

vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0+budgetfix  
