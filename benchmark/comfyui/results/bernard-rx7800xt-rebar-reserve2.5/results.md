# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 5 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory --reserve-vram 2.5`.

| | vulkan |
|---|---|
| status | ok |
| images OK | 5 |
| device | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 93.21 |
| warm image median, s | 74.05 |
| warm image min / max, s | 61.63 / 79.65 |
| sampling, s/it (median) | 7.19 |
| all images, s | 382.6 |
| GPU busy avg, % | 74.7 |
| VRAM start / peak, MB | 2173.0 / 15680.0 |
| GTT peak, MB | 4889.0 |
| process RSS peak, MB | 15211.0 |
| host MemAvailable min, MB | 29790.0 |
| ComfyUI CPU avg / max, % of one core | 37.0 / 410.1 |
| host CPU avg, % | 7.1 |
| GPU edge temp start / avg / max, °C | 48.0 / 54.0 / 60.0 |
| GPU junction temp avg / max, °C | 70.4 / 88.0 |
| GPU mem temp avg / max, °C | 71.8 / 78.0 |
| GPU power avg / max, W | 132.4 / 306.0 |
| GPU sclk avg, MHz | 1842.5 |
| GPU mclk / fclk avg, MHz | – / – |
| CPU temp max, °C | 77.0 |

**vulkan CPU fallbacks logged:** `Silent fallback detected for operation: aten::add.out. Set TORCHVULKAN_STRICT=1 to enable strict mode.`

vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0+budgetfix  
