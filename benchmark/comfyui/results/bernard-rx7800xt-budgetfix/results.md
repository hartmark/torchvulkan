# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | vulkan |
|---|---|
| status | ok |
| images OK | 10 |
| device | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 91.85 |
| warm image median, s | 54.67 |
| warm image min / max, s | 53.79 / 55.61 |
| sampling, s/it (median) | 5.0 |
| all images, s | 584.7 |
| GPU busy avg, % | 75.3 |
| VRAM start / peak, MB | 2125.0 / 15695.0 |
| GTT peak, MB | 5123.0 |
| process RSS peak, MB | 15447.0 |
| host MemAvailable min, MB | 34036.0 |
| ComfyUI CPU avg / max, % of one core | 39.1 / 314.9 |
| host CPU avg, % | 7.5 |
| GPU edge temp start / avg / max, °C | 45.0 / 57.1 / 63.0 |
| GPU junction temp avg / max, °C | 75.8 / 89.0 |
| GPU mem temp avg / max, °C | 75.4 / 80.0 |
| GPU power avg / max, W | 170.6 / 305.0 |
| GPU sclk avg, MHz | 1787.7 |
| CPU temp max, °C | – |

**vulkan CPU fallbacks logged:** `Silent fallback detected for operation: aten::add.out. Set TORCHVULKAN_STRICT=1 to enable strict mode.`

vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0+budgetfix  
