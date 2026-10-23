# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory --reserve-vram 7`.

| | rocm | vulkan |
|---|---|---|
| status | ok | fail |
| images OK | 10 | 2 |
| device | AMD Radeon Graphics (gfx1101) | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 33.82 | 101.02 |
| warm image median, s | 21.04 | 65.63 |
| warm image min / max, s | 21.04 / 21.54 | 65.63 / 65.63 |
| sampling, s/it (median) | 2.25 | 5.12 |
| all images, s | 223.8 | 166.7 |
| GPU busy avg, % | 89.5 | 54.0 |
| VRAM start / peak, MB | 2129.0 / 14053.0 | 2125.0 / 15375.0 |
| GTT peak, MB | 195.0 | 4611.0 |
| process RSS peak, MB | 17073.0 | 13633.0 |
| host MemAvailable min, MB | 34798.0 | 32833.0 |
| ComfyUI CPU avg / max, % of one core | 200.9 / 413.2 | 58.5 / 341.3 |
| host CPU avg, % | 13.9 | 8.2 |
| GPU edge temp start / avg / max, °C | 43.0 / 58.4 / 63.0 | 54.0 / 54.0 / 61.0 |
| GPU junction temp avg / max, °C | 79.6 / 89.0 | 69.9 / 87.0 |
| GPU mem temp avg / max, °C | 74.1 / 78.0 | 73.9 / 78.0 |
| GPU power avg / max, W | 194.5 / 259.0 | 131.2 / 299.0 |
| GPU sclk avg, MHz | 2171.7 | 1263.3 |
| CPU temp max, °C | – | – |

**vulkan error:** `image 3 failed: torch.OutOfMemoryError: torchvulkan [ERROR]: Out of memory on AMD Radeon RX 7800 XT (RADV NAVI32). Tried to allocate 268435456 bytes with 7516192768 of 7635114188 bytes in use.`
**vulkan CPU fallbacks logged:** `Silent fallback detected for operation: aten::add.out. Set TORCHVULKAN_STRICT=1 to enable strict mode.`

rocm versions: backend=rocm, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+rocm7.1, torchvision 0.28.0+rocm7.1, triton-rocm 3.7.1  
vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0  
