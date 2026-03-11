https://www.sebastianaaltonen.com/blog/no-graphics-api

PCIe **Resize BAR（Base Address Register）** 是一种现代 PCI Express 特性，用来让 **CPU 可以直接访问 GPU 显存（VRAM）** 的更大部分，而无需通过传统的 256 MB 或 4 GB 地址窗口限制。它是现代 GPU 和显卡性能优化中非常关键的一环。下面我详细说明：

---

## 1. **背景 — 为什么需要 Resize BAR**

传统 PCIe 显卡访问 VRAM 的方式：

- 显卡在系统总线上映射自己的显存区域（BAR），CPU 通过 PCIe 读取/写入这块地址空间。
    
- **传统 BAR 大小有限**，通常 256 MB（早期）或 4 GB（新卡），CPU 想访问超过这个容量的显存时，需要分块映射和多次切换。
    
- **问题**：频繁切换 BAR 或多次映射会产生 **overhead**，限制 GPU 的大规模数据访问，尤其在 GPU 驱动直接操作显存（如 GPU driven rendering 或 compute workloads）时。
    

Resizeable BAR 就是为了解决这个问题。

---

## 2. **基本原理**

Resize BAR 允许：

1. 显卡在 PCIe 配置空间里声明一个 **可调整大小的 BAR**。
    
2. 系统 BIOS/UEFI 或操作系统支持时，可以把这个 BAR 的大小从传统固定值（256 MB）扩展到显存总量（比如 8 GB、16 GB）。
    
3. CPU 可以 **一次性直接访问整个显存空间**，无需多次切换窗口。
    

简单理解：

> 以前 CPU 访问 GPU VRAM 只能开小窗，现在开了整块门，CPU 可以直接访问 GPU 整个显存。

---

## 3. **技术细节**

- **实现方式**：PCIe 规范支持的 BAR 地址范围是 64 位的。Resize BAR 利用 **PCIe 2.0+ 或 PCIe 3.0/4.0 的 64-bit BAR**，并通过 **PCIe配置空间**让系统知道 BAR 大小可以调整。
    
- **需要硬件支持**：
    
    - GPU / 显卡需要支持 Resizable BAR
        
    - 主板 BIOS/UEFI 需要支持
        
    - CPU 和操作系统驱动需要支持
        
- **性能提升**：在大数据量 GPU driven workloads 下，可以显著降低 CPU → GPU 数据传输延迟，减少 PCIe overhead。
    

---

## 4. **和 “Unified Memory / GPU Direct Access” 的关系**

- Resize BAR 是实现 **GPU 可直接访问 CPU 内存或 CPU 可直接访问 GPU 内存** 的底层机制之一。
    
- 像文章里提到的 “No Graphics API” 架构，假设 CPU 可以直接通过指针访问 GPU 内存，就很依赖 **Resize BAR 或类似硬件特性**。
    
- 配合现代 GPU coherent cache（缓存一致性）和 UMA（统一内存架构），就可以让 GPU → CPU → GPU 数据传输几乎无开销。
    

---

**总结**：

|特性|作用|
|---|---|
|PCIe BAR|GPU 映射到系统总线的显存地址窗口|
|Resizeable BAR|CPU 可以一次性访问整个显存，减少窗口切换开销|
|优点|减少 CPU-GPU 数据传输延迟，提高 GPU driven workload 性能|
|硬件要求|GPU + 主板 BIOS + CPU + 驱动支持|

---

如果你需要，我可以画一张图，把 **传统 BAR 和 Resize BAR 的访问对比** 可视化，让理解更直观。

你希望我画吗？