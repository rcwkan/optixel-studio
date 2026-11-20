// Lightweight Windows DXGI helper for querying the active high-performance GPU name,
// dedicated VRAM capacity, and current local VRAM budget/usage across NVIDIA, AMD, and Intel.
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_6.h>

extern "C" int ol_dxgi_gpu_info(
    char *name_buf,
    int name_cap,
    unsigned int *out_total_mb,
    unsigned int *out_free_mb,
    unsigned int *out_budget_mb) {
  if (out_total_mb) *out_total_mb = 0;
  if (out_free_mb) *out_free_mb = 0;
  if (out_budget_mb) *out_budget_mb = 0;
  if (name_buf && name_cap > 0) name_buf[0] = '\0';

  IDXGIFactory1 *factory1 = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&factory1))) || !factory1) {
    return 0;
  }

  IDXGIAdapter1 *best = nullptr;
  DXGI_ADAPTER_DESC1 best_desc = {};

  IDXGIFactory6 *factory6 = nullptr;
  if (SUCCEEDED(factory1->QueryInterface(__uuidof(IDXGIFactory6), reinterpret_cast<void **>(&factory6))) && factory6) {
    for (UINT idx = 0;; ++idx) {
      IDXGIAdapter1 *cand = nullptr;
      if (factory6->EnumAdapterByGpuPreference(
              idx, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, __uuidof(IDXGIAdapter1), reinterpret_cast<void **>(&cand)) == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      if (!cand) continue;
      DXGI_ADAPTER_DESC1 desc = {};
      if (SUCCEEDED(cand->GetDesc1(&desc)) && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
        best = cand;
        best_desc = desc;
        break;
      }
      cand->Release();
    }
    factory6->Release();
  }

  if (!best) {
    SIZE_T best_vram = 0;
    for (UINT idx = 0;; ++idx) {
      IDXGIAdapter1 *cand = nullptr;
      if (factory1->EnumAdapters1(idx, &cand) == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      if (!cand) continue;
      DXGI_ADAPTER_DESC1 desc = {};
      if (SUCCEEDED(cand->GetDesc1(&desc)) && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
        if (!best || desc.DedicatedVideoMemory > best_vram) {
          if (best) best->Release();
          best = cand;
          best_desc = desc;
          best_vram = desc.DedicatedVideoMemory;
          continue;
        }
      }
      cand->Release();
    }
  }

  factory1->Release();
  if (!best) {
    return 0;
  }

  if (name_buf && name_cap > 0) {
    int n = WideCharToMultiByte(CP_UTF8, 0, best_desc.Description, -1, name_buf, name_cap, nullptr, nullptr);
    if (n <= 0) name_buf[0] = '\0';
  }
  const unsigned long long mib = 1024ULL * 1024ULL;
  if (out_total_mb) {
    *out_total_mb = static_cast<unsigned int>(best_desc.DedicatedVideoMemory / mib);
  }

  IDXGIAdapter3 *adapter3 = nullptr;
  if (SUCCEEDED(best->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void **>(&adapter3))) && adapter3) {
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
      const unsigned long long budget_mb = info.Budget / mib;
      const unsigned long long used_mb = info.CurrentUsage / mib;
      if (out_budget_mb) *out_budget_mb = static_cast<unsigned int>(budget_mb);
      if (out_free_mb) *out_free_mb = static_cast<unsigned int>(budget_mb > used_mb ? (budget_mb - used_mb) : 0ULL);
    }
    adapter3->Release();
  }

  best->Release();
  return 1;
}

#else

extern "C" int ol_dxgi_gpu_info(
    char *name_buf,
    int name_cap,
    unsigned int *out_total_mb,
    unsigned int *out_free_mb,
    unsigned int *out_budget_mb) {
  if (out_total_mb) *out_total_mb = 0;
  if (out_free_mb) *out_free_mb = 0;
  if (out_budget_mb) *out_budget_mb = 0;
  if (name_buf && name_cap > 0) name_buf[0] = '\0';
  return 0;
}

#endif
