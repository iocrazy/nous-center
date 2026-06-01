"""ComfyUI 调度器(scheduler)的 sigma 计算 —— port 自 ComfyUI comfy/samplers.py。

为什么:diffusers FlowMatchEulerDiscreteScheduler 原生只支持 4 个 sigma 选项
(normal / use_karras_sigmas / use_exponential_sigmas / use_beta_sigmas)。ComfyUI 有 9 个
(+ simple / sgm_uniform / ddim_uniform / linear_quadratic / kl_optimal)。要对齐 ComfyUI,
缺的 5 个手动算 sigma list,经 Flux2KleinPipeline.__call__(sigmas=...)注入。

**纯 stdlib(math + 列表)实现,不依赖 numpy/scipy** —— CI 的 test venv 走 `uv sync --frozen`
不装 inference extra(无 numpy/torch/scipy),本模块要能在 CI import(test_sigma_schedules.py
纯数值对比可在 CI 跑)。

忠实复刻(对齐前读真源,见 feedback_read_comfyui_source):
  - ComfyUI comfy/model_sampling.py:ModelSamplingDiscreteFlow(flow-match sigma 表 +
    time_snr_shift(shift, t)= sigma);Flux2/Anima 用 shift=3.0, multiplier=1.0。
  - ComfyUI comfy/samplers.py:SCHEDULER_HANDLERS 的各 sigma 函数。
数值经 ComfyUI venv ground-truth 逐 scheduler 对齐验证(test_sigma_schedules.py)。

注:diffusers pipe(sigmas=...)期望**不含末尾 0**的序列(它自己 append 0);ComfyUI scheduler
返回**含末尾 0**(steps+1 个)。本模块返回含末尾 0(对齐 ComfyUI ground truth 便于测试),
注入 pipe 前由调用方去掉末尾 0(见 image_modular 注入点)。
"""
from __future__ import annotations

import math


def _time_snr_shift(shift: float, t: float) -> float:
    """ComfyUI model_sampling.time_snr_shift。flow-match:t∈[0,1] → shifted sigma。"""
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    """np.linspace 等价(endpoint=True)。num==1 → [start]。"""
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


class FlowModelSampling:
    """复刻 ComfyUI ModelSamplingDiscreteFlow(flow-match)的 sigma 表 + 转换。

    Flux2/Anima:shift=3.0, multiplier=1.0。sigmas[i]=time_snr_shift(shift,(i+1)/1000)。
    """

    def __init__(self, shift: float = 3.0, multiplier: float = 1.0, timesteps: int = 1000):
        self.shift = float(shift)
        self.multiplier = float(multiplier)
        # ts = (arange(1,T+1)/T)*mult;sigmas = time_snr_shift(shift, ts/mult) = ((i+1)/T)
        self.sigmas: list[float] = [
            _time_snr_shift(self.shift, (i + 1) / timesteps) for i in range(timesteps)
        ]

    @property
    def sigma_min(self) -> float:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> float:
        return self.sigmas[-1]

    def timestep(self, sigma: float) -> float:
        return sigma * self.multiplier

    def sigma(self, t: float) -> float:
        return _time_snr_shift(self.shift, t / self.multiplier)


# ---- 9 个 scheduler 的 sigma 算法(port ComfyUI,返回含末尾 0 的 list)----


def _normal(ms: FlowModelSampling, steps: int, sgm: bool = False) -> list[float]:
    start = ms.timestep(ms.sigma_max)
    end = ms.timestep(ms.sigma_min)
    append_zero = True
    if sgm:
        timesteps = _linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(ms.sigma(end), 0.0, abs_tol=1e-5):
            steps += 1
            append_zero = False
        timesteps = _linspace(start, end, steps)
    sigs = [ms.sigma(t) for t in timesteps]
    if append_zero:
        sigs += [0.0]
    return sigs


def _simple(ms: FlowModelSampling, steps: int) -> list[float]:
    s = ms.sigmas
    sigs = []
    ss = len(s) / steps
    for x in range(steps):
        sigs.append(float(s[-(1 + int(x * ss))]))
    sigs += [0.0]
    return sigs


def _ddim_uniform(ms: FlowModelSampling, steps: int) -> list[float]:
    s = ms.sigmas
    x = 1
    if math.isclose(float(s[x]), 0.0, abs_tol=1e-5):
        steps += 1
        sigs: list[float] = []
    else:
        sigs = [0.0]
    ss = max(len(s) // steps, 1)
    while x < len(s):
        sigs.append(float(s[x]))
        x += ss
    sigs = sigs[::-1]
    return sigs


def _karras(ms: FlowModelSampling, steps: int, rho: float = 7.0) -> list[float]:
    sigma_min, sigma_max = ms.sigma_min, ms.sigma_max
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigs = [
        (max_inv_rho + r * (min_inv_rho - max_inv_rho)) ** rho
        for r in _linspace(0.0, 1.0, steps)
    ]
    return sigs + [0.0]


def _exponential(ms: FlowModelSampling, steps: int) -> list[float]:
    sigma_min, sigma_max = ms.sigma_min, ms.sigma_max
    logs = _linspace(math.log(sigma_max), math.log(sigma_min), steps)
    return [math.exp(x) for x in logs] + [0.0]


def _beta(ms: FlowModelSampling, steps: int, alpha: float = 0.6, beta: float = 0.6) -> list[float]:
    # beta 生产走 diffusers use_beta_sigmas;本地参考实现依赖 scipy(CI 不装),lazy import。
    from scipy import stats  # noqa: PLC0415

    total = len(ms.sigmas) - 1
    ts = [1 - x for x in _linspace_no_endpoint(0.0, 1.0, steps)]
    ts = [round(stats.beta.ppf(t, alpha, beta) * total) for t in ts]
    sigs: list[float] = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs.append(float(ms.sigmas[int(t)]))
        last_t = t
    sigs += [0.0]
    return sigs


def _linspace_no_endpoint(start: float, stop: float, num: int) -> list[float]:
    """np.linspace(endpoint=False)。"""
    step = (stop - start) / num
    return [start + step * i for i in range(num)]


def _linear_quadratic(ms: FlowModelSampling, steps: int,
                      threshold_noise: float = 0.025, linear_steps: int | None = None) -> list[float]:
    if steps == 1:
        sched = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        thr_diff = linear_steps - threshold_noise * steps
        quad_steps = steps - linear_steps
        quad_coef = thr_diff / (linear_steps * quad_steps ** 2)
        lin_coef = threshold_noise / linear_steps - 2 * thr_diff / (quad_steps ** 2)
        const = quad_coef * (linear_steps ** 2)
        quad = [quad_coef * (i ** 2) + lin_coef * i + const for i in range(linear_steps, steps)]
        sched = linear + quad + [1.0]
        sched = [1.0 - x for x in sched]
    return [x * ms.sigma_max for x in sched]


def _kl_optimal(ms: FlowModelSampling, steps: int) -> list[float]:
    n = steps
    atan_min = math.atan(ms.sigma_min)
    atan_max = math.atan(ms.sigma_max)
    sigmas = [0.0] * (n + 1)
    for i in range(n):
        adj = i / (n - 1)
        sigmas[i] = math.tan(adj * atan_min + (1 - adj) * atan_max)
    return sigmas


SIGMA_SCHEDULES = {
    "normal": lambda ms, steps: _normal(ms, steps, sgm=False),
    "sgm_uniform": lambda ms, steps: _normal(ms, steps, sgm=True),
    "simple": _simple,
    "ddim_uniform": _ddim_uniform,
    "karras": _karras,
    "exponential": _exponential,
    "beta": _beta,
    "linear_quadratic": _linear_quadratic,
    "kl_optimal": _kl_optimal,
}

# 这 5 个 diffusers FlowMatch 原生不支持(无 use_*_sigmas flag),必须手动算 sigma 注入。
INJECTED_SCHEDULERS = {"simple", "sgm_uniform", "ddim_uniform", "linear_quadratic", "kl_optimal"}

# 这 4 个 diffusers 原生支持(use_*_sigmas);image_modular 走 _apply_scheduler。
NATIVE_SCHEDULERS = {"normal", "karras", "exponential", "beta"}


def compute_sigmas(scheduler: str, steps: int, shift: float = 3.0) -> list[float]:
    """返回该 scheduler 的 sigma list(含末尾 0,对齐 ComfyUI ground truth,共 steps+1 个)。"""
    fn = SIGMA_SCHEDULES.get(scheduler)
    if fn is None:
        raise ValueError(f"未知 scheduler: {scheduler!r}(支持:{sorted(SIGMA_SCHEDULES)})")
    ms = FlowModelSampling(shift=shift)
    return fn(ms, steps)
