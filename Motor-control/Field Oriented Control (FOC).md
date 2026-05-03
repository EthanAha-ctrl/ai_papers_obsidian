Field Oriented Control (FOC) 是现代 motor control 领域中最为 core 且 powerful 的 control strategy 之一。为了 build your intuition，我们将从 First Principle 出发，剥离复杂的术语，直击其物理本质。

### 核心直觉：从交流到直流的“视角转换”

FOC 的本质其实非常简单：**我们想要像控制 DC Motor 那样简单地控制 AC Motor (PMSM or Induction Motor)。**

在 DC Motor 中，field (excitation) 和 torque (armature) 是 natural decoupled 的。你可以独立地控制励磁电流（产生磁场）和电枢电流（产生力矩）。但在 AC Motor 中，stator 的三相电流 ($I_a, I_b, I_c$) 同时负责产生磁场和力矩，它们纠缠在一起，且随着 rotor rotation 不断变化，这使得 control 变得极其困难。

FOC 通过一系列 coordinate transformation，将这个旋转的、纠缠的 AC system，“凝固”成了一个静止的、解耦的 DC-like system。

---

### 第一性原理推导：转矩是如何产生的？

从 Lorentz Force Law 出发，motor 的 torque 本质上是 stator 磁场矢量 ($\vec{\psi}_s$) 和 rotor 磁场矢量 ($\vec{\psi}_r$) 相互作用的结果。

$$ T_e = k \cdot |\vec{\psi}_s| \cdot |\vec{\psi}_r| \cdot \sin(\delta) $$

其中：
*   $T_e$：Electromagnetic Torque (电磁转矩)。
*   $k$：Machine constant (电机常数)。
*   $\delta$：Torque angle (转矩角)，即 stator 磁场和 rotor 磁场之间的夹角。

**Intuition:** 当 $\delta = 90^\circ$ 时，$\sin(90^\circ) = 1$，Torque 达到最大值。这意味着，为了获得最高的 efficiency 和 torque density，我们必须时刻保持 stator 磁场矢量与 rotor 磁场矢量垂直 (正交)。

这就是 FOC 的终极目标：**无论 rotor 转到哪里，都要控制 stator 磁场矢量始终垂直于 rotor 磁场矢量。**

---

### FOC 的技术架构与数学详解

为了实现上述目标，FOC 引入了两个关键的数学变换。

#### 1. Clarke Transformation ($abc \rightarrow \alpha\beta$): 降维

物理世界中，motor 有三根线 ($a, b, c$)，电流在空间上互差 $120^\circ$。这是一个 3D system。我们首先将其降维到 2D system，方便计算。

假设三相电流为：
$$ I_a = I_m \sin(\omega t) $$
$$ I_b = I_m \sin(\omega t - 120^\circ) $$
$$ I_c = I_m \sin(\omega t + 120^\circ) $$

通过 Clarke Transform，我们将 $abc$ 坐标系投影到静止的 $\alpha\beta$ 坐标系：

$$
\begin{bmatrix} I_\alpha \\ I_\beta \\ I_0 \end{bmatrix} = \frac{2}{3}
\begin{bmatrix}
1 & -\frac{1}{2} & -\frac{1}{2} \\
0 & \frac{\sqrt{3}}{2} & -\frac{\sqrt{3}}{2} \\
\frac{1}{2} & \frac{1}{2} & \frac{1}{2}
\end{bmatrix}
\begin{bmatrix} I_a \\ I_b \\ I_c \end{bmatrix}
$$

**变量解析：**
*   $I_\alpha$：静止坐标系中，与 a 轴重合的电流分量。
*   $I_\beta$：静止坐标系中，与 a 轴垂直 ($90^\circ$) 的电流分量。
*   $I_0$：Zero-sequence component (零序分量)，在三相对称系统中通常为 0，可忽略。
*   Factor $\frac{2}{3}$：这是 amplitude-invariant (恒幅变换) 的系数，保证了变换前后矢量的幅值不变（也有功率不变变换系数不同，这里以恒幅为例）。

**结果：** 现在我们有了一个在静止平面上旋转的矢量 ($I_\alpha, I_\beta$)，但它依然是个 AC quantity (随着 rotor 转动在旋转)，直接控制依然困难。

#### 2. Park Transformation ($\alpha\beta \rightarrow dq$): 坐标系的“共舞”

这是 FOC 最核心的魔法。既然 stator 磁场在旋转，我们不如把坐标系也跟着 rotor 一起旋转。这就好比你在看旋转木马，站在地上看（$\alpha\beta$）是晕的，但坐在木马上看（$dq$），木马就是静止的。

我们需要知道 rotor 的实时位置 $\theta_e$ (Electrical Angle)。

$$
\begin{bmatrix} I_d \\ I_q \end{bmatrix} =
\begin{bmatrix}
\cos(\theta_e) & \sin(\theta_e) \\
-\sin(\theta_e) & \cos(\theta_e)
\end{bmatrix}
\begin{bmatrix} I_\alpha \\ I_\beta \end{bmatrix}
$$

**变量解析：**
*   $\theta_e$：Electrical rotor angle (电角度)。注意 $\theta_e = p \times \theta_m$，其中 $p$ 是 pole pairs (极对数)，$\theta_m$ 是机械角度。
*   $d$-axis (Direct axis)：与 rotor 磁场方向一致的轴。
*   $q$-axis (Quadrature axis)：与 rotor 磁场方向垂直 ($90^\circ$) 的轴。
*   $I_d$：Flux component (励磁分量)。它负责产生磁场。对于 PMSM，如果不需要弱磁控制，通常设定 $I_d^{ref} = 0$。
*   $I_q$：Torque component (转矩分量)。它产生力矩，类似于 DC motor 的 armature current。

**物理意义：**
通过 Park Transform，原本随时间正弦变化的 AC currents ($I_a, I_b, I_c$)，在 $dq$ 坐标系下变成了 **constant DC values**。现在，我们可以使用简单的 PI Controller 来精确控制 $I_d$ 和 $I_q$，就像控制 DC Motor 一样。

---

### FOC 控制环路详解

FOC 的 control loop 通常包含以下层次：

**1. Speed/Position Loop (Outer Loop):**
*   Input: Target speed ($\omega_{ref}$) vs Actual speed ($\omega_{fb}$).
*   Controller: PI Controller.
*   Output: Target torque current $I_q^{ref}$.
*   Logic: 如果 speed 不够，就增加 $I_q$。

**2. Current Loop (Inner Loop):**
这是 FOC 的 heart。我们需要控制 $I_d$ 和 $I_q$ 追踪其参考值。
*   Input: $I_d^{ref}$ (通常为 0), $I_q^{ref}$ (来自速度环) vs Feedback $I_d, I_q$.
*   Controller: Two independent PI Controllers.
    *   $PI_d$: Error $e_d = I_d^{ref} - I_d \rightarrow V_d$.
    *   $PI_q$: Error $e_q = I_q^{ref} - I_q \rightarrow V_q$.
*   **公式解析 (PI Controller):**
    $$ V_{out}(t) = K_p e(t) + K_i \int e(t) dt $$
    其中 $K_p$ 决定 response speed，$K_i$ 决定 steady-state error elimination。

**3. Inverse Park Transformation:**
PI Controller 输出的是 $dq$ 坐标系下的电压 $V_d, V_q$。我们需要把它们变回去，告诉 Inverter 具体该怎么开关。
$$
\begin{bmatrix} V_\alpha \\ V_\beta \end{bmatrix} =
\begin{bmatrix}
\cos(\theta_e) & -\sin(\theta_e) \\
\sin(\theta_e) & \cos(\theta_e)
\end{bmatrix}
\begin{bmatrix} V_d \\ V_q \end{bmatrix}
$$

**4. Space Vector PWM (SVPWM):**
这是最后执行层面的 magic。Inverter 有 6 个 switches (IGBTs/MOSFETs)，只能输出 6 个 active vectors 和 2 个 zero vectors。
*   **原理：** 利用脉宽调制，合成任意方向的电压矢量 $\vec{V}_{ref}$。
*   **扇区判断：** 根据 $V_\alpha, V_\beta$ 判断矢量落在哪个 Sector (1-6)。
*   **作用时间计算：**
    假设落在 Sector 1，由 $V_1 (100)$ 和 $V_2 (110)$ 合成：
    $$ T_1 = \frac{\sqrt{3} T_s V_{ref} \sin(\frac{\pi}{3} - \theta)}{V_{dc}} $$
    $$ T_2 = \frac{\sqrt{3} T_s V_{ref} \sin(\theta)}{V_{dc}} $$
    $$ T_0 = T_s - T_1 - T_2 $$
    其中 $T_s$ 是 PWM 周期，$V_{dc}$ 是直流母线电压。
*   **优势：** SVPWM 相比 Sinusoidal PWM (SPWM) 有约 15% 的 higher DC bus voltage utilization ($V_{max} = \frac{V_{dc}}{\sqrt{3}}$)。

---

### 实验数据表参考

为了直观展示 FOC 的优势，对比传统的 Scalar Control (V/f)：

| Feature | Scalar Control (V/f) | FOC (Vector Control) |
| :--- | :--- | :--- |
| **Control Variable** | Voltage Magnitude & Frequency | Current Vector ($I_d, I_q$) |
| **Torque Response** | Slow (受 slip frequency 限制) | Fast (Current loop bandwidth 极高) |
| **Torque Ripple** | High (尤其在 low speed) | Low (Smooth operation) |
| **Efficiency** | Moderate | High (Optimal $I_d=0$ control) |
| **Low Speed Performance**| Poor (Torque collapse) | Excellent (Full torque at 0 speed) |
| **Dynamic Performance** | Open-loop like | Closed-loop, high bandwidth |
| **Computational Cost** | Low | High (需要 DSP/FPGA 进行 Matrix calc) |

### Intuition 总结：构建你的思维模型

想象你在遛狗。
*   **Scalar Control:** 你只能控制绳子的长度和狗跑的速度，但狗想往哪跑你管不了，只能顺着它。如果狗突然停下来，绳子就松了。
*   **FOC:** 你不仅知道狗的速度，还实时知道狗的位置和朝向。你可以精确地施加一个垂直于狗跑方向的力，让狗始终围绕你转圈，且始终保持最大的拉力。

FOC 通过数学变换，让 controller 始终“看着”rotor 的磁场方向，并精确地施加一个垂直的力。这就是为什么无人机可以瞬间起飞，电动汽车可以瞬间起步——因为 controller 在微秒级别地调整着 $I_q$，始终维持着最优的 $90^\circ$ 推力。

### References & Further Reading

1.  **TI Application Note:** "Field Oriented Control of 3-Phase AC Motors"
    *   Link: [www.ti.com/lit/an/bpra073/bpra073.pdf](https://www.ti.com/lit/an/bpra073/bpra073.pdf) (Excellent technical depth on transformations).
2.  **STMicroelectronics (ST) AN1939:** "Sensor Field Oriented Control (FOC) of a PMSM"
    *   Link: [www.st.com/resource/en/application_note/dm00100359.pdf](https://www.st.com/resource/en/application_note/dm00100359.pdf) (Great for implementation details).
3.  **MathWorks Documentation:** "Field-Oriented Control"
    *   Link: [www.mathworks.com/discovery/field-oriented-control.html](https://www.mathworks.com/discovery/field-oriented-control.html) (Good for simulation and block diagrams).
4.  **Wikipedia:** "Vector control (motor)"
    *   Link: [en.wikipedia.org/wiki/Vector_control_(motor)](https://en.wikipedia.org/wiki/Vector_control_(motor)) (General overview).